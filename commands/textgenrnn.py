import asyncio
import glob
import json
import logging
import os
import time
from typing import Tuple, Set

from aiohttp import web
from textgenrnn.run_utils import rnn_guess, rnn_generate, get_paths

import config as cfg
from ext import Response

DATA_PATH = os.path.join(cfg.DATA_PATH, "textgenrnn")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
loop = asyncio.get_event_loop()


async def handler_be_list_best(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    model_data = []
    best_models = []
    # Find best model and store in dict
    for model in find_all_models():
        best_models.append(find_lowest_loss(model))
    for m in sorted(best_models):
        paths = get_paths(model_name=m, models_dir=DATA_PATH)
        with open(paths['config_path'], 'r') as infile:
            model_data.append(json.load(infile))
    resp.data = model_data
    return resp.web_response


async def handler_be_list_all(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    model_data = []
    for m in sorted(os.listdir(DATA_PATH)):
        paths = get_paths(model_name=m, models_dir=DATA_PATH)
        with open(paths['config_path'], 'r') as infile:
            model_data.append(json.load(infile))
    resp.data = model_data
    return resp.web_response


async def handler_be_run(r: web.Request) -> web.Response:
    logger.debug(r.path)
    params = {}
    resp = Response()
    try:
        params["words"] = int(r.query.get("words", "50"))
        params["temp"] = float(r.query.get("temp", "0.5"))
    except Exception as e:
        logger.exception('be parameter parsing error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    resp.params = params
    model = find_model(r.match_info['model'])
    if not model:
        resp.error = f"No model {r.match_info['model']} found"
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    params["model"] = model
    try:
        start = time.perf_counter()
        ret_str, num_words = await loop.run_in_executor(None, lambda: run_be(**params))
        resp.time = time.perf_counter() - start
        resp.data = dict(text=ret_str, words=num_words)
    except Exception as e:
        logger.exception('be run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


async def handler_guess_run(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    params = {}
    try:
        data = await r.json()
        check_models = data.get('check_models')
        if not check_models or not isinstance(check_models, list):
            raise Exception('List of models (check_models) to check against is required')
        params['in_str'] = data.get('in_str')
        if not params['in_str']:
            raise Exception('Input text (in_str) is required')
    except Exception as e:
        logger.exception('guess parameter parsing error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    # Validate models
    params['check_models'] = []
    for name in check_models:
        model = find_model(name)
        if not model:
            resp.error = f"No model {name} found"
            resp.status = web.HTTPBadRequest.status_code
            return resp.web_response
        params['check_models'].append(model)
    resp.params = params
    try:
        start = time.perf_counter()
        resp.data = await loop.run_in_executor(None, lambda: rnn_guess(models_dir=DATA_PATH, reset=True, **params))
        resp.time = time.perf_counter() - start
    except Exception as e:
        logger.exception('guess run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


def run_be(model: str, words: int = 50, temp: float = 0.5) -> Tuple[str, int]:
    paths = get_paths(model_name=model, models_dir=DATA_PATH)
    paths.pop('model_dir')
    text, words = rnn_generate(**paths, min_words=words, temperature=temp, reset=True)
    return text, words


def find_model(model: str) -> str:
    """Does nothing if model exists exactly, otherwise returns model with lowest loss"""
    # Check exact match
    model = model.lower()
    for m in os.listdir(DATA_PATH):
        if m.lower() == model:
            return model
    # Check best match
    return find_lowest_loss(model)


def find_all_models() -> Set[str]:
    """Returns all base model names in DATA_PATH"""
    ret = set()
    for m in os.listdir(DATA_PATH):
        tmp = m.split('_', 1)
        if len(tmp) == 2:
            ret.add(tmp[0])
    return ret


def find_lowest_loss(model: str) -> str:
    """Returns the model with the lowest loss"""
    loss_dict = {}
    override_dict = {'jens': 'jens_3l128bi'}
    if override_dict.get(model, None) is not None:
        return override_dict[model]
    for cfg_file in glob.glob(os.path.join(DATA_PATH, f'{model}_*/*_config.json')):
        with open(cfg_file, 'r') as infile:
            data = json.load(infile)
            loss = data['loss']
            # Set key to model folder name
            loss_dict[cfg_file.split('/')[-2]] = loss
    # We found nothing
    if not loss_dict:
        return ''
    return min(loss_dict, key=loss_dict.get)


ROUTES = [
    web.get("/be/list/best", handler_be_list_best),
    web.get("/be/list/all", handler_be_list_all),
    web.get("/be/run/{model}", handler_be_run),
    web.post("/guess/run", handler_guess_run),
]
