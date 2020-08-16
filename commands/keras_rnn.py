import asyncio
import glob
import json
import logging
import os
import time
from typing import Tuple, Set, Dict, List

import numpy as np
import tensorflow as tf
from aiohttp import web
from keras import backend as K
from textgenrnn.model import textgenrnn_model

import config as cfg
from ext.response import Response

DATA_PATH = os.path.join(cfg.DATA_PATH, "Keras")
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
        with open(get_config_path(m), 'r') as infile:
            model_data.append(json.load(infile))
    resp.data = model_data
    return resp.web_response


async def handler_be_list_all(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    model_data = []
    for m in sorted(os.listdir(DATA_PATH)):
        with open(get_config_path(m), 'r') as infile:
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
        resp.data = await loop.run_in_executor(None, lambda: run_guess(**params))
        resp.time = time.perf_counter() - start
    except Exception as e:
        logger.exception('guess run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


def run_be(model: str, words: int = 50, temp: float = 0.5) -> Tuple[str, int]:
    # Load configs
    with open(get_config_path(model), 'r', encoding='utf8', errors='ignore') as json_file:
        config = json.load(json_file)
    with open(get_vocab_path(model), 'r', encoding='utf8', errors='ignore') as json_file:
        vocab = json.load(json_file)
    # Prepare vars
    num_classes = len(vocab) + 1
    indices_char = {v: k for k, v in vocab.items()}
    # Build model
    model = textgenrnn_model(num_classes, cfg=config, weights_path=get_weights_path(model))
    # Config vars
    maxlen = config['max_length']
    # Start with random letter
    # ret_str = np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    ret_str = '\n'
    if len(model.inputs) > 1:
        model = tf.keras.models.Model(inputs=model.inputs[0], outputs=model.outputs[1])

    num_words = 0
    num_char = 0
    while num_char < (words+50)*6:
        encoded = np.array([vocab.get(x, 0) for x in ret_str])
        encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=maxlen)

        preds = np.asarray(model.predict(encoded_text, batch_size=1)[0]).astype('float64')
        if temp is None or temp == 0.0:
            index = np.argmax(preds)
        else:
            preds = np.log(preds + tf.keras.backend.epsilon()) / temp
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            index = np.argmax(probas)
            # prevent function from being able to choose 0 (placeholder)
            # choose 2nd best index from preds
            if index == 0:
                index = np.argsort(preds)[-2]

        next_char = indices_char[index]
        ret_str += next_char
        num_char += 1
        if next_char == ' ':
            num_words += 1
        # Only stop after new line
        if (num_words >= words) and (next_char == '\n'):
            break

    K.clear_session()
    tf.reset_default_graph()
    return ret_str, num_words


def run_guess(check_models: List[str], in_str: str) -> Dict[str, float]:
    ret_dict: Dict[str, float] = {}
    for name in check_models:
        # Load configs
        with open(get_config_path(name), 'r', encoding='utf8', errors='ignore') as json_file:
            config = json.load(json_file)
        with open(get_vocab_path(name), 'r', encoding='utf8', errors='ignore') as json_file:
            vocab = json.load(json_file)
        # Prepare vars
        num_classes = len(vocab) + 1
        # Build model
        model = textgenrnn_model(num_classes, cfg=config, weights_path=get_weights_path(name))
        # Config vars
        maxlen = config['max_length']
        if len(model.inputs) > 1:
            model = tf.keras.models.Model(inputs=model.inputs[0], outputs=model.outputs[1])

        encoded = np.array([vocab.get(x, 0) for x in in_str[:-1]])
        encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=maxlen)
        preds = np.asarray(model.predict(encoded_text, batch_size=1)[0]).astype('float64')
        pred_next = preds[vocab.get(in_str[-1], 0)]
        ret_dict[name] = pred_next * 100

    K.clear_session()
    tf.reset_default_graph()
    return ret_dict


def get_weights_path(model: str) -> str:
    return os.path.join(DATA_PATH, model, f'{model}_weights.hdf5')


def get_vocab_path(model: str) -> str:
    return os.path.join(DATA_PATH, model, f'{model}_vocab.json')


def get_config_path(model: str) -> str:
    return os.path.join(DATA_PATH, model, f'{model}_config.json')


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
