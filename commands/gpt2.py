import asyncio
import json
import logging
import os
import time

import numpy as np
import tensorflow as tf
from aiohttp import web

import config as cfg
from ext import ArgumentParser, Response
from ext.gpt2 import encoder, model, sample

DATA_PATH = os.path.join(cfg.DATA_PATH, "gpt-2", "models")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def handler_list(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    model_data = {}
    for m in sorted(os.listdir(DATA_PATH)):
        with open(get_info_path(m), 'r') as infile:
            model_data[m] = json.load(infile)
    resp.data = model_data
    return resp.web_response


async def handler_run(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    parser = ArgumentParser()
    parser.add_argument("--raw_text", default=None, type=str, required=True)
    parser.add_argument("--model_name", default="117M", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--length", default=75, type=int)
    parser.add_argument("--temperature", default=0.75, type=float)
    parser.add_argument("--top_k", default=0, type=int)
    try:
        got_params: dict = await r.json()
        # Don't parse None value
        if 'seed' in got_params and got_params['seed'] is None:
            got_params.pop('seed')
        params = vars(parser.parse_kwargs(got_params))
    except Exception as e:
        logger.error(f'continue parameter parsing error: {str(e)}')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    run_model = find_model(params['model_name'])
    if not run_model:
        resp.error = f"No model {params['model_name']} found"
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    params["model_name"] = run_model
    resp.params = params
    try:
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        ret_str = await loop.run_in_executor(None, lambda: run(**params))
        resp.time = time.perf_counter() - start
        resp.data = ret_str
    except Exception as e:
        logger.exception('continue run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


def run(raw_text, model_name, seed, batch_size, length, temperature, top_k) -> str:
    """
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    tf.reset_default_graph()
    enc = get_encoder(model_name)
    hparams = model.default_hparams()
    with open(get_hparams_path(model_name)) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(get_model_path(model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
        return enc.decode(out[0])


def get_model_path(name: str) -> str:
    return os.path.join(DATA_PATH, name)


def get_encoder_path(name: str) -> str:
    return os.path.join(DATA_PATH, name, 'encoder.json')


def get_vocab_path(name: str) -> str:
    return os.path.join(DATA_PATH, name, 'vocab.bpe')


def get_hparams_path(name: str) -> str:
    return os.path.join(DATA_PATH, name, 'hparams.json')


def get_info_path(name: str) -> str:
    return os.path.join(DATA_PATH, name, 'info.json')


def get_encoder(model_name):
    with open(get_encoder_path(model_name), 'r') as f:
        enc_data = json.load(f)
    with open(get_vocab_path(model_name), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return encoder.Encoder(
        encoder=enc_data,
        bpe_merges=bpe_merges,
    )


def find_model(name: str) -> str:
    """Does nothing if model exists, otherwise returns empty string"""
    # Check exact match
    for m in os.listdir(DATA_PATH):
        if m.lower() == name.lower():
            return m
    return ''


ROUTES = [
    web.get("/continue/list", handler_list),
    web.post("/continue/run", handler_run),
]
