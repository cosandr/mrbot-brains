import asyncio
import glob
import logging
import os
import time
from base64 import b64decode
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from aiohttp import web

import config as cfg
from ext.parser import Parser
from ext.response import Response
from ext.utils import bytes_from_url

DATA_PATH = os.path.join(cfg.DATA_PATH, "IMG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
loop = asyncio.get_event_loop()


async def handler_list(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response(data={})
    # Did we request a specific category?
    req_cat = r.query.get('category', '')
    if req_cat:
        label_file = os.path.join(DATA_PATH, f'{req_cat}_labels.txt')
        if not os.path.exists(label_file):
            resp.error = f'No category {req_cat} found'
            resp.status = web.HTTPBadRequest.status_code
            return resp.web_response
        resp.data[req_cat] = _load_labels(label_file)
        return resp.web_response
    # Return all categories
    for label_file in glob.glob(os.path.join(DATA_PATH, '*_labels.txt')):
        cat_name = os.path.basename(label_file).split('_labels.txt', 1)[0]
        resp.data[cat_name] = _load_labels(label_file)
    return resp.web_response


async def handler_run(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    parser = Parser()
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--url", default=None, type=str)
    parser.add_argument("--image", default=None, type=str)
    parser.add_argument("--pnas", default=False, type=bool)
    try:
        got_params: dict = await r.json()
        # Don't parse bool
        pnas = got_params.pop('pnas', False)
        params = vars(parser.parse_kwargs(got_params))
        if not isinstance(pnas, bool):
            raise Exception('pnas should be a bool')
        params['pnas'] = pnas
        if not params['url'] and not params['image']:
            raise Exception('Need a URL to an image (url) or base64 encoded image (image)')
    except Exception as e:
        logger.exception('image label parameter parsing error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    # Download or decode image
    try:
        if params['url']:
            params['image'] = await bytes_from_url(params['url'])
            params.pop('url')
        else:
            params['image'] = b64decode(params['image'])
    except Exception as e:
        logger.exception('image decode/download error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response

    try:
        start = time.perf_counter()
        results, labels = await loop.run_in_executor(None, lambda: run(**params))
        resp.data = dict(results=results.tolist(), labels=labels)
        resp.time = time.perf_counter() - start
    except Exception as e:
        logger.exception('image label run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


def run(image: Union[str, bytes], model_type: str, pnas: bool = False) -> Tuple[np.ndarray, List[str]]:
    input_mean = 0
    input_std = 255
    if pnas:
        input_height = 331
        input_width = 331
        model_file = os.path.join(DATA_PATH, f'p{model_type}_graph.pb')
        label_file = os.path.join(DATA_PATH, f'p{model_type}_labels.txt')
    else:
        input_height = 299
        input_width = 299
        model_file = os.path.join(DATA_PATH, f'{model_type}_graph.pb')
        label_file = os.path.join(DATA_PATH, f'{model_type}_labels.txt')
    labels = _load_labels(label_file)
    graph = _load_graph(model_file)
    input_operation = graph.get_operation_by_name("import/Placeholder")
    output_operation = graph.get_operation_by_name("import/final_result")
    t = _read_tensor_from_image_file(image, input_height, input_width, input_mean, input_std)
    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    results = np.squeeze(results)
    return results, labels


def _load_graph(model_file: str) -> tf.Graph:
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def _read_tensor_from_image_file(file_name: Union[str, bytes], input_height: int = 299, input_width: int = 299,
                                 input_mean: int = 0, input_std: int = 255):
    image_reader = tf.image.decode_jpeg(file_name, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    with tf.compat.v1.Session() as sess:
        result = sess.run(normalized)
    return result


def _load_labels(label_file: str):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


ROUTES = [
    web.get("/image_label/list", handler_list),
    web.post("/image_label/run", handler_run),
]
