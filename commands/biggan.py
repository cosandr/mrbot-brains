import asyncio
import json
import logging
import os
import time
from base64 import b64encode
from typing import Dict

from aiohttp import web

import config as cfg
from ext import ArgumentParser, BigGAN, Response
from ext.utils import check_opencv_codec

DATA_PATH = os.path.join(cfg.DATA_PATH, "biggan")
CACHED_GAN: BigGAN = None
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def handler_categories(r: web.Request) -> web.Response:
    """Return list of categories, keys are their names"""
    logger.debug(r.path)
    resp = Response()
    with open(os.path.join(DATA_PATH, 'categories.json'), 'r') as fr:
        # <category name>: <id>
        # <id> is not unique
        categories: Dict[str, int] = json.load(fr)
    resp.data = categories
    return resp.web_response


async def handler_categories_backwards(r: web.Request) -> web.Response:
    """Return list of categories, keys are their IDs"""
    logger.debug(r.path)
    resp = Response()
    with open(os.path.join(DATA_PATH, 'categories_backwards.json'), 'r') as fr:
        # <id>: <category name(s)>
        # Multiple names are delimited by commas
        categories: Dict[str, str] = json.load(fr)
    resp.data = categories
    return resp.web_response


async def handler_run(r: web.Request) -> web.Response:
    logger.debug(r.path)
    get_image = 'get_image' in r.query
    what = r.match_info['what']
    cat_a_name = ''
    cat_b_name = ''
    resp = Response()
    parser = ArgumentParser()
    parser.add_argument("--cat_a", default=None, type=int, required=True)
    parser.add_argument("--cat_b", default=None, type=int)
    parser.add_argument("--video", default=False, type=bool)
    parser.add_argument("--samples", default=12, type=int)
    parser.add_argument("--truncation", default=0.2, type=float)
    parser.add_argument("--noise_a", default=5, type=int)
    parser.add_argument("--noise_b", default=5, type=int)
    parser.add_argument("--interps", default=12, type=int)
    parser.add_argument("--fps", default=1, type=int)
    try:
        got_params: dict = await r.json()
        # Don't parse bool
        video = got_params.pop('video', False)
        params = vars(parser.parse_kwargs(got_params))
        if not isinstance(video, bool):
            raise Exception('Video should be a bool')
        params['video'] = video
        if params["cat_a"] is not None:
            cat_a_name = get_gan_category(params["cat_a"])
            if not cat_a_name:
                raise Exception('Invalid category')
        if what == 'transform':
            if params["cat_b"] is not None:
                cat_b_name = get_gan_category(params["cat_b"])
                if not cat_b_name:
                    raise Exception('Invalid target category')
            else:
                raise Exception('Target category is required')
        gan_size = int(r.match_info['size'])
        if gan_size not in (128, 256, 512):
            raise Exception('Invalid GAN size, choose between 128, 256 and 512')
    except Exception as e:
        logger.error(f'{what} parameter parsing error: {str(e)}')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response

    # Make sure we can actually run video
    if params['video']:
        if not check_opencv_codec(cfg.MP4_CODEC):
            resp.error = f'Codec for {cfg.MP4_CODEC} video is not available, try using GIF instead.'
            resp.status = web.HTTPServerError.status_code
            return resp.web_response
    file_ext = 'mp4' if params['video'] else 'gif'

    if what == 'hell':
        params.pop('noise_b', None)
        params.pop('cat_b', None)
        params.pop('interps', None)
        filename = 'hell_{cat_a}_{samples}s_{truncation}t_{noise_a}na_{fps}fps_{size}.{file_ext}'.format(
            file_ext=file_ext, size=gan_size, **params)
    elif what == 'slerp':
        params.pop('cat_b', None)
        # Cannot reshape array if samples > 1
        filename = 'slerp_{cat_a}_{samples}s_{interps}i_{truncation}t_{noise_a}na_{noise_b}nb_{fps}fps_{size}.{file_ext}'.format(
            file_ext=file_ext, size=gan_size, **params)
    elif what == 'transform':
        filename = '{cat_a}_to_{cat_b}_{samples}s_{interps}i_{truncation}t_{noise_a}na_{noise_b}nb_{fps}fps_{size}.{file_ext}'.format(
            file_ext=file_ext, size=gan_size, **params)
    else:
        resp.error = "Invalid operation, choose one of hell, slerp, transform"
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response

    resp.params = params.copy()
    resp.params['size'] = gan_size
    resp.params['cat_a_name'] = cat_a_name
    if cat_b_name:
        resp.params['cat_b_name'] = cat_b_name

    file_path = os.path.join(cfg.UPLOAD_PATH, filename)
    try:
        start = time.perf_counter()
        if not os.path.exists(file_path):
            loop = asyncio.get_running_loop()
            if not CACHED_GAN or gan_size != CACHED_GAN.biggan_size:
                await loop.run_in_executor(None, lambda: load_biggan(gan_size))
            if what == 'hell':
                logger.debug(f'{what} generating {filename}')
                await loop.run_in_executor(None, lambda: CACHED_GAN.hell(filename=file_path, **params))
            elif what == 'slerp':
                logger.debug(f'{what} generating {filename}')
                await loop.run_in_executor(None, lambda: CACHED_GAN.generate_slerp(filename=file_path, **params))
            elif what == 'transform':
                logger.debug(f'{what} generating {filename}')
                await loop.run_in_executor(None, lambda: CACHED_GAN.transform(filename=file_path, **params))
        resp.data = dict(filename=filename)
        # Respond with URL to image if we have a hostname configured
        if cfg.UPLOAD_URL:
            resp.data['url'] = f'{cfg.UPLOAD_URL}/{filename}'
        # Encode image and send it directly
        if get_image or not cfg.UPLOAD_URL:
            logger.debug(f'Hell encoding {filename}')
            with open(file_path, 'rb') as fr:
                resp.data['image'] = b64encode(fr.read()).decode('utf-8')
        resp.time = time.perf_counter() - start
        if params['video']:
            resp.data['codec'] = cfg.MP4_CODEC
    except Exception as e:
        logger.exception(f'{what} run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


def get_gan_category(cat: int) -> str:
    """Return category name from id"""
    with open(os.path.join(DATA_PATH, 'categories_backwards.json'), 'r') as fr:
        data: Dict[str, str] = json.load(fr)
    return data.get(str(cat), '')


def load_biggan(gan_size: int):
    global CACHED_GAN
    CACHED_GAN = BigGAN(data_path=DATA_PATH, biggan_size=gan_size)


ROUTES = [
    web.get("/biggan/categories", handler_categories),
    web.get("/biggan/categories/backwards", handler_categories_backwards),
    web.post("/biggan/run/{what}/{size}", handler_run),
]
