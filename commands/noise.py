import asyncio
import logging
import multiprocessing as mp
import os
import time
from base64 import b64encode

import cv2
import imageio
import numpy as np
from PIL import Image
from aiohttp import web
from noise import snoise2, snoise3

import config as cfg
from ext.parser import Parser
from ext.response import Response
from ext.utils import check_opencv_codec

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
loop = asyncio.get_event_loop()
NUM_PROCESSES = mp.cpu_count()


async def handler_run_image(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    parser = Parser()
    parser.add_argument('--w', type=int, default=640),
    parser.add_argument('--h', type=int, default=480),
    parser.add_argument('--octaves', type=int, default=10),
    parser.add_argument('--scale', type=float, default=1 / 50),
    parser.add_argument('--persistence', type=float, default=0.3),
    parser.add_argument('--r', type=int, default=255),
    parser.add_argument('--g', type=int, default=50),
    parser.add_argument('--b', type=int, default=255),
    parser.add_argument('--x_rep', type=int, default=100),
    parser.add_argument('--y_rep', type=int, default=100),
    parser.add_argument('--colour', type=str, default='rgb', choices=('rgb', 'bw'))
    # Are we asked to always return image bytes?
    get_image = 'get_image' in r.query
    try:
        got_params: dict = await r.json()
        params = vars(parser.parse_kwargs(got_params))
    except Exception as e:
        logger.error(f'noise parameter parsing error: {str(e)}')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response
    resp.params = params
    filename = "{w}x{h}_{octaves}o_{persistence:.1f}p_{scale:.1f}s_{r}r_{g}g_{b}b_{x_rep}xr_{y_rep}yr_{colour}.jpg".format(
        **params,
    )
    file_path = os.path.join(cfg.UPLOAD_PATH, filename)
    try:
        start = time.perf_counter()
        if not os.path.exists(file_path):
            await loop.run_in_executor(None, lambda: run_noise(file_path=file_path, **params))

        resp.data = dict(filename=filename, processes=NUM_PROCESSES)
        # Respond with URL to image if we have a hostname configured
        if cfg.UPLOAD_URL:
            resp.data['url'] = f'{cfg.UPLOAD_URL}/{filename}'
        # Encode image and send it directly
        if get_image or not cfg.UPLOAD_URL:
            logger.debug(f'Base64 encoding {filename}')
            with open(file_path, 'rb') as fr:
                resp.data['image'] = b64encode(fr.read()).decode('utf-8')
        resp.time = time.perf_counter() - start
    except Exception as e:
        logger.exception('gif run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


async def handler_run_gif(r: web.Request) -> web.Response:
    logger.debug(r.path)
    resp = Response()
    parser = Parser()
    parser.add_argument('--w', type=int, default=100),
    parser.add_argument('--h', type=int, default=100),
    parser.add_argument('--octaves', type=int, default=5),
    parser.add_argument('--scale', type=float, default=1/20),
    parser.add_argument('--timescale', type=int, default=10),
    parser.add_argument('--persistence', type=float, default=0.3),
    parser.add_argument('--lacunarity',type=float, default=2.0),
    parser.add_argument('--video', default=False),
    parser.add_argument('--fps', type=int, default=24),
    parser.add_argument('--frames', type=int, default=10)
    # Are we asked to always return image bytes?
    get_image = 'get_image' in r.query
    try:
        got_params: dict = await r.json()
        # Don't parse bool
        video = got_params.pop('video', False)
        params = vars(parser.parse_kwargs(got_params))
        if not isinstance(video, bool):
            raise Exception('Video should be a bool')
        params['video'] = video
    except Exception as e:
        logger.exception('gif parameter parsing error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
        return resp.web_response

    # Make sure we can actually run video
    if params['video']:
        if not check_opencv_codec(cfg.MP4_CODEC):
            resp.error = f'Codec for {cfg.MP4_CODEC} video is not available, try using GIF instead.'
            resp.status = web.HTTPServerError.status_code
            return resp.web_response

    resp.params = params
    fmt_params = params.copy()
    fmt_params.pop('video')
    fmt_params['file_ext'] = 'mp4' if params['video'] else 'gif'
    filename = '{w}x{h}_{octaves}o_{persistence:.1f}p_{scale:.1f}s_{timescale}ts_{lacunarity}l_{fps}fps{frames}.{file_ext}'.format(
        **fmt_params,
    )
    file_path = os.path.join(cfg.UPLOAD_PATH, filename)
    try:
        start = time.perf_counter()
        if not os.path.exists(file_path):
            await loop.run_in_executor(None, lambda: run_gif(file_path=file_path, **params))

        resp.data = dict(filename=filename, processes=NUM_PROCESSES)
        # Respond with URL to image if we have a hostname configured
        if cfg.UPLOAD_URL:
            resp.data['url'] = f'{cfg.UPLOAD_URL}/{filename}'
        # Encode image and send it directly
        if get_image or not cfg.UPLOAD_URL:
            logger.debug(f'Base64 encoding {filename}')
            with open(file_path, 'rb') as fr:
                resp.data['image'] = b64encode(fr.read()).decode('utf-8')
        resp.time = time.perf_counter() - start
    except Exception as e:
        logger.exception('gif run error')
        resp.error = str(e)
        resp.status = web.HTTPBadRequest.status_code
    return resp.web_response


def run_gif(w, h, octaves, persistence, scale, timescale, lacunarity, fps, frames, file_path, video=False):
    r = 255
    p_list = []
    queue = mp.Queue()
    my_fp = 0
    for ID in range(NUM_PROCESSES):
        my_f = my_fp + int(frames / NUM_PROCESSES)
        if ID < frames % NUM_PROCESSES:
            my_f += 1
        p = mp.Process(
            target=_noise_parallel_3d,
            args=(timescale, queue, my_fp, w, h, my_f, octaves, persistence, lacunarity, scale)
        )
        p_list.append(p)
        p.start()
        my_fp = my_f

    frame_arr = []
    for i in range(frames):
        frame_arr.append(i)

    for _ in p_list:
        ret = queue.get()
        my_fp = ret[0][0]
        my_f = ret[0][1]
        for j in range(my_fp, my_f):
            p_arr = np.zeros((h, w), np.uint8)
            p_arr[:, :] = (ret[1][j - my_fp] * r).astype(int)
            frame_arr[j] = p_arr

    [p.join() for p in p_list]

    if video:
        cc = cv2.VideoWriter_fourcc(*cfg.MP4_CODEC)
        out = cv2.VideoWriter(file_path, cc, float(fps), (w, h))
        if not os.path.exists(file_path):
            out.release()
            raise Exception('Cannot create video, codec is likely missing, try using GIF instead.')
        for i in range(0, frames):
            tmp_img = cv2.cvtColor(np.asarray(frame_arr[i], dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            out.write(tmp_img)
        out.release()
    else:
        imageio.mimsave(file_path, frame_arr, 'GIF', fps=fps)


def run_noise(w, h, octaves, persistence, scale, r, g, b, x_rep, y_rep, colour, file_path):
    queue = mp.Queue()
    p_list = []
    my_wp = 0
    for ID in range(0, NUM_PROCESSES):
        my_w = my_wp + int(w / NUM_PROCESSES)
        if ID < w % NUM_PROCESSES:
            my_w += 1
        p = mp.Process(
            target=_noise_parallel,
            args=(queue, my_wp, my_w, (h, scale, octaves, x_rep, y_rep, persistence))
        )
        p_list.append(p)
        p.start()
        my_wp = my_w

    img_bw = np.zeros((h, w))
    for _ in p_list:
        ret = queue.get()
        img_bw[:, (ret[0][0]):(ret[0][1])] = ret[1]

    [p.join() for p in p_list]

    if colour == 'rgb':
        img_arr = np.zeros((h, w, 3), np.uint8)
        img_arr[:, :, 0] = (img_bw * r).astype(int)
        img_arr[:, :, 1] = (img_bw * g).astype(int)
        img_arr[:, :, 2] = (img_bw * b).astype(int)
    else:
        img_arr = np.zeros((h, w), np.uint8)
        img_arr[:, :] = (img_bw * r).astype(int)

    img = Image.fromarray(img_arr)
    img.save(file_path, format='jpeg')


def _noise_simple(wp, w, h, scale, octave, x_rep, y_rep, persistence) -> np.ndarray:
    data = np.zeros((h, w - wp))
    for x in range(wp, w):
        for y in range(h):
            i = snoise2(x * scale, y * scale, octave, repeatx=x_rep, repeaty=y_rep, persistence=persistence)
            if i < 0:
                i *= -1
            data[y, x - wp] = i

    return data


def _noise_simple_3d(w, h, t, octave, persistence, lacunarity, scale):
    data = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            i = snoise3(x * scale, y * scale, t, octave, persistence, lacunarity)
            if i < 0:
                i *= -1
            data[y, x] = i

    return data


def _noise_parallel(q, wp, w, args):
    q.put(([wp, w], _noise_simple(wp, w, *args)))


def _noise_parallel_3d(ts, q, tp, w, h, t, octave, persistence, lacunarity, scale):
    my_frames = []
    for i in range(tp, t, 1):
        my_frames.append(_noise_simple_3d(w, h, i / ts, octave, persistence, lacunarity, scale))
    q.put(([tp, t], my_frames))


ROUTES = [
    web.post("/noise/run/image", handler_run_image),
    web.post("/noise/run/gif", handler_run_gif),
]
