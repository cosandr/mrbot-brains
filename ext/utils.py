import logging
import os

from aiohttp import ClientSession

logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)
SESS: ClientSession = None


async def init_sess(app):
    global SESS
    SESS = ClientSession()
    logger.info('Client session initialized')


async def close_sess(app):
    global SESS
    await SESS.close()
    logger.info('Client session closed')


async def bytes_from_url(url: str, **kwargs) -> bytes:
    """
    Save `url` content into a `BytesIO` buffer and return it.

    :param url: URL to download
    :returns: Buffer with downloaded content
    """
    global SESS
    async with SESS.get(url, **kwargs) as resp:
        return await resp.read()


def merge_dict(d1: dict, d2: dict, keep_none=False) -> dict:
    """Merge d2 into d1, ignoring extra keys in d2"""
    ret = d1.copy()
    for k in d1.keys():
        if k in d2:
            if not keep_none and d2[k] is None:
                continue
            ret[k] = d2[k]
    return ret


def check_opencv_codec(codec: str) -> bool:
    try:
        import cv2
        # Check if avc1 codec is available
        _test_name = f'{codec}_test.mp4'
        _cc = cv2.VideoWriter_fourcc(*codec)
        _out = cv2.VideoWriter(_test_name, _cc, 10.0, (50, 50))
        if os.path.exists(_test_name):
            os.unlink(_test_name)
            return True
        return False
    except ImportError:
        return False
