import logging
import os

from aiohttp import web

import config as cfg
from ext import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def handler_health(r: web.Request) -> web.Response:
    logger.debug(r.path)
    status = 200
    kwargs = {"status": "OK"}
    check_paths = [
        os.path.join(cfg.DATA_PATH, "biggan"),
        os.path.join(cfg.DATA_PATH, "gpt-2", "models"),
        os.path.join(cfg.DATA_PATH, "IMG"),
        os.path.join(cfg.DATA_PATH, "textgenrnn"),
    ]
    for path in check_paths:
        if not os.path.exists(path):
            # Only set status the first time, but log every other failure
            logger.error("'%s' not found", path)
            if status == 200:
                kwargs["status"] = "MISSING_DATA"
                kwargs["error"] = f"'{path}' not found"
                status = 500

    resp = Response(**kwargs)
    resp.status = status
    return resp.web_response


ROUTES = [
    web.get("/health", handler_health),
]
