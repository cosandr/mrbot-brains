#!/usr/bin/env python3

import argparse
import glob
import importlib
import logging
import os

from aiohttp import web

from ext import utils


def main():
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S'))
    # File Handler
    fh = logging.FileHandler(filename='launcher.log', encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

    logger = logging.getLogger('launcher')
    logger.setLevel(logging.DEBUG)

    app = web.Application()
    app.on_startup.append(utils.init_sess)
    app.on_shutdown.append(utils.close_sess)

    # Import all commands
    modules = []
    for f in glob.glob('commands/*.py'):
        name = f[:-3].replace('/', '.')
        try:
            modules.append(importlib.import_module(name))
        except Exception as e:
            logger.info(f'Cannot import {name}: {e}')
    # Get all handlers
    for m in modules:
        module_routes = getattr(m, "ROUTES", [])
        if not module_routes:
            logger.info(f'- Module {m.__name__} has no routes')
            continue
        # Add one by one to print status and skip invalid routes
        logger.info(f'- Found {len(module_routes)} routes for {m.__name__}')
        for r in module_routes:
            try:
                app.add_routes([r])
                logger.info(f'-- OK {r.path}')
            except Exception as e:
                logger.info(f'-- FAIL {r.path}: {e}')
    # Port 7762 (PROC)
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen-address', type=str, default='0.0.0.0:7762',
                        help='Listen address, absolute path indicates unix socket')
    args = parser.parse_args()
    if args.listen_address.startswith('/'):
        web.run_app(app, path=args.listen_address)
    else:
        host, port = args.listen_address.split(':', 1)
        web.run_app(app, host=host, port=int(port))
    # Delete old socket file if needed
    if args.listen_address.startswith('/'):
        os.unlink(args.listen_address)
        logger.info(f'Deleted socket file {args.listen_address}')


if __name__ == "__main__":
    main()
