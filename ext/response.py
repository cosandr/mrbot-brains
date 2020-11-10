import json
from typing import Union

from aiohttp import web


class Response:
    def __init__(self, **kwargs):
        self._web = web.Response(content_type="application/json")
        self._attrs = dict(
            data=None,    # Return data, probably a dict or string
            error='',     # An error string
            params=None,  # The parameters used by the function that was run
            time=0,       # How long the request took to complete
        )
        self._attrs.update(kwargs)

    @property
    def data(self) -> Union[str, dict, list]:
        return self._attrs.get('data', None)

    @data.setter
    def data(self, val: Union[str, dict, list]):
        self._attrs['data'] = val

    @property
    def error(self) -> str:
        return self._attrs.get('error', '')

    @error.setter
    def error(self, val: str):
        self._attrs['error'] = val

    @property
    def params(self) -> dict:
        return self._attrs.get('params', {})

    @params.setter
    def params(self, val: dict):
        self._attrs['params'] = val

    @property
    def time(self) -> dict:
        return self._attrs.get('time', 0)

    @time.setter
    def time(self, val: float):
        self._attrs['time'] = val

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self._attrs.items() if v})

    @property
    def status(self) -> int:
        return self._web.status

    @status.setter
    def status(self, val: int):
        self._web.set_status(val)

    @property
    def web_response(self) -> web.Response:
        """Return internal web response"""
        self._web.body = self.to_json().encode('utf-8')
        return self._web

    def to_web(self, **kwargs) -> web.Response:
        """Return aiohttp web response, keyword args are passed to web.Response constructor"""
        return web.Response(body=self.to_json().encode('utf-8'), content_type="application/json", **kwargs)
