# MrBot Brains
A REST API for use with my Discord bot. Its purpose is to run CPU intensive tasks separately from the bot. Most endpoints take JSON data as their input (POST), an error indicating what went wrong may be returned.

## OpenCV
For browser-playable videos OpenCV must be able to use the avc1 codec,
the PyPi package does not work! Some distributions include non-free codecs when
building OpenCV (ArchLinux for example) but otherwise you will have to build it yourself.
See my [Dockerfile](https://github.com/cosandr/containers/blob/master/containers/opencv/deb.Dockerfile)
for an example.

## Environment variables

* `DATA_PATH` is where trained models and other data is stored
* `UPLOAD_PATH` is where images/videos are placed, should be on a web server
* `UPLOAD_URL` corresponding URL of UPLOAD_PATH
* `MP4_FALLBACK` change fallback mp4 codec if avc1 is not available, default mp4v

## Docker

See the [Dockerfile](https://github.com/cosandr/containers/blob/master/containers/mrbot/brains.Dockerfile).
