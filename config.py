import os

from ext.utils import check_opencv_codec

DATA_PATH = os.getenv('DATA_PATH', None)
if not DATA_PATH:
    print('Data path variable DATA_PATH missing')
    exit(0)

UPLOAD_PATH = os.getenv('UPLOAD_PATH', '')
if not UPLOAD_PATH:
    if not os.path.exists('./data'):
        os.mkdir("./data", 0o755)
    UPLOAD_PATH = "data"
UPLOAD_URL = os.getenv('UPLOAD_URL', '')

MP4_CODEC = 'avc1'
# mp4v will "work" but it doesn't display in any browser I've tried
MP4_FALLBACK = os.getenv('MP4_FALLBACK', 'mp4v')

if check_opencv_codec(MP4_CODEC):
    print(f'\tCheck OpenCV: {MP4_CODEC} codec OK')
else:
    if MP4_FALLBACK:
        if check_opencv_codec(MP4_FALLBACK):
            print(f'\tCheck OpenCV: {MP4_CODEC} codec unavailable but fallback {MP4_FALLBACK} is')
            MP4_CODEC = MP4_FALLBACK
        else:
            print((f'\tCheck OpenCV: {MP4_CODEC} codec unavailable, fallback {MP4_FALLBACK} unavailable'
                   '\n\t\t!!! OpenCV will fail !!!'))
    else:
        print((f'\tCheck OpenCV: {MP4_CODEC} codec unavailable, no fallback'
              '\n\t\t!!! OpenCV will fail !!!'))
