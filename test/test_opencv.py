import os

import PIL.Image
import cv2
import numpy as np
import pytest


def test_opencv_avc1():
    filename = 'test.mp4'
    if os.path.exists(filename):
        os.unlink(filename)
    width = height = 256
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    assert os.path.exists(filename), 'No video file created'
    for frame in sorted(os.listdir("frames")):
        img = PIL.Image.open("frames/" + frame)
        tmp_img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        out.write(tmp_img)

    # Release everything if job is finished
    out.release()
    assert os.stat(filename).st_size > 0, 'Video file is empty'


@pytest.mark.skip
def test_opencv_mp4v():
    filename = 'test.mp4'
    if os.path.exists(filename):
        os.unlink(filename)
    width = height = 256
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    assert os.path.exists(filename), 'No video file created'
    for frame in sorted(os.listdir("frames")):
        img = PIL.Image.open("frames/" + frame)
        tmp_img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        out.write(tmp_img)

    # Release everything if job is finished
    out.release()
    assert os.stat(filename).st_size > 0, 'Video file is empty'

# import imageio
# frame_arr = []
# for frame in sorted(os.listdir("frames")):
#     img = PIL.Image.open("frames/" + frame)
#     tmp_img = np.asarray(img, dtype=np.uint8)
#     frame_arr.append(tmp_img)

# imageio.mimwrite('test.gif', frame_arr, "GIF", fps=60)
