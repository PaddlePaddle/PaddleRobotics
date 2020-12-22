import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from subprocess import Popen, PIPE
import pycocotools.mask as coco_mask_util


def draw_bboxes(image, bboxes, labels=None, output_file=None, fill='red'):
    """
    Draw bounding boxes on image.
    Return image with drawings as BGR ndarray.

    Args:
        image (string | ndarray): input image path or image BGR ndarray.
        bboxes (np.array): bounding boxes.
        labels (list of string): the label names of bboxes.
        output_file (string): output image path.
    """
    if labels:
        assert len(bboxes) == len(labels)

    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1], mode='RGB')
    else:
        raise ValueError('`image` should be image path in string or '
                         'image ndarray.')

    draw = ImageDraw.Draw(image)
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        left, right, top, bottom = xmin, xmax, ymin, ymax
        lines = [(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)]
        draw.line(lines, width=4, fill=fill)
        if labels and image.mode == 'RGB':
            draw.text((left, top), labels[i], (255, 255, 0))

    if output_file:
        print('The image with bbox is saved as {}'.format(output_file))
        image.save(output_file)

    return np.array(image)[:, :, ::-1]


def save_as_gif(images, gif_file, fps=5):
    """
    Save numpy images as gif file using ffmpeg.

    Args:
        images (list|ndarray): a list of uint8 images or uint8 ndarray
            with shape [time, height, width, channels]. `channels` can
            be 1 or 3.
        gif_file (str): path to saved gif file.
        fps (int): frames per second of the animation.
    """
    h, w, c = images[0].shape
    cmd = [
      'ffmpeg', '-y',
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',
      '-r', '%.02f' % fps,
      '-s', '%dx%d' % (w, h),
      '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
      '-i', '-',
      '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
      '-r', '%.02f' % fps,
      '-f', 'gif',
      '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc

    with open(gif_file, 'wb') as f:
        f.write(out)


def colormap(rgb=False):
    """
    Get colormap
    """
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list
