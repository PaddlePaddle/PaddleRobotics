import os
import pytest

from perception.common.video import clip_video_to_frames
from perception.common.visualize import save_as_gif

VIDEO = 'data/no_interaction.mp4'


def test_save_as_gif():
    frames = clip_video_to_frames(VIDEO, 1000., 3000.)
    gif_file = 'data/test_save_as_gif.gif'
    save_as_gif(frames, gif_file)
    assert os.path.exists(gif_file)
