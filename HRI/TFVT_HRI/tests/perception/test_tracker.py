import pytest
import numpy as np

from perception.common.video import clip_video_to_frames
from perception.tracker.re_id import create_box_encoder

VIDEO = 'data/potential_interaction.mp4'
Encoder_on_MARS_Dataset = 'pretrain_models/mars-small128.pb'


def test_image_encoder():
    patch_n = 2
    frames = clip_video_to_frames(VIDEO, 3001., 4000.)
    image_encoder = create_box_encoder(
        Encoder_on_MARS_Dataset, batch_size=patch_n)

    h, w, _ = frames[0].shape
    boxes = [[int(w*0.1), int(h*0.1), int(w*0.5), int(h*0.5)]] * patch_n

    features = image_encoder(frames[0], boxes)
    assert len(features) == len(boxes)
