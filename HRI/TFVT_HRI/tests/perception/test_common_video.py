import os
import cv2
import pytest
import numpy as np

from perception.common.video import clip_video_to_frames

VIDEO = 'data/no_interaction.mp4'


def test_clip_video_to_frames():
    ffmpeg_clip = 'data/clip_by_ffmpeg.mp4'
    try:
        # NOTE: ffmpeg would append extra copied frames
        os.system('ffmpeg -ss 1 -i %s -c copy -t 2 -y %s' %
                  (VIDEO, ffmpeg_clip))
    except Exception:
        pass

    if not os.path.exists(ffmpeg_clip):
        return True

    frames = clip_video_to_frames(VIDEO, 1000., 3000.)
    cap = cv2.VideoCapture(ffmpeg_clip)
    success, i = True, 0
    while success and i < len(frames):
        success, frame = cap.read()
        assert np.all(frame == frames[i])
        i += 1

    cap.release()
