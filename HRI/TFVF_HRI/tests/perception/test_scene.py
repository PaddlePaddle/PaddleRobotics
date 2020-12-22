import pytest
import cv2
import time
import numpy as np

from perception.scene.eval import SceneSensor
from perception.common.video import clip_video_to_frames, VideoWriter
from perception.common.visualize import draw_bboxes
from perception.common.utils import robot2_frame_crop_resize

VIDEO = 'data/potential_interaction.mp4'
R2_VIDEO = 'data/passing_by.mp4'
YOLOv3_MODEL = 'pretrain_models/yolov3_r34'
YOLOv4_MODEL = 'tools/yolov4_paddle/inference_model'


class TestYOLOv3SceneSensor(object):
    def setup_class(self):
        self.scene_sensor = SceneSensor(
            YOLOv3_MODEL, gpu=0, algorithm='yolov3')
        self.frames = clip_video_to_frames(VIDEO, 3001., 4000.)

    def test_get_instances(self, export=True):
        instances_lst = self.scene_sensor.get_instances(self.frames)
        assert len(instances_lst) == len(self.frames)

        if export:
            h, w, fps = 480, 640, 24.  # read from VIDEO
            video_writer = VideoWriter(
                'data/scene_yolo_demo.mp4', (w, h), fps)
            for frame, instances in zip(self.frames, instances_lst):
                bboxes = np.array([i['bbox'] for i in instances])
                labels = [i['category'] for i in instances]

                frame_draw = draw_bboxes(frame, bboxes, labels=labels)
                video_writer.add_frame(frame_draw)
            video_writer.close()

    def test_get_feature_map(self):
        feature_maps = self.scene_sensor.get_feature_map(self.frames)
        assert len(feature_maps) == len(self.frames)


class TestYOLOv4SceneSensor(object):
    def setup_class(self):
        self.roi_feat_resolution = 5
        self.scene_sensor = SceneSensor(
            YOLOv4_MODEL,
            gpu=0,
            img_shape=[3, 416, 416],
            roi_feat_resolution=self.roi_feat_resolution,
            algorithm='yolov4')
        self.frames = clip_video_to_frames(R2_VIDEO, 0., None)

    def test_get_instances(self, export=True):
        instances_lst = self.scene_sensor.get_instances(self.frames)
        assert len(instances_lst) == len(self.frames)

        if export:
            h, w, fps = 720, 1280, 24.  # read from VIDEO
            video_writer = VideoWriter(
                'data/scene_yolo4_demo.mp4', (w, h), fps)
            for frame, instances in zip(self.frames, instances_lst):
                bboxes = np.array([i['bbox'] for i in instances])
                labels = [i['category'] for i in instances]

                frame_draw = draw_bboxes(frame, bboxes, labels=labels)
                video_writer.add_frame(frame_draw)
            video_writer.close()

    def test_get_instances_with_feats(self):
        instances_lst, fm_lst = self.scene_sensor.get_instances_with_feats(
            self.frames, get_full_fm=True)
        _, h, w = instances_lst[0][0]['fm'].shape
        assert h == w == self.roi_feat_resolution
        assert len(instances_lst) == len(fm_lst) == len(self.frames)
