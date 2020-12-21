import os
import numpy as np
from paddle import fluid

from perception.scene.yolo_v3 import YOLOv3
from perception.scene.yolo_v4 import YOLOv4
from perception.common.utils import yolov3_img_preprocess, \
    yolov4_img_preprocess, yolov3_instance_postprocess, LodTensor_to_Tensor


class SceneSensor(object):
    def __init__(self,
                 model_dir,
                 gpu=None,
                 batch_size=8,
                 img_shape=[3, 480, 640],
                 pad_to_stride=32,
                 confidence_threshold=0.5,
                 fm_only=False,
                 roi_feat_resolution=5,
                 algorithm='yolov4'):

        # TODO: support zero padding with given `pad_to_stride`
        if not os.path.exists(model_dir):
            raise ValueError('The model path `%s` does not exit.' % model_dir)
        if algorithm not in ['yolov3', 'yolov4']:
            raise NotImplementedError
        if fm_only and algorithm == 'yolov4':
            raise NotImplementedError

        if gpu is not None and type(gpu) is int:
            place = fluid.CUDAPlace(gpu)
        else:
            place = fluid.CPUPlace()

        self.fm_only = fm_only
        self.roi_feat_resolution = roi_feat_resolution
        self.algorithm = algorithm
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.confidence_th = confidence_threshold
        self.exe = fluid.Executor(place)
        self.infer_scope = fluid.Scope()

        with fluid.scope_guard(self.infer_scope):
            if self.fm_only:
                self._build_fm_program(model_dir, img_shape)
            else:
                self._build_infer_program(model_dir, img_shape)

    def close(self):
        self.exe.close()

    def get_instances(self, img):
        """
        Get instances detection bounding box (maybe with mask) for given image.

        Args:
            img (np.array | list of np.array): image or a list of image in HWC.
        """
        assert not self.fm_only, \
            'Current SceneSensor only supports to extract feature map'
        if not type(img) is list:
            img = [img]

        h, w = img[0].shape[0], img[0].shape[1]
        if self.algorithm == 'yolov3':
            img_input = np.array([yolov3_img_preprocess(i) for i in img])
            img_size_input = np.array([[h, w]] * len(img), dtype=np.int32)
            feed_data = {'image': img_input, 'im_size': img_size_input}
            fetch_preds = self.fetch_list[:1]
        elif self.algorithm == 'yolov4':
            img_input = np.array([yolov4_img_preprocess(i) for i in img])
            img_size_input = np.array([[h, w]] * len(img), dtype=np.int32)
            in_size_input = np.array([self.img_shape[1:]] * len(img),
                                     dtype=np.int32)
            feed_data = {
                'image': img_input,
                'im_size': img_size_input,
                'in_size': in_size_input
            }
            fetch_preds = self.fetch_list[:1]
        else:
            raise NotImplementedError

        instances_lst = []
        for i in range(0, len(img_input), self.batch_size):
            feed_vars = dict()
            for k, v in feed_data.items():
                feed_vars[k] = v[i:i+self.batch_size]

            with fluid.scope_guard(self.infer_scope):
                preds = self.exe.run(
                    self.infer_program,
                    feed=feed_vars,
                    fetch_list=fetch_preds,
                    return_numpy=False)

                c = self.confidence_th
                if self.algorithm in ['yolov3', 'yolov4']:
                    # `preds` == `bbox_preds_LoD`
                    bbox_preds = LodTensor_to_Tensor(preds[0])
                    if len(bbox_preds) != feed_vars['image'].shape[0]:
                        expected_len = feed_vars['image'].shape[0]
                        bbox_preds = [bbox_preds[0]] * expected_len

                    instances_preds = [yolov3_instance_postprocess(
                        b, c, h, w) for b in bbox_preds]

            instances_lst.extend(instances_preds)

        return instances_lst

    def get_feature_map(self, img, flatten=False):
        """
        Get feature map for given image.
        """
        if not type(img) is list:
            img = [img]

        if self.algorithm == 'yolov3':
            img_input = np.array([yolov3_img_preprocess(i) for i in img])
            feed_data = {'image': img_input}
            if not self.fm_only:
                h, w = img[0].shape[0], img[0].shape[1]
                # im_size: h, w
                img_size_input = np.array(
                    [[h, w]] * len(img), dtype=np.int32)
                feed_data['im_size'] = img_size_input
        elif self.algorithm == 'yolov4':
            img_input = np.array([yolov4_img_preprocess(i) for i in img])
            img_size_input = np.array([[h, w]] * len(img), dtype=np.int32)
            in_size_input = np.array([self.img_shape[1:]] * len(img),
                                     dtype=np.int32)
            feed_data = {
                'image': img_input,
                'im_size': img_size_input,
                'in_size': in_size_input
            }
        else:
            raise NotImplementedError

        feature_maps = []
        for i in range(0, len(img_input), self.batch_size):
            feed_vars = dict()
            for k, v in feed_data.items():
                feed_vars[k] = v[i:i+self.batch_size]

            with fluid.scope_guard(self.infer_scope):
                feature_map_batch = self.exe.run(
                    self.infer_program,
                    feed=feed_vars,
                    fetch_list=[self.fetch_list[-1]])[0]

            batch_n = feature_map_batch.shape[0]
            if flatten:
                feature_map_batch = np.reshape(
                    feature_map_batch, [batch_n, -1])

            feature_maps.extend(
                [feature_map_batch[j] for j in range(batch_n)])

        return feature_maps

    def get_instances_with_feats(self,
                                 img,
                                 get_full_fm=False):
        """
        Get instances with features for each instance,
        that is extracted from feature map using RoIAlign.
        """
        assert self.algorithm == 'yolov4', 'Only support YOLOv4 now'
        assert not self.fm_only, \
            'Current SceneSensor only supports to extract feature map'
        if not type(img) is list:
            img = [img]

        h, w = img[0].shape[0], img[0].shape[1]
        img_input = np.array([yolov4_img_preprocess(i) for i in img])
        img_size_input = np.array([[h, w]] * len(img), dtype=np.int32)
        in_size_input = np.array(
            [self.img_shape[1:]] * len(img), dtype=np.int32)
        feed_data = {
            'image': img_input,
            'im_size': img_size_input,
            'in_size': in_size_input
        }

        instances_lst, fm_lst = [], []
        for i in range(0, len(img_input), self.batch_size):
            feed_vars = dict()
            for k, v in feed_data.items():
                feed_vars[k] = v[i:i+self.batch_size]
                original_img = img[i:i+self.batch_size]

            with fluid.scope_guard(self.infer_scope):
                bbox_preds, rois_feats, feature_maps = self.exe.run(
                    self.infer_program,
                    feed=feed_vars,
                    fetch_list=self.fetch_list,
                    return_numpy=False)
                rois_feats = LodTensor_to_Tensor(
                    rois_feats, lod=bbox_preds.lod())
                bbox_preds = LodTensor_to_Tensor(bbox_preds)
                if len(bbox_preds) != feed_vars['image'].shape[0]:
                    expected_len = feed_vars['image'].shape[0]
                    bbox_preds = [bbox_preds[0]] * expected_len
                    rois_feats = [rois_feats[0]] * expected_len
                feature_maps = np.array(feature_maps)

                c = self.confidence_th
                instances_preds = [
                    yolov3_instance_postprocess(b, c, h, w, fm=fm, img=im)
                    for b, fm, im in zip(bbox_preds, rois_feats,
                                         original_img)]

            instances_lst.extend(instances_preds)
            fm_lst.extend(list(feature_maps))

        if get_full_fm:
            return instances_lst, fm_lst
        else:
            return instances_lst

    def _build_infer_program(self, model_dir, img_shape):
        main, startup = fluid.Program(), fluid.Program()
        with fluid.program_guard(main, startup):
            if self.algorithm == 'yolov3':
                net = YOLOv3(img_shape)
            elif self.algorithm == 'yolov4':
                net = YOLOv4(
                    img_shape=img_shape,
                    get_roi_feat=True,
                    roi_feat_resolution=self.roi_feat_resolution)

            self.infer_program, self.fetch_list = net.infer(main)
            fluid.io.load_params(
                self.exe, model_dir, main_program=self.infer_program)

    def _build_fm_program(self, model_dir, img_shape):
        main, startup = fluid.Program(), fluid.Program()
        with fluid.program_guard(main, startup):
            image = fluid.layers.data(name='image', shape=img_shape)
            net = SceneSensor.network(img_shape, self.algorithm)
            fm = net.build_fm_extractor({'image': image})

            self.infer_program = main.clone(for_test=True)
            self.fetch_list = [fm]
            fluid.io.load_params(
                self.exe, model_dir, main_program=self.infer_program)

    @staticmethod
    def network(img_shape,
                algo,
                prefix_name='',
                get_roi_feat=True,
                roi_feat_after_gap=False,
                roi_feat_resolution=5):
        if algo == 'yolov3':
            return YOLOv3(img_shape, prefix_name=prefix_name)
        elif algo == 'yolov4':
            return YOLOv4(img_shape=img_shape,
                          get_roi_feat=get_roi_feat,
                          roi_feat_after_gap=roi_feat_after_gap,
                          roi_feat_resolution=roi_feat_resolution)
        else:
            raise ValueError('Unknown algorithm for scene sensor.')

    @staticmethod
    def img_preprocess(algo, img):
        if algo == 'yolov3':
            return yolov3_img_preprocess(img)
        elif algo == 'yolov4':
            return yolov4_img_preprocess(img)
        else:
            raise ValueError('Unknown algorithm for scene sensor.')
