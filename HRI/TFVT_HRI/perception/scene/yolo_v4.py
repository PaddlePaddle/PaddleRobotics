import numpy as np
from paddle import fluid

from perception.scene.x2paddle_yolov4 import x2paddle_net


DEFAULT_MULTI_CLASS_NMS_CONF = {
    'score_threshold': 0.5,
    'nms_top_k': 1000,
    'keep_top_k': 200,
    'nms_threshold': 0.5,
    'background_label': -1
}


class YOLOv4(object):
    """
    YOLOv4 network, see https://github.com/AlexeyAB/darknet

    This class is for inference only.
    """
    def __init__(self,
                 img_shape=[3, 416, 416],
                 num_classes=80,
                 anchors=[[12, 16], [19, 36], [40, 28], [36, 75],
                          [76, 55], [72, 146], [142, 110],
                          [192, 243], [459, 401]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 nms_cfg=DEFAULT_MULTI_CLASS_NMS_CONF,
                 get_roi_feat=False,
                 roi_feat_after_gap=False,
                 roi_feat_resolution=5,
                 fm_spacial_scale=1/32.):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.nms_cfg = nms_cfg
        self.get_roi_feat = get_roi_feat
        self.roi_feat_after_gap = roi_feat_after_gap
        self.roi_feat_resolution = roi_feat_resolution
        self.fm_spacial_scale = fm_spacial_scale

    def _parse_anchors(self, anchors):
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _correct_boxes(self, box, input_size, im_size):
        input_size = fluid.layers.cast(input_size, dtype='float32')
        im_size = fluid.layers.cast(im_size, dtype='float32')

        new_size = fluid.layers.elementwise_mul(
            im_size,
            fluid.layers.reduce_min(
                fluid.layers.elementwise_div(input_size, im_size),
                dim=1, keep_dim=True))

        offset = 0.5 * fluid.layers.elementwise_sub(input_size, new_size)
        offset = fluid.layers.elementwise_div(offset, input_size)
        scale = fluid.layers.elementwise_div(input_size, new_size)

        in_h = fluid.layers.unsqueeze(
            fluid.layers.slice(input_size, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        in_w = fluid.layers.unsqueeze(
            fluid.layers.slice(input_size, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        xmin = fluid.layers.slice(box, axes=[2], starts=[0], ends=[1])
        ymin = fluid.layers.slice(box, axes=[2], starts=[1], ends=[2])
        xmax = fluid.layers.slice(box, axes=[2], starts=[2], ends=[3])
        ymax = fluid.layers.slice(box, axes=[2], starts=[3], ends=[4])

        cx = fluid.layers.elementwise_div(
            0.5 * fluid.layers.elementwise_add(xmin, xmax), in_w)
        cy = fluid.layers.elementwise_div(
            0.5 * fluid.layers.elementwise_add(ymin, ymax), in_h)
        h = fluid.layers.elementwise_div(
            fluid.layers.elementwise_sub(ymax, ymin), in_h)
        w = fluid.layers.elementwise_div(
            fluid.layers.elementwise_sub(xmax, xmin), in_w)

        y_offset = fluid.layers.unsqueeze(
            fluid.layers.slice(offset, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        x_offset = fluid.layers.unsqueeze(
            fluid.layers.slice(offset, axes=[1], starts=[1], ends=[2]),
            axes=[1])
        h_scale = fluid.layers.unsqueeze(
            fluid.layers.slice(scale, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        w_scale = fluid.layers.unsqueeze(
            fluid.layers.slice(scale, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        cx = fluid.layers.elementwise_mul(
            fluid.layers.elementwise_sub(cx, x_offset), w_scale)
        cy = fluid.layers.elementwise_mul(
            fluid.layers.elementwise_sub(cy, y_offset), h_scale)
        h = fluid.layers.elementwise_mul(h, h_scale)
        w = fluid.layers.elementwise_mul(w, w_scale)

        im_h = fluid.layers.unsqueeze(
            fluid.layers.slice(im_size, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        im_w = fluid.layers.unsqueeze(
            fluid.layers.slice(im_size, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        new_xmin = fluid.layers.elementwise_mul(
            im_w, fluid.layers.elementwise_sub(cx, 0.5 * w))
        new_xmax = fluid.layers.elementwise_mul(
            im_w, fluid.layers.elementwise_add(cx, 0.5 * w))
        new_ymin = fluid.layers.elementwise_mul(
            im_h, fluid.layers.elementwise_sub(cy, 0.5 * h))
        new_ymax = fluid.layers.elementwise_mul(
            im_h, fluid.layers.elementwise_add(cy, 0.5 * h))

        new_box = fluid.layers.concat(
            [new_xmin, new_ymin, new_xmax, new_ymax], axis=-1)
        return new_box

    def _correct_rois(self, rois, input_size, im_size):
        input_size = fluid.layers.cast(input_size, dtype='float32')
        im_size = fluid.layers.cast(im_size, dtype='float32')
        new_size = fluid.layers.elementwise_mul(
            im_size,
            fluid.layers.reduce_min(
                fluid.layers.elementwise_div(input_size, im_size),
                dim=1, keep_dim=True))

        offset = 0.5 * fluid.layers.elementwise_sub(input_size, new_size)
        y_offset = fluid.layers.unsqueeze(
            fluid.layers.slice(offset, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        x_offset = fluid.layers.unsqueeze(
            fluid.layers.slice(offset, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        scale = fluid.layers.elementwise_div(new_size, im_size)
        y_scale = fluid.layers.unsqueeze(
            fluid.layers.slice(scale, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        x_scale = fluid.layers.unsqueeze(
            fluid.layers.slice(scale, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        # NOTE: due to training batch data may contain r2 and r1 data with different scales and offsets,
        # for convenice, here use py_func layer
        corrected_rois = create_tmp_var(
            'corrected_rois', 'float32', rois.shape)
        fluid.layers.py_func(
            func=correct_rois,
            x=[rois, x_scale, y_scale, x_offset, y_offset],
            out=corrected_rois)
        return corrected_rois

    def _no_instance_found(self, pred):
        def _np_func(pred_):
            pred_ = np.array(pred_)
            if np.all(pred_ == -1):
                return np.array([[True]])
            else:
                return np.array([[False]])

        cond = create_tmp_var('no_instance', 'bool', [1, 1])
        fluid.layers.py_func(func=_np_func, x=[pred], out=cond)
        return cond

    def infer(self, main_program=None):
        if main_program is None:
            test_program = fluid.default_main_program().clone(for_test=True)
        else:
            test_program = main_program.clone(for_test=True)

        with fluid.program_guard(test_program):
            _, fetch_list = self.build()
            return test_program, fetch_list

    def build(self):
        inputs, outputs, feature_map = x2paddle_net(self.img_shape)
        feature_map = fluid.layers.transpose(feature_map, [0, 3, 1, 2])
        im_size = fluid.layers.data(
            name='im_size', shape=[2], dtype='int32')  # (h, w) of each im
        # NOTE: instead of forced resize, we expand image to
        # keep the aspect ratio, input_size is needed for calibariation
        input_size = fluid.layers.data(
            name='in_size', shape=[2], dtype='int32')
        inputs.extend([im_size, input_size])

        boxes, scores = [], []
        downsample = 32
        for i, output in enumerate(reversed(outputs)):
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=input_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms_cfg['score_threshold'],
                downsample_ratio=downsample,
                clip_bbox=False,
                name='yolo_box' + str(i))
            box = self._correct_boxes(box, input_size, im_size)
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))

            downsample //= 2

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)
        # FIXME: using nms2 unable to train the attention controller model!
        # pred = fluid.contrib.multiclass_nms2(
        #     bboxes=yolo_boxes, scores=yolo_scores, **self.nms_cfg)
        pred = fluid.layers.multiclass_nms(
            bboxes=yolo_boxes, scores=yolo_scores, **self.nms_cfg)
        if not self.get_roi_feat:
            return inputs, [pred, feature_map]

        # Process rois feats
        def _true_func():
            return pred

        def _false_func():
            rois = fluid.layers.slice(
                pred, axes=[1], starts=[2], ends=[6])
            rois = self._correct_rois(rois, input_size, im_size)

            # FIXME: @paddle-dev, `roi_align' layer does not keep the lod
            # information!!! i.e. rois_feats.lod() == []
            rois_feats = fluid.layers.roi_align(
                input=feature_map,
                rois=rois,
                pooled_height=self.roi_feat_resolution,
                pooled_width=self.roi_feat_resolution,
                spatial_scale=self.fm_spacial_scale)

            if self.roi_feat_after_gap:
                # Global average pooling
                rois_feats = fluid.layers.reduce_sum(
                    rois_feats, dim=[2, 3]) / (self.roi_feat_resolution ** 2)

            return rois_feats

        rois_feats = fluid.layers.cond(
            self._no_instance_found(pred), _true_func, _false_func)
        return inputs, [pred, rois_feats, feature_map]


def correct_rois(rois, x_scale, y_scale, x_offset, y_offset):
    lod = rois.lod()[0]
    rois = np.array(rois)

    rois_lst = []
    for i in range(len(lod) - 1):
        rois_lst.append(rois[lod[i]:lod[i+1]])

    x_scale = np.reshape(np.array(x_scale), [-1])
    y_scale = np.reshape(np.array(y_scale), [-1])
    x_offset = np.reshape(np.array(x_offset), [-1])
    y_offset = np.reshape(np.array(y_offset), [-1])

    rois_lst_ = []
    for r, xs, ys, xo, yo in zip(rois_lst, x_scale, y_scale,
                                 x_offset, y_offset):
        scale = np.array([xs, ys, xs, ys])
        offset = np.array([xo, yo, xo, yo])
        rois_lst_.append(r * scale + offset)
    flatten_corrected_rois = np.concatenate(rois_lst_, axis=0)
    res = fluid.LoDTensor()
    res.set(flatten_corrected_rois, fluid.CPUPlace())
    res.set_lod([lod])
    return res


def create_tmp_var(name, dtype, shape):
    return fluid.default_main_program().current_block().create_var(
        name=name, dtype=dtype, shape=shape)
