from collections import OrderedDict
from paddle import fluid

from perception.common.backbone import ResNet
from perception.common.ppdet_anchor_head import YOLOv3Head

# YOLOv3 ResNet34 config
# see github PaddleDetection/configs/yolov3_r34.yml
ResNet34_CONF = {
    'depth': 34,
    'feature_maps': [3, 4, 5],
    'freeze_at': 0,
    'norm_type': 'sync_bn',
    'freeze_norm': False,
    'norm_decay': 0.
}


class YOLOv3(object):
    """
    YOLOv3 network, see https://arxiv.org/abs/1804.02767

    Args:
        backbone (object): an backbone instance
        yolo_head (object): an `YOLOv3Head` instance
    """
    def __init__(self,
                 data_shape,
                 backbone=ResNet(**ResNet34_CONF),
                 yolo_head=YOLOv3Head(),
                 prefix_name=''):
        assert isinstance(yolo_head, YOLOv3Head)
        self.data_shape = data_shape
        self.backbone = backbone
        self.yolo_head = yolo_head

        self.backbone.prefix_name = prefix_name
        self.yolo_head.prefix_name = prefix_name

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.backbone(im)

        if isinstance(body_feats, OrderedDict):
            body_feat_names = list(body_feats.keys())
            body_feats = [body_feats[name] for name in body_feat_names]

        self.fm = body_feats[-1]

        if mode == 'train':
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            gt_score = feed_vars['gt_score']

            yolo_loss = self.yolo_head.get_loss(
                body_feats, gt_box, gt_label, gt_score)
            return {'loss': yolo_loss}
        else:
            im_size = feed_vars['im_size']
            return self.yolo_head.get_prediction(body_feats, im_size)

    def build_fm_extractor(self, feed_vars):
        im = feed_vars['image']
        body_feats = self.backbone(im)

        if isinstance(body_feats, OrderedDict):
            k = list(body_feats.keys())[-1]
            return body_feats[k]
        else:
            return body_feats[-1]

    def train(self, feed_vars):
        return self.build(feed_vars, mode='train')

    def eval(self, feed_vars):
        return self.build(feed_vars, mode='test')

    def infer(self, main_program=None):
        if main_program is None:
            test_program = fluid.default_main_program().clone(for_test=True)
        else:
            test_program = main_program.clone(for_test=True)

        with fluid.program_guard(test_program):
            image = fluid.layers.data(
                name='image', shape=self.data_shape, dtype='float32')
            im_size = fluid.layers.data(
                name='im_size', shape=[2], dtype='int32')  # (h, w) of each im
            feed_vars = {
                'image': image,
                'im_size': im_size
            }
            fetch_list = self.build(feed_vars, 'infer')
            bbox_pred = fetch_list['bbox']
            return test_program, [bbox_pred, self.fm]
