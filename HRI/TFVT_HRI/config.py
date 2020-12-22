YOLOv3_img_resize_to = 320
YOLOv4_img_resize_to = 416


class XiaoduHiConfig(object):
    def __init__(self):
        # self.frame_shape = [480, 640, 3]
        self.frame_shape = [720, 1280, 3]  # raw video frame shape
        self.input_frame_shape = [360, 640, 3]  # resized shape for dataloader
        self.talk_emb_shape = [-1, 768]
        self.scene_sensor_algo = 'yolov4'

        self.ob_window_len = 10
        self.interval = 100.
        self.tokens_per_frame = 20
        self.roi_feat_resolution = 5
        self.det_confidence_th = 0.5

        self._backbone_strides = {
            'yolov3': 32,
            'yolov4': 32
        }

        self._backbone_filters = {
            'yolov3': 512,
            'yolov4': 512
        }

        self._opt_flow_stride = 16

    @property
    def img_shape(self):
        T = self.ob_window_len
        if self.scene_sensor_algo in ['yolov3', 'yolov4']:
            C = 3
            H = W = YOLOv3_img_resize_to
            if self.scene_sensor_algo == 'yolov4':
                H = W = YOLOv4_img_resize_to
        return [-1, T, C, H, W]

    @property
    def single_img_shape(self):
        _, _, C, H, W = self.img_shape
        return [-1, C, H, W]

    @property
    def feature_map_shape(self):
        stride = self._backbone_strides[self.scene_sensor_algo]
        _, T, _, H, W = self.img_shape
        C = self._backbone_filters[self.scene_sensor_algo]
        H = int(round(H / stride))
        W = int(round(W / stride))
        return [-1, T, C, H, W]

    @property
    def visual_token_dim(self):
        pos_emb_dim = 2 * self.roi_feat_resolution ** 2
        gap_dim = self._backbone_filters['yolov4']
        return pos_emb_dim + gap_dim
