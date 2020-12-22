import os
from paddle import fluid

from perception.scene.eval import SceneSensor
from interaction.attention_ctrl import AttentionController
from interaction.salutation_cls import SalutationClsTree


class AttnModelServiceProgram(object):
    def __init__(self,
                 yolov4_model_dir,
                 attn_model_dir,
                 inputs_type='visual_token',
                 salutation_model_dir=None,
                 feature_map_channel=512,
                 view_shape=[3, 360, 640],
                 image_shape=[3, 416, 416],
                 det_confidence_th=0.5,
                 roi_feat_resolution=5,
                 num_actions=317,
                 num_frames=10,
                 tokens_per_frame=20,
                 visual_token_dim=562,
                 model_dim=512,
                 num_decoder_blocks=6,
                 num_heads=8,
                 ffn_dim=2048,
                 normalize_before=False,
                 with_salutation_cls=False,
                 salutation_cls_hiddens=[512, 256]):
        self.yolov4_model_dir = yolov4_model_dir
        self.attn_model_dir = attn_model_dir
        self.inputs_type = inputs_type

        # Conf for yolov4
        self.feature_map_channel = feature_map_channel
        self.view_shape = view_shape
        self.input_shape = image_shape
        self.det_confidence_th = det_confidence_th
        self.roi_feat_resolution = roi_feat_resolution

        # Conf for attention controller
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.visual_token_dim = visual_token_dim
        self.model_dim = model_dim
        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.normalize_before = normalize_before

        # Conf for salutation classifier
        if with_salutation_cls:
            assert salutation_model_dir is not None
        self.with_salutation_cls = with_salutation_cls
        self.salutation_model_dir = salutation_model_dir
        self.salutation_cls_hiddens = salutation_cls_hiddens

        self.called_init_params = False

        self._build_detector_program()
        self._build_attn_ctrl_program()

        if not self.inputs_type.startswith('inst_crop'):
            self._build_visual_tokenizer_program()

    def _build_detector_program(self):
        self.detector_prog = fluid.Program()
        self.detector_startup_prog = fluid.Program()
        with fluid.program_guard(self.detector_prog,
                                 self.detector_startup_prog):
            yolov4_detector = SceneSensor.network(
                self.input_shape, 'yolov4',
                get_roi_feat=False,
                roi_feat_resolution=self.roi_feat_resolution)
            feed_list, fetch_list = yolov4_detector.build()
            self.detector_feeds = [i.name for i in feed_list]
            self.detector_fetch = fetch_list

    def _build_visual_tokenizer_program(self):
        self.visual_tokenizer_prog = fluid.Program()
        self.visual_tokenizer_startup_prog = fluid.Program()
        with fluid.program_guard(self.visual_tokenizer_prog,
                                 self.visual_tokenizer_startup_prog):
            _, h, w = self.input_shape
            c = self.feature_map_channel
            fm = fluid.data(
                name='fm', shape=[-1, c, h // 32, w // 32], dtype='float32')
            pred = fluid.data(
                name='pred', shape=[-1, 6], dtype='float32', lod_level=2)
            rois = fluid.layers.slice(
                pred, axes=[1], starts=[2], ends=[6])
            self.visual_tokenizer_feeds = ['fm', 'pred']

            # xmin = fluid.layers.slice(
            #     pred, axes=[1], starts=[2], ends=[3])
            # ymin = fluid.layers.slice(
            #     pred, axes=[1], starts=[3], ends=[4])
            # xmax = fluid.layers.slice(
            #     pred, axes=[1], starts=[4], ends=[5])
            # ymax = fluid.layers.slice(
            #     pred, axes=[1], starts=[5], ends=[6])

            # h, w = self.view_shape[1:]
            # new_xmin = fluid.layers.elementwise_max(
            #     xmin, fluid.layers.full_like(xmin, 0.0))
            # new_ymin = fluid.layers.elementwise_max(
            #     ymin, fluid.layers.full_like(ymin, 0.0))
            # new_xmax = fluid.layers.elementwise_min(
            #     xmax, fluid.layers.full_like(xmax, w - 1.0))
            # new_ymax = fluid.layers.elementwise_min(
            #     ymax, fluid.layers.full_like(ymax, h - 1.0))
            # rois = fluid.layers.concat(
            #     [new_xmin, new_ymin, new_xmax, new_ymax], axis=-1)

            offsets, scales = self._get_offsets_and_scales()
            corrected_rois = fluid.layers.elementwise_add(
                fluid.layers.elementwise_mul(rois, scales, axis=1),
                offsets, axis=1)

            rois_fm = fluid.layers.roi_align(
                input=fm,
                rois=corrected_rois,
                pooled_height=self.roi_feat_resolution,
                pooled_width=self.roi_feat_resolution,
                spatial_scale=1/32.,
                sampling_ratio=-1)

            if self.inputs_type == 'visual_token':
                # GAP
                rois_feats = fluid.layers.reduce_sum(rois_fm, dim=[2, 3]) / \
                    (self.roi_feat_resolution ** 2)
                self.visual_tokenizer_fetch = [rois_feats]
            else:
                self.visual_tokenizer_fetch = [rois_fm]

            if self.with_salutation_cls:
                salutation_cls = SalutationClsTree(
                    rois_fm, hidden_dims=self.salutation_cls_hiddens)
                self.visual_tokenizer_fetch.extend(salutation_cls.predict())

    def _build_attn_ctrl_program(self):
        self.attn_ctrl_prog = fluid.Program()
        self.attn_ctrl_startup_prog = fluid.Program()
        with fluid.program_guard(self.attn_ctrl_prog,
                                 self.attn_ctrl_startup_prog):
            attn_ctrl = AttentionController(
                inputs_type=self.inputs_type,
                num_actions=self.num_actions,
                num_frames=self.num_frames,
                tokens_per_frame=self.tokens_per_frame,
                visual_token_dim=self.visual_token_dim,
                model_dim=self.model_dim,
                num_decoder_blocks=self.num_decoder_blocks,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                normalize_before=self.normalize_before,
                attn_mask_as_input=True,
                attn_weights_as_output=True,
                mode='test')
            self.attn_ctrl_feeds = [i.name for i in attn_ctrl.feed_list]
            self.attn_ctrl_fetch = attn_ctrl.predict()

    def _get_offsets_and_scales(self):
        # NOTE: this works for inference mode, in which
        # view shape is same for each image
        _, h0, w0 = self.view_shape
        _, h1, w1 = self.input_shape
        aspect = min(h1 / h0, w1 / w0)
        h0_, w0_ = int(h0 * aspect), int(w0 * aspect)

        x_offset, y_offset = 0.5 * (w1 - w0_), 0.5 * (h1 - h0_)
        x_scale, y_scale = w0_ / w0, h0_ / h0

        x_offset = fluid.layers.fill_constant([1], 'float32', x_offset)
        y_offset = fluid.layers.fill_constant([1], 'float32', y_offset)
        x_scale = fluid.layers.fill_constant([1], 'float32', x_scale)
        y_scale = fluid.layers.fill_constant([1], 'float32', y_scale)

        offsets = fluid.layers.concat(
            [x_offset, y_offset, x_offset, y_offset], axis=0)
        scales = fluid.layers.concat(
            [x_scale, y_scale, x_scale, y_scale], axis=0)
        return offsets, scales

    def init_params(self, exe):
        exe.run(self.detector_startup_prog)
        if not self.inputs_type.startswith('inst_crop'):
            exe.run(self.visual_tokenizer_startup_prog)
        exe.run(self.attn_ctrl_startup_prog)

        fluid.io.load_params(exe, self.yolov4_model_dir,
                             main_program=self.detector_prog)

        if self.with_salutation_cls and \
           not self.inputs_type.startswith('inst_crop'):
            fluid.io.load_params(exe, self.salutation_model_dir,
                                 main_program=self.visual_tokenizer_prog)

        fluid.io.load_params(exe, self.attn_model_dir,
                             main_program=self.attn_ctrl_prog)

        self.called_init_params = True

    def save_inference_models(self, exe, output_dir):
        if not self.called_init_params:
            raise RuntimeError('You need to call `init_params` at first.')

        fluid.io.save_inference_model(
            output_dir, self.detector_feeds, self.detector_fetch, exe,
            main_program=self.detector_prog,
            model_filename='detector_model',
            params_filename='detector_params')

        fluid.io.save_inference_model(
            output_dir, self.attn_ctrl_feeds, self.attn_ctrl_fetch, exe,
            main_program=self.attn_ctrl_prog,
            model_filename='attn_ctrl_model',
            params_filename='attn_ctrl_params')

        if not self.inputs_type.startswith('inst_crop'):
            if self.with_salutation_cls:
                fluid.io.save_inference_model(
                    output_dir, self.visual_tokenizer_feeds,
                    self.visual_tokenizer_fetch, exe,
                    main_program=self.visual_tokenizer_prog,
                    model_filename='visual_tokenizer_model',
                    params_filename='visual_tokenizer_params')
            else:
                fluid.io.save_inference_model(
                    os.path.join(output_dir, 'visual_tokenizer_model'),
                    self.visual_tokenizer_feeds,
                    self.visual_tokenizer_fetch, exe,
                    main_program=self.visual_tokenizer_prog,
                    program_only=False)
