import numpy as np
from paddle import fluid

from interaction.transformer import TransformerDecoder
from interaction.trigger import TriggerController
from perception.common.backbone import MobileNetV2


class AttentionController(object):
    def __init__(self,
                 inputs_type='visual_token',
                 num_actions=1000,
                 act_tr_dim=778,
                 act_emb_ndarray=None,
                 num_frames=10,
                 tokens_per_frame=20,
                 inst_crop_shape=[3, 128, 128],
                 inst_crop_flatten_dim=512,
                 inst_fm_shape=[512, 5, 5],
                 inst_fm_conv_reduce_dim=128,
                 inst_fm_flatten_dim=512,
                 inst_cls_dim=80,
                 inst_pos_dim=50,
                 visual_token_dim=562,
                 model_dim=512,
                 num_decoder_blocks=6,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0,
                 normalize_before=False,
                 frame_emb_trainable=True,
                 frame_sin_emb_temp=10000,
                 trigger_loss_coef=5.0,
                 obj_loss_coef=1.0,
                 act_loss_coef=1.0,
                 use_last_act_loss=False,
                 attn_mask_as_input=False,
                 attn_weights_as_output=False,
                 mode='train'):
        assert inputs_type in ['visual_token', 'instance', 'without_inst_fm',
                               'without_inst_cls', 'without_inst_pos',
                               'inst_crop', 'inst_crop_wo_crop',
                               'inst_crop_wo_cls', 'inst_crop_wo_pos']
        self.num_actions = num_actions
        self.act_tr_dim = act_tr_dim
        self.act_emb_ndarray = act_emb_ndarray
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.inst_crop_shape = inst_crop_shape
        self.inst_crop_flatten_dim = inst_crop_flatten_dim
        self.inst_fm_shape = inst_fm_shape
        self.inst_fm_conv_reduce_dim = inst_fm_conv_reduce_dim
        self.inst_fm_flatten_dim = inst_fm_flatten_dim
        self.inst_cls_dim = inst_cls_dim
        self.inst_pos_dim = inst_pos_dim
        self.visual_token_dim = visual_token_dim
        self.model_dim = model_dim
        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.normalize_before = normalize_before
        self.frame_emb_trainable = frame_emb_trainable
        self.frame_sin_emb_temp = frame_sin_emb_temp
        self.trigger_loss_coef = trigger_loss_coef
        self.obj_loss_coef = obj_loss_coef
        self.act_loss_coef = act_loss_coef
        self.use_last_act_loss = use_last_act_loss
        self.attn_mask_as_input = attn_mask_as_input
        self.attn_weights_as_output = attn_weights_as_output
        self.mode = mode

        self._build_input_layer(inputs_type)
        self._create_embeddings()
        self._build_model()

    def _build_input_layer(self, inputs_type):
        if self.mode in ['train', 'test']:
            bs = -1
            tgt_seq_len = self.num_frames * self.tokens_per_frame
        else:
            bs = 1
            tgt_seq_len = self.tokens_per_frame
            past_seq_len = -1

        # For ablation study
        inputs = []
        if inputs_type == 'visual_token':
            inputs.append('visual_token')
        elif inputs_type == 'instance':
            inputs.extend(['inst_fm', 'inst_cls', 'inst_pos_emb'])
        elif inputs_type == 'without_inst_fm':
            inputs.extend(['inst_cls', 'inst_pos_emb'])
        elif inputs_type == 'without_inst_cls':
            inputs.extend(['inst_fm', 'inst_pos_emb'])
        elif inputs_type == 'without_inst_pos':
            inputs.extend(['inst_fm', 'inst_cls'])
        elif inputs_type == 'inst_crop':
            inputs.extend(['inst_crop', 'inst_cls', 'inst_pos_emb'])
        elif inputs_type == 'inst_crop_wo_crop':
            inputs.extend(['inst_cls', 'inst_pos_emb'])
        elif inputs_type == 'inst_crop_wo_cls':
            inputs.extend(['inst_crop', 'inst_pos_emb'])
        elif inputs_type == 'inst_crop_wo_pos':
            inputs.extend(['inst_crop', 'inst_cls'])

        self.feed_list = []
        for i in inputs:
            if i == 'visual_token':
                # `visual_tokens`: are from GAP of RoIAligned feature map
                # and pos embedding of instances
                self.visual_tokens = fluid.data(
                    'visual_tokens', [bs, tgt_seq_len, self.visual_token_dim],
                    dtype='float32')
                self.feed_list.append(self.visual_tokens)
            elif i == 'inst_fm':
                self.inst_fm = fluid.data(
                    'inst_fm', [bs, tgt_seq_len] + self.inst_fm_shape,
                    dtype='float32')
                self.feed_list.append(self.inst_fm)
            elif i == 'inst_cls':
                self.inst_cls = fluid.data(
                    'inst_cls', [bs, tgt_seq_len, self.inst_cls_dim],
                    dtype='float32')
                self.feed_list.append(self.inst_cls)
            elif i == 'inst_pos_emb':
                self.inst_pos_emb = fluid.data(
                    'inst_pos_emb', [bs, tgt_seq_len, self.inst_pos_dim],
                    dtype='float32')
                self.feed_list.append(self.inst_pos_emb)
            elif i == 'inst_crop':
                self.inst_crop = fluid.data(
                    'inst_crop', [bs, tgt_seq_len] + self.inst_crop_shape,
                    dtype='float32')
                self.feed_list.append(self.inst_crop)

        self.frame_ids = fluid.data(
            'frame_ids', [bs, tgt_seq_len], dtype='int64')
        self.padding_mask = fluid.data(
            'padding_mask', [bs, tgt_seq_len], dtype='float32')
        self.feed_list.extend([self.frame_ids, self.padding_mask])

        if self.mode == 'train':
            self.past_kv_arr = None
            self.past_padding_mask = None

            nframes = tgt_seq_len // self.tokens_per_frame
            self.act_ids = fluid.data(
                'act_ids', [bs, nframes], dtype='int64')
            self.has_act = fluid.data(
                'has_act', [bs, nframes], dtype='float32')
            self.is_obj = fluid.data(
                'is_obj', [bs, tgt_seq_len], dtype='float32')
            self.feed_list.extend([self.act_ids, self.has_act, self.is_obj])
        elif self.mode == 'test':
            self.past_kv_arr = None
            self.past_padding_mask = None

            # temperature hyperparameter for softmax
            self.softmax_temp = fluid.data(
                'softmax_temp', [1], dtype='float32')
            self.top_k = fluid.data('top_k', [1], dtype='int64')
            self.feed_list.append(self.softmax_temp)

            if self.attn_mask_as_input:
                # NOTE: for jetson, as the converting from frame_ids
                # to attn_mask reequires py_func, which wouldn't
                # work for paddle inference.
                self.attn_mask = fluid.data(
                    'attn_mask', [bs, tgt_seq_len, tgt_seq_len],
                    dtype='float32')
                self.feed_list.append(self.attn_mask)
        elif self.mode == 'inference':
            # TODO: check whether this step by step inference works
            self.past_kv_arr = fluid.data(
                'past_kv_arr',
                [bs, self.num_decoder_blocks, 2, self.num_heads,
                 past_seq_len, self.model_dim // self.num_heads],
                dtype='float32')
            self.past_padding_mask = fluid.data(
                'past_padding_mask', [bs, past_seq_len], dtype='float32')
            self.feed_list.extend([self.past_kv_arr, self.past_padding_mask])

    def _create_embeddings(self):
        if self.frame_emb_trainable:
            emb_data = np.random.random((self.num_frames + 1, self.model_dim))
        else:
            raise NotImplementedError
            emb_data = None

        self.wfe = fluid.ParamAttr(
            name='wfe',
            learning_rate=0.0001 if self.frame_emb_trainable else 0.0,
            initializer=fluid.initializer.NumpyArrayInitializer(emb_data),
            trainable=self.frame_emb_trainable)

        # For inference, use emb after projection
        act_dim = self.act_tr_dim if self.mode in ['train', 'test'] \
            else self.model_dim

        if self.act_emb_ndarray is None:
            act_emb_ndarray = np.random.random(
                (self.num_actions + 1, act_dim))
        else:
            # Add zero pad
            act_emb_ndarray = np.concatenate(
                [self.act_emb_ndarray,
                 np.zeros((1, act_dim), dtype=np.float32)])
        self.wae = fluid.ParamAttr(
            name='wae',
            learning_rate=0.0001 if self.mode == 'train' else 0.0,
            initializer=fluid.initializer.NumpyArrayInitializer(
                act_emb_ndarray),
            trainable=self.mode == 'train')

    def _convert_frame_ids2attnmask(self, frame_ids):
        # NOTE: make sure the frame_ids is non-decreasing
        # e.g. [1, 1, 2, 2, 1, 3] is not allowed
        def _idlst2mask(ids):
            # e.g. convert to [1, 1, 2, 2]
            # [[1, 1, 0, 0],
            #  [1, 1, 0, 0],
            #  [1, 1, 1, 1],
            #  [1, 1, 1, 1]]
            n = len(ids)
            repeat, r, c = [0], 0, ids[0]
            for i in ids:
                if i == c:
                    r += 1
                else:
                    repeat.append(r)
                    r, c = 1, i
            repeat.append(r)
            cum = np.cumsum(repeat)

            mask = np.zeros((n, n), dtype=np.float32)
            for i in range(1, len(cum)):
                mask[cum[i-1]:cum[i], 0:cum[i]] = 1.0
            return mask

        def _np_func(batch_ids):
            batch_ids = np.array(batch_ids)
            mask_lst = list(map(_idlst2mask, list(batch_ids)))
            return np.array(mask_lst, dtype=np.float32)

        shape = list(frame_ids.shape) + [frame_ids.shape[-1]]
        attn_mask = fluid.default_main_program().current_block().create_var(
            name='attn_mask', dtype='float32', shape=shape)
        fluid.layers.py_func(func=_np_func, x=[frame_ids], out=attn_mask)
        return attn_mask

    def _top_k_sampling(self, logits, top_k, null_act_idx=0):
        null_act_idx = fluid.layers.fill_constant(
            [1, 1], 'int64', null_act_idx)
        non_null_act = 1.0 - fluid.layers.one_hot(
            null_act_idx, logits.shape[-1])
        non_null_act = fluid.layers.squeeze(non_null_act, [0])

        # logits for actions that are not null
        non_null_logits = logits * non_null_act - 1e10 * (1 - non_null_act)
        probs = fluid.layers.softmax(non_null_logits, use_cudnn=False)

        topk_probs, _ = fluid.layers.topk(probs, top_k)
        ge_cond = fluid.layers.cast(
            fluid.layers.greater_equal(
                probs,
                fluid.layers.unsqueeze(topk_probs[:, :, -1], [2])),
            'float32')
        probs = probs * ge_cond / fluid.layers.reduce_sum(
            topk_probs, dim=-1, keep_dim=True)

        _, nframes, dim = probs.shape
        sampling_ids = fluid.layers.sampling_id(
            fluid.layers.reshape(probs, [-1, dim]))
        sampling_ids = fluid.layers.reshape(sampling_ids, [-1, nframes])
        sampling_ids = fluid.layers.cast(sampling_ids, 'float32')
        return sampling_ids

    def _build_model(self):
        self.decoder = TransformerDecoder(
            self.num_decoder_blocks, self.model_dim,
            self.num_heads, self.ffn_dim,
            dropout=self.dropout,
            tokens_per_frame=self.tokens_per_frame,
            normalize_before=self.normalize_before)

        if self.attn_mask_as_input:
            attn_mask = self.attn_mask
        else:
            attn_mask = self._convert_frame_ids2attnmask(self.frame_ids)

        frame_emb = fluid.embedding(
            self.frame_ids, (self.num_frames + 1, self.model_dim),
            padding_idx=0, param_attr=self.wfe, dtype='float32')

        if hasattr(self, 'visual_tokens'):
            visual_tokens = fluid.layers.fc(
                self.visual_tokens, self.model_dim,
                num_flatten_dims=2, bias_attr=False, name='vt_fc')
        else:
            inputs = []  # save flatten inputs

            if hasattr(self, 'inst_fm'):
                tgt_seq_len = self.inst_fm.shape[1]
                inst_fm = fluid.layers.reshape(
                    self.inst_fm, [-1] + self.inst_fm_shape)
                inst_fm = fluid.layers.conv2d(
                    inst_fm, num_filters=self.inst_fm_conv_reduce_dim,
                    filter_size=1, act='relu',
                    param_attr=fluid.ParamAttr(
                        name='inst_fm_conv2d_0.w_0',
                        initializer=fluid.initializer.Xavier(uniform=False)),
                    bias_attr=fluid.ParamAttr(
                        name='inst_fm_conv2d_0.b_0',
                        initializer=fluid.initializer.Constant(value=0.0)))
                inst_fm = fluid.layers.reshape(
                    inst_fm, [-1, tgt_seq_len] + list(inst_fm.shape)[1:])

                inst_fm = fluid.layers.fc(
                    inst_fm, self.inst_fm_flatten_dim,
                    num_flatten_dims=2, act='relu',
                    param_attr=fluid.ParamAttr(
                        name='inst_fm_fc_0.w_0',
                        initializer=fluid.initializer.Xavier(uniform=True)),
                    bias_attr=fluid.ParamAttr(
                        name='inst_fm_fc_0.b_0',
                        initializer=fluid.initializer.Constant(value=0.0)))
                inputs.append(inst_fm)

            if hasattr(self, 'inst_crop'):
                tgt_seq_len = self.inst_crop.shape[1]
                inst_crop = fluid.layers.reshape(
                    self.inst_crop, [-1] + self.inst_crop_shape)
                inst_encoder = MobileNetV2()
                crop_feat = inst_encoder(inst_crop)
                crop_feat = fluid.layers.reshape(
                    crop_feat, [-1, tgt_seq_len] + list(crop_feat.shape)[1:])

                crop_feat = fluid.layers.fc(
                    crop_feat, self.inst_crop_flatten_dim,
                    num_flatten_dims=2, act='relu',
                    param_attr=fluid.ParamAttr(
                        name='inst_crop_fc_0.w_0',
                        initializer=fluid.initializer.Xavier(uniform=True)),
                    bias_attr=fluid.ParamAttr(
                        name='inst_crop_fc_0.b_0',
                        initializer=fluid.initializer.Constant(value=0.0)))
                inputs.append(crop_feat)

            if hasattr(self, 'inst_cls'):
                inputs.append(self.inst_cls)

            if hasattr(self, 'inst_pos_emb'):
                inputs.append(self.inst_pos_emb)

            visual_tokens = fluid.layers.concat(inputs, axis=-1)
            visual_tokens = fluid.layers.fc(
                visual_tokens, self.model_dim,
                num_flatten_dims=2, act='relu',
                param_attr=fluid.ParamAttr(
                    name='inst_vt_fc.w',
                    initializer=fluid.initializer.Xavier(uniform=True)),
                bias_attr=fluid.ParamAttr(
                    name='inst_vt_fc.b',
                    initializer=fluid.initializer.Constant(value=0.0)))

        hid, frame_hid, present_kv_arr, self.attn_weights_arr = self.decoder(
            visual_tokens, frame_emb, attn_mask, self.padding_mask,
            past_kv_arr=self.past_kv_arr,
            past_padding_mask=self.past_padding_mask)

        self.trigger = TriggerController(frame_hid, name='trigger')
        self.obj_cls = TriggerController(hid, name='obj_cls')

        if self.mode in ['train', 'test']:
            # TODO: @paddle-dev, check alternative implementation
            # to use emb in fc, instead of using manually converted variable
            wae = fluid.embedding(
                fluid.layers.arange(0, self.num_actions, dtype='int64'),
                (self.num_actions+1, self.act_tr_dim),
                padding_idx=-1, param_attr=self.wae, dtype='float32')
            wae_proj = fluid.layers.fc(wae, self.model_dim, name='wae_proj')
        else:
            # NOTE: save projected emb when save inference model
            # and reuse it as new self.wae
            wae_proj = fluid.embedding(
                fluid.layers.arange(0, self.num_actions, dtype='int64'),
                (self.num_actions+1, self.model_dim),
                padding_idx=-1, param_attr=self.wae, dtype='float32')

        wae_proj = fluid.layers.reshape(
            wae_proj, [self.num_actions, self.model_dim])
        act_logits = fluid.layers.matmul(
            frame_hid, wae_proj, transpose_y=True)
        # act_pred: [bs, nframes, num_actions]
        # NOTE: it seems for paddle-1.8, use_cudnn=True by default
        # and it can lead to nan
        if self.mode == 'test':
            after_temp = act_logits / self.softmax_temp
            top_k = fluid.layers.cast(self.top_k, 'int32')
            self.act_pred = fluid.layers.softmax(
                after_temp, use_cudnn=False)
            self.act_topk_sampling = self._top_k_sampling(after_temp, top_k)
        else:
            self.act_pred = fluid.layers.softmax(
                act_logits, use_cudnn=False)

        if self.mode == 'train':
            self.trigger_loss = self.trigger.loss(self.has_act)

            obj_loss = self.obj_cls.loss(self.is_obj, reduce_mean=False)
            obj_loss = fluid.layers.elementwise_mul(
                obj_loss, self.padding_mask)
            self.obj_loss = fluid.layers.reduce_mean(obj_loss)

            act_logits = fluid.layers.log(self.act_pred)
            act_ids_flatten = fluid.layers.reshape(self.act_ids, [-1, 1])
            act_ids = fluid.layers.reshape(
                fluid.layers.one_hot(act_ids_flatten, self.num_actions),
                [-1, act_logits.shape[1], self.num_actions])
            nll = fluid.layers.elementwise_mul(act_logits, act_ids) * -1.0
            if self.use_last_act_loss:
                nll = fluid.layers.reduce_sum(nll, dim=2)
                nll = nll[:, -1]
            else:
                nll = fluid.layers.reduce_sum(nll, dim=[1, 2]) / \
                    self.num_frames
            self.act_loss = fluid.layers.reduce_mean(nll)

            self.loss = self.trigger_loss_coef * self.trigger_loss + \
                self.obj_loss_coef * self.obj_loss + \
                self.act_loss_coef * self.act_loss

    def predict(self):
        obj_pred = fluid.layers.elementwise_mul(
            self.obj_cls.predict(), self.padding_mask, axis=0)
        preds = [self.trigger.predict(), obj_pred, self.act_pred]

        if self.mode == 'test':
            preds.append(self.act_topk_sampling)

        if self.attn_weights_as_output:
            preds.append(self.attn_weights_arr)

        return preds
