import numpy as np
from paddle import fluid


class MaskedMultiHeadAttention(object):
    def __init__(self, model_dim, num_heads, dropout=0.0):
        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.per_head_dim = model_dim // num_heads
        self.dropout = dropout

    def _split(self, x):
        """Split state to query, key and value"""
        _, seq_len, qkv_dim = x.shape
        x = fluid.layers.reshape(
            x, [-1, seq_len, 3, qkv_dim // 3], inplace=True)
        return fluid.layers.unstack(x, axis=2)

    def _split_heads(self, x):
        """Split single head for multi-heads"""
        split_x = fluid.layers.reshape(
            x, [0, 0, self.num_heads, self.per_head_dim], inplace=True)
        split_x = fluid.layers.transpose(split_x, perm=[0, 2, 1, 3])
        return split_x

    def _merge_heads(self, x):
        """Merge multi-heads for single head"""
        merged_x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])
        merged_dim = merged_x.shape[2] * merged_x.shape[3]
        merged_x = fluid.layers.reshape(
            merged_x, [0, 0, merged_dim], inplace=True)
        return merged_x

    def _apply_attn_score_mask(self, product, attn_mask):
        product = product * attn_mask - 1e10 * (1 - attn_mask)
        return product

    def _scaled_dot_product_attention(self, query, key, value, attn_mask,
                                      d_key, attn_bias=None, dropout=0.0):
        # Q is in shape [bs, nheads, tgt_seq_len, per_head_dim]
        # K and V are in shape [bs, nheads, src_seq_len, per_head_dim]
        # attn_mask is in shape [bs, tgt_seq_len, src_seq_len]
        product = fluid.layers.matmul(query, key, transpose_y=True,
                                      alpha=d_key**-0.5)
        if attn_bias is not None:
            product += attn_bias

        attn_mask = fluid.layers.expand(
            fluid.layers.unsqueeze(attn_mask, axes=[1]),
            [1, self.num_heads, 1, 1])
        product = self._apply_attn_score_mask(product, attn_mask)

        # weights is in shape [bs, nheads, tgt_seq_len, src_seq_len]
        weights = fluid.layers.softmax(product)
        weights = weights * attn_mask
        if dropout > 0:
            weights = fluid.layers.dropout(
                weights, dropout, dropout_implementation='upscale_in_train')

        # attn is in shape [bs, nheads, tgt_seq_len, per_head_dim]
        attn = fluid.layers.matmul(weights, value)
        return attn, weights

    def __call__(self, x, attn_mask, past_kv=None, attn_bias=None):
        # Parameters:
        # qkv_fc: x_dim * model_dim * 3
        # scaled_dot_product_attention: 0
        # out_fc: model_dim * model_dim

        # Computation (assume bs = 1):
        # let N1 = tgt_seq_len * x_dim * model_dim * 3
        #     N2 = nheads * tgt_seq_len * src_seq_len
        #     N3 = nheads * tgt_seq_len * per_head_dim = tl * md
        #     ph = per_head_dim; md = model_dim
        #     sl = src_seq_len; tl = tgt_seq_len
        # qkv_fc: N1 * (model_dim (mul_op) + model_dim (add_op))
        # scaled_dot_product_attention:
        #     N2 * (2*ph(mul_op) + 2*ph(add_op) + ph(div_op) + ph(exp_op))
        #     + N3 * (sl (mul_op) + sl (add_op))
        # out_fc: tl * md * (md (mul_op) + md (add_op))
        #
        # for sl = tl = 200, md = 512, ph = 64, nh = 8, around 10^12

        assert len(x.shape) == 3
        # TODO: add customize parameter initializer for QKV project.
        c = fluid.layers.fc(x, self.model_dim * 3, num_flatten_dims=2,
                            bias_attr=False, name='qkv_fc')
        # Q, K, V is in shape [bs, tgt_seq_len, model_dim]
        # attn_mask is in shape [bs, tgt_seq_len, src_seq_len]
        # past_kv is None or in [bs, 2, nheads, past_seq_len, per_head_dim]
        # when past_kv is None, tgt_seq_len = src_seq_len
        # otherwise, src_seq_len = past_seq_len + tgt_seq_len
        query, key, value = self._split(c)
        assert len(query.shape) == len(key.shape) == len(value.shape) == 3

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        present_kv = fluid.layers.stack([key, value], axis=1)

        if past_kv is not None:
            pk, pv = fluid.layers.unstack(past_kv, axis=1)
            key = fluid.layers.concat([pk, key], axis=-2)
            value = fluid.layers.concat([pv, value], axis=-2)

        attn, attn_weights = self._scaled_dot_product_attention(
            query, key, value, attn_mask, self.per_head_dim,
            attn_bias=attn_bias, dropout=self.dropout)
        attn = self._merge_heads(attn)
        attn = fluid.layers.fc(attn, self.model_dim, num_flatten_dims=2,
                               bias_attr=False, name='out_fc')

        return attn, present_kv, attn_weights


class TransformerDecoderBlock(object):
    def __init__(self, model_dim, num_heads, ffn_dim,
                 dropout=0.0, normalize_before=False):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.normalize_before = normalize_before

        self.masked_self_attn = MaskedMultiHeadAttention(
            model_dim, num_heads, dropout=dropout)

    def _merge_mask(self, attn_mask, padding_mask):
        pm = fluid.layers.unsqueeze(padding_mask, 2)
        pm_t = fluid.layers.unsqueeze(padding_mask, 1)
        new_pm = fluid.layers.matmul(pm, pm_t)
        attn_mask = fluid.layers.elementwise_mul(attn_mask, new_pm, axis=0)
        return attn_mask

    def _with_frame_emb(self, x, frame_emb):
        return x if frame_emb is None else x + frame_emb

    def _pad_past_attn_mask(self, attn_mask, past_padding_mask):
        def _np_func(m, pm):
            m, pm = np.array(m), np.array(pm)
            pad = np.ones((m.shape[0], m.shape[1], pm.shape[1]),
                          dtype=m.dtype)
            m_ = np.concatenate([pad, m], axis=2)
            return m_

        name = fluid.unique_name.generate(attn_mask.name)
        new_mask = fluid.default_main_program().current_block().create_var(
            name=name, dtype=attn_mask.dtype, shape=attn_mask.shape)
        fluid.layers.py_func(
            func=_np_func, x=[attn_mask, past_padding_mask], out=new_mask)
        return new_mask

    def _mlp(self, x, n_state, dropout=0.0):
        # TODO: try other activation
        nx = x.shape[-1]
        h1 = fluid.layers.fc(x, n_state, num_flatten_dims=2, act='gelu')
        if dropout > 0:
            h1 = fluid.layers.dropout(
                h1, dropout, dropout_implementation='upscale_in_train')
        h2 = fluid.layers.fc(h1, nx, num_flatten_dims=2)
        return h2

    def _forward_post(self, x, frame_emb, attn_mask, padding_mask,
                      past_kv, past_padding_mask):
        x = self._with_frame_emb(x, frame_emb)
        if past_padding_mask is not None:
            # [bs, src_seq_len]
            padding_mask = fluid.layers.concat(
                [past_padding_mask, padding_mask], axis=-1)

        attn_mask = self._merge_mask(attn_mask, padding_mask)
        attn, present_kv, attn_weights = self.masked_self_attn(
            x, attn_mask, past_kv=past_kv)
        if self.dropout > 0:
            attn = fluid.layers.dropout(
                attn, self.dropout, dropout_implementation='upscale_in_train')

        x = x + attn
        x = fluid.layers.layer_norm(
            x, begin_norm_axis=2, epsilon=1e-6,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.)))

        m = self._mlp(x, self.ffn_dim, dropout=self.dropout)
        if self.dropout > 0:
            m = fluid.layers.dropout(
                m, self.dropout, dropout_implementation='upscale_in_train')
        x = x + m
        x = fluid.layers.layer_norm(
            x, begin_norm_axis=2, epsilon=1e-6,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.)))
        return x, present_kv, attn_weights

    def _forward_pre(self, x, frame_emb, attn_mask, padding_mask,
                     past_kv, past_padding_mask):
        x_ = fluid.layers.layer_norm(
            x, begin_norm_axis=2, epsilon=1e-6,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.)))
        x_ = self._with_frame_emb(x_, frame_emb)

        if past_padding_mask is not None:
            # [bs, src_seq_len]
            padding_mask = fluid.layers.concat(
                [past_padding_mask, padding_mask], axis=-1)

        attn_mask = self._merge_mask(attn_mask, padding_mask)
        attn, present_kv, attn_weights = self.masked_self_attn(
            x_, attn_mask, past_kv=past_kv)
        if self.dropout > 0:
            attn = fluid.layers.dropout(
                attn, self.dropout, dropout_implementation='upscale_in_train')

        x = x + attn
        x_ = fluid.layers.layer_norm(
            x, begin_norm_axis=2, epsilon=1e-6,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.)))

        m = self._mlp(x_, self.ffn_dim, dropout=self.dropout)
        if self.dropout > 0:
            m = fluid.layers.dropout(
                m, self.dropout, dropout_implementation='upscale_in_train')

        x = x + m
        return x, present_kv, attn_weights

    def __call__(self, x, frame_emb, attn_mask, padding_mask,
                 past_kv=None, past_padding_mask=None):
        # x: [bs, tgt_seq_len, model_dim]
        # frame_emb: [bs, tgt_seq_len, model_dim]
        # attn_mask: [bs, tgt_seq_len, tgt_seq_len]
        # padding_mask: [bs, tgt_seq_len]
        # past_kv: [bs, 2, nheads, past_seq_len, per_head_dim]
        # past_padding_mask: [bs, past_seq_len]
        # src_seq_len = tgt_seq_len + past_seq_len
        if past_padding_mask is not None:
            # Now attn_mask: [bs, tgt_seq_len, src_seq_len]
            attn_mask = self._pad_past_attn_mask(attn_mask, past_padding_mask)
        if self.normalize_before:
            return self._forward_pre(x, frame_emb, attn_mask,
                                     padding_mask, past_kv,
                                     past_padding_mask)

        return self._forward_post(x, frame_emb, attn_mask, padding_mask,
                                  past_kv, past_padding_mask)


class TransformerDecoder(object):
    def __init__(self, num_blocks, model_dim, num_heads, ffn_dim,
                 tokens_per_frame=10, dropout=0.0, normalize_before=False):
        self.num_blocks = num_blocks
        self.tokens_per_frame = tokens_per_frame

        self.blocks = []
        for _ in range(num_blocks):
            decoder_block = TransformerDecoderBlock(
                model_dim, num_heads, ffn_dim,
                dropout=dropout,
                normalize_before=normalize_before)
            self.blocks.append(decoder_block)

    def _apply_padding_mask(self, x, padding_mask):
        h = padding_mask * x - 1e10 * (1 - padding_mask)
        return h

    def _pooling_over_frames(self, x, padding_mask):
        _, tgt_seq_len, model_dim = x.shape
        num_frames = tgt_seq_len // self.tokens_per_frame

        padding_mask_ = fluid.layers.expand(
            fluid.layers.unsqueeze(padding_mask, [2]),
            [1, 1, model_dim])
        h = self._apply_padding_mask(x, padding_mask_)
        h = fluid.layers.reshape(
            h, [-1, num_frames, self.tokens_per_frame, model_dim])
        h = fluid.layers.reduce_max(h, dim=2)
        return h

    def __call__(self, x, frame_emb, attn_mask, padding_mask,
                 past_kv_arr=None, past_padding_mask=None):
        assert x.shape[1] % self.tokens_per_frame == 0
        if past_kv_arr is not None:
            past_kv_arr = fluid.layers.unstack(past_kv_arr, axis=1)
        else:
            past_kv_arr = [None] * self.num_blocks

        present_kv_lst, attn_weights_lst = [], []
        for i, past_kv in enumerate(past_kv_arr):
            x, present_kv, attn_weights = self.blocks[i](
                x, frame_emb, attn_mask, padding_mask,
                past_kv=past_kv, past_padding_mask=past_padding_mask)
            present_kv_lst.append(present_kv)
            attn_weights_lst.append(attn_weights)

        present_kv_arr = fluid.layers.stack(present_kv_lst, axis=1)
        attn_weights_arr = fluid.layers.stack(attn_weights_lst, axis=1)

        frame_hid = self._pooling_over_frames(x, padding_mask)
        return x, frame_hid, present_kv_arr, attn_weights_arr
