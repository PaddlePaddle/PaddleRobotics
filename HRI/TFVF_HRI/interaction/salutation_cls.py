from paddle import fluid


class SalutationClsTree(object):
    # Tree structure: [[man (young_boy, uncle)],
    #                  [woman (young_girl, aunt)]]
    # output dim: 6
    def __init__(self,
                 fm,
                 hidden_dims=[512, 256],
                 name='salu_cls_tree',
                 hidden_act='relu'):
        self.hidden_dims = hidden_dims
        self.name = name
        self.hidden_act = hidden_act
        self._build_model(fm)

    def _build_model(self, fm):
        # 1x1 Conv to reduce dimension
        n = 128
        x = fluid.layers.conv2d(
            fm, num_filters=n, filter_size=1, act='relu',
            param_attr=fluid.ParamAttr(
                name=self.name + '_conv2d_0.w_0',
                initializer=fluid.initializer.Xavier(uniform=False)),
            bias_attr=fluid.ParamAttr(
                name=self.name + '_conv2d_0.b_0',
                initializer=fluid.initializer.Constant(value=0.0)))

        h, w = fm.shape[2:]
        x = fluid.layers.reshape(x, [-1, n * h * w])

        for layer_id, dim in enumerate(self.hidden_dims + [6]):
            if layer_id == 0:
                hidden = x

            if layer_id == len(self.hidden_dims):
                act = None
            else:
                act = self.hidden_act

            hidden = fluid.layers.fc(
                input=hidden, size=dim, act=act,
                param_attr=fluid.ParamAttr(
                    name=self.name + '_fc_%s.w_0' % layer_id,
                    initializer=fluid.initializer.Xavier(uniform=True)),
                bias_attr=fluid.ParamAttr(
                    name=self.name + '_fc_%s.b_0' % layer_id,
                    initializer=fluid.initializer.Constant(value=0.0)))

        self.root_pred = fluid.layers.softmax(hidden[:, :2])
        self.left_pred = fluid.layers.softmax(hidden[:, 2:4])
        self.right_pred = fluid.layers.softmax(hidden[:, 4:6])

    def predict(self):
        return [self.root_pred, self.left_pred, self.right_pred]

    def loss(self, tree_mask, root_cls, left_cls, right_cls,
             reduce_mean=True):
        root_loss = fluid.layers.cross_entropy(
            input=self.root_pred, label=root_cls)
        left_loss = fluid.layers.cross_entropy(
            input=self.left_pred, label=left_cls)
        right_loss = fluid.layers.cross_entropy(
            input=self.right_pred, label=right_cls)

        loss = fluid.layers.stack(
            [root_loss, left_loss, right_loss], axis=1)
        loss = fluid.layers.squeeze(loss, [2])
        loss = fluid.layers.elementwise_mul(loss, tree_mask)
        loss = fluid.layers.reduce_sum(loss, dim=1)

        if reduce_mean:
            return fluid.layers.reduce_mean(loss)
        else:
            loss
