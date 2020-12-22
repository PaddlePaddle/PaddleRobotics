from paddle import fluid


class TriggerController(object):
    def __init__(self, feat, hidden_dims=[256], hidden_act='relu',
                 name='trigger'):
        self.name = name
        self.hidden_dims = hidden_dims
        self.hidden_act = hidden_act
        self._build_model(feat)

    def _build_model(self, feat):
        num_flatten_dims = 1
        if len(feat.shape) == 3:
            # For attentional controller, see attention_ctrl.py
            num_flatten_dims = 2

        for layer_id, dim in enumerate(self.hidden_dims + [1]):
            if layer_id == 0:
                hidden = feat

            if layer_id == len(self.hidden_dims):
                act = None
            else:
                act = self.hidden_act

            hidden = fluid.layers.fc(
                input=hidden, size=dim, act=act,
                num_flatten_dims=num_flatten_dims,
                param_attr=fluid.ParamAttr(
                    name=self.name + '_fc_%s.w_0' % layer_id,
                    initializer=fluid.initializer.Xavier(uniform=True)),
                bias_attr=fluid.ParamAttr(
                    name=self.name + '_fc_%s.b_0' % layer_id,
                    initializer=fluid.initializer.Constant(value=0.0)))

        self.logits = hidden
        self.pred = fluid.layers.sigmoid(hidden)

    def loss(self, label, reduce_mean=True):
        if len(self.logits.shape) == 3:
            logits = fluid.layers.squeeze(self.logits, [2])
        else:
            logits = self.logits

        total_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            logits, label)
        if reduce_mean:
            return fluid.layers.reduce_mean(total_loss)
        else:
            return total_loss

    def predict(self):
        return self.pred
