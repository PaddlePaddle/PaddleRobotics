from paddle import fluid


class DiscreteController(object):
    def __init__(self, name, feat, output_dim, hidden_dims=[256],
                 hidden_act='relu'):
        assert output_dim > 1
        self.name = name
        self.hidden_dims = hidden_dims
        self.hidden_act = hidden_act
        self._build_model(feat, output_dim)

    def _build_model(self, feat, output_dim):
        for layer_id, dim in enumerate(self.hidden_dims + [output_dim]):
            if layer_id == 0:
                hidden = feat

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

        self.pred = fluid.layers.softmax(hidden, axis=-1)

    def predict(self):
        return self.pred

    def loss(self, label, reduce_mean=True):
        total_loss = fluid.layers.cross_entropy(input=self.pred, label=label)
        if reduce_mean:
            return fluid.layers.reduce_mean(total_loss)
        else:
            return total_loss
