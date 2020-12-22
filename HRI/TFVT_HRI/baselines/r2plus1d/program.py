"""
NOTE: Assume the Paddle version has been exported as folder
`r2plus1d_wae_paddle`. Use evaluate.ipnb to do so.
"""
import os
import sys
from paddle import fluid

x2paddle_output_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'r2plus1d_wae_paddle'))
sys.path.append(x2paddle_output_dir)

from model_with_code.model import x2paddle_net as pd_r2plus1d


class R2plus1DProgram(object):
    def __init__(self,
                 model_dir=os.path.join(
                     x2paddle_output_dir, 'model_with_code')):
        self.model_dir = model_dir
        self.called_init_params = False

        self._build_ctrl_program()

    def _build_ctrl_program(self):
        self.ctrl_prog = fluid.Program()
        self.ctrl_startup_prog = fluid.Program()
        with fluid.program_guard(self.ctrl_prog, self.ctrl_startup_prog):
            inputs, outputs = pd_r2plus1d()

            softmax_temp = fluid.data(
                'softmax_temp', [1], dtype='float32')
            top_k = fluid.data('top_k', [1], dtype='int64')
            inputs.extend([softmax_temp, top_k])

            logits = outputs[0]
            after_temp = logits / softmax_temp
            pred = fluid.layers.softmax(after_temp, use_cudnn=False)
            topk_sampling = self._top_k_sampling(after_temp, top_k)

            self.ctrl_feeds = [i.name for i in inputs]
            self.ctrl_fetch = [pred, topk_sampling]

    def _top_k_sampling(self, logits, top_k, null_act_idx=0):
        null_act_idx = fluid.layers.fill_constant(
            [1, 1], 'int64', null_act_idx)
        non_null_act = 1.0 - fluid.layers.one_hot(
            null_act_idx, logits.shape[-1])
        non_null_act = fluid.layers.squeeze(non_null_act, [0])

        # logits for actions that are not null
        non_null_logits = logits * non_null_act - 1e10 * (1 - non_null_act)
        probs = fluid.layers.softmax(non_null_logits, use_cudnn=False)

        top_k = fluid.layers.cast(top_k, 'int32')
        topk_probs, _ = fluid.layers.topk(probs, top_k)
        ge_cond = fluid.layers.cast(
            fluid.layers.greater_equal(
                probs,
                fluid.layers.unsqueeze(topk_probs[:, -1], [1])),
            'float32')
        probs = probs * ge_cond / fluid.layers.reduce_sum(
            topk_probs, dim=-1, keep_dim=True)

        sampling_ids = fluid.layers.sampling_id(probs)
        sampling_ids = fluid.layers.cast(sampling_ids, 'float32')
        return sampling_ids

    def init_params(self, exe):
        exe.run(self.ctrl_startup_prog)

        def _if_exist(var):
            return os.path.exists(os.path.join(self.model_dir, var.name))

        fluid.io.load_vars(exe, self.model_dir,
                           main_program=self.ctrl_prog,
                           predicate=_if_exist)
        self.called_init_params = True

    def save_inference_models(self, exe, output_dir):
        if not self.called_init_params:
            raise RuntimeError('You need to call `init_params` at first.')

        fluid.io.save_inference_model(
            output_dir, self.ctrl_feeds, self.ctrl_fetch, exe,
            main_program=self.ctrl_prog,
            model_filename='ctrl_model',
            params_filename='ctrl_params')


if __name__ == '__main__':
    program = R2plus1DProgram()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    program.init_params(exe)

    output_dir = '/mnt/xueyang/Code/xiaodu-hi/jetson/baseline_r2plus1d'
    program.save_inference_models(exe, output_dir)
