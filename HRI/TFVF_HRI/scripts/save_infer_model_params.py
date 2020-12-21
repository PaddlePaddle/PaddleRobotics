import os
import sys
import pickle
from paddle import fluid

sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from interaction.common.utils import get_utterance_set, \
    is_attention_model, has_inst_vt_fc
from server.attn_program import AttnModelServiceProgram
from config import XiaoduHiConfig


YOLOv4_MODEL = 'tools/yolov4_paddle/inference_model'
MODEL_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
DATASET_PKL = sys.argv[3]

if len(sys.argv) > 4:
    EXTRA_ARGS = sys.argv[4:]


if ',' in MODEL_DIR:
    MODEL_DIR, SAL_MODEL_DIR = MODEL_DIR.split(',')
else:
    SAL_MODEL_DIR = None

cfg = XiaoduHiConfig()
cfg.scene_sensor_algo = 'yolov3'
if is_attention_model(MODEL_DIR):
    cfg.scene_sensor_algo = 'yolov4'

if is_attention_model(MODEL_DIR):
    inputs_type = 'visual_token'
    if has_inst_vt_fc(MODEL_DIR):
        inputs_type = EXTRA_ARGS[0]

    program = AttnModelServiceProgram(
        YOLOv4_MODEL,
        MODEL_DIR,
        inputs_type=inputs_type,
        salutation_model_dir=SAL_MODEL_DIR,
        with_salutation_cls=SAL_MODEL_DIR is not None,
        image_shape=cfg.single_img_shape[1:],
        det_confidence_th=cfg.det_confidence_th,
        roi_feat_resolution=cfg.roi_feat_resolution)
else:
    raise Exception('{} is not a TFVT-HRI model.'.format(MODEL_DIR))

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
program.init_params(exe)
program.save_inference_models(exe, OUTPUT_DIR)

if is_attention_model(MODEL_DIR):
    # TODO: add txt to record multimodal actions for act ids
    # NOTE: for attention model, DATASET_PKL is wae list.
    with open(DATASET_PKL, 'rb') as f:
        wae_lst = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, 'multimodal_actions.txt'), 'w') as f:
        for i in range(len(wae_lst)):
            f.write(wae_lst[i]['talk'] + '\n')
            f.write(wae_lst[i]['exp'] + '\n')
            f.write(wae_lst[i]['act'] + '\n')
else:
    utterance_set = dict()
    for pkl in DATASET_PKL.split(','):
        subset = get_utterance_set(pkl)
        for k in subset.keys():
            utterance_set[k] = subset[k]
    with open(os.path.join(OUTPUT_DIR, 'utterance_set.txt'), 'w') as f:
        for k, v in utterance_set.items():
            emb = [str(i) for i in list(v[1])]
            f.write('{}\n{}\n{}\n'.format(k, v[0], ','.join(emb)))
