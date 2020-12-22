import os
import sys
import json
import pickle
import argparse
import numpy as np

sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from perception.utterance.eval import UtteranceEncoder
from interaction.common.utils import stable_utterance_hash
from interaction.action import action_to_id, action_set_size
from interaction.expression import expression_to_id, expression_set_size


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect original multimodal action from V2 annotations.')
    parser.add_argument(
        '--anno_dir', '-ad', type=str, default='data/annos',
        help='Directory of annotations, where contains txt files.')
    parser.add_argument(
        '--output', '-o', type=str, default='data/raw_wae',
        help='The directory to output raw multimodal action embeddings.')

    ernie_group = parser.add_argument_group('ERNIE_v1')
    ernie_group.add_argument(
        '--ernie_model_dir', type=str,
        default='pretrain_models/ERNIE_v1/params',
        help='The model directory of utterance ERNIE_v1 encoder.')
    ernie_group.add_argument(
        '--ernie_vocab', type=str,
        default='pretrain_models/ERNIE_v1/vocab.txt',
        help='The path to dictionary file for ERNIE_v1 encoder.')
    ernie_group.add_argument(
        '--ernie_model_cfg', type=str,
        default='pretrain_models/ERNIE_v1/ernie_config.json',
        help='The path to ERNIE model config file.')
    return parser.parse_args()


def action2emb(act):
    act_id = action_to_id(act, version='v2')
    emb = np.zeros(action_set_size(version='v2'), dtype=np.float32)
    emb[act_id] = 1.0
    return emb


def exp2emb(exp):
    exp_id = expression_to_id(exp, version='v2')
    emb = np.zeros(expression_set_size(version='v2'), dtype=np.float32)
    emb[exp_id] = 1.0
    return emb


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('-h')
    args = parse_args()

    encoder = UtteranceEncoder(
        args.ernie_model_dir, args.ernie_vocab,
        algorithm='ernie_v1', model_cfg_path=args.ernie_model_cfg)

    wae_dict = dict()
    for txt in os.listdir(args.anno_dir):
        hour = int(txt.split('_')[1])
        txt = os.path.join(args.anno_dir, txt)
        with open(txt, 'r') as f:
            for line in f.readlines():
                anno = json.loads(line)
                anno['Hour'] = hour

                act_emb = action2emb(anno['Action'])
                exp_emb = exp2emb(anno['Expression'])
                u_emb = list(encoder.get_encoding([anno['Talk']]))[0]

                u_hash = stable_utterance_hash(anno['Talk'])
                ma_hash = '{}_{}_{}'.format(
                    str(u_hash), anno['Action'], anno['Expression'])

                # NOTE: we ignore the movement
                wae_dict[ma_hash] = {
                    'emb': np.concatenate([u_emb, act_emb, exp_emb]),
                    'talk': anno['Talk'],
                    'act': anno['Action'],
                    'exp': anno['Expression']
                }

    null_ma_hash = '{}_{}_{}'.format(
        str(stable_utterance_hash('')), 'null', 'null')
    null_utterance_emb = list(encoder.get_encoding(['']))[0]
    null_act_emb = action2emb('null')
    null_exp_emb = exp2emb('null')
    wae_padding = np.concatenate(
        [null_utterance_emb, null_act_emb, null_exp_emb])
    null_ma_emb = {
        'id': 0,
        'emb': wae_padding,
        'talk': '',
        'act': 'null',
        'exp': 'null'
    }

    wae_lst = [null_ma_emb]
    idx = 1
    for k in wae_dict.keys():
        wae_dict[k]['id'] = idx
        wae_lst.append(wae_dict[k])
        idx += 1

    wae_dict[null_ma_hash] = null_ma_emb

    wae = [i['emb'] for i in wae_lst]
    wae = np.stack(wae)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    wae_npy = os.path.join(args.output, 'raw_wae.npy')
    np.save(wae_npy, wae)

    wae_lst_pkl = os.path.join(args.output, 'wae_lst.pkl')
    with open(wae_lst_pkl, 'wb') as f:
        pickle.dump(wae_lst, f)

    wae_dict_pkl = os.path.join(args.output, 'wae_dict.pkl')
    with open(wae_dict_pkl, 'wb') as f:
        pickle.dump(wae_dict, f)
