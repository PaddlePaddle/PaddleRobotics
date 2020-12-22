import os
import pickle
import hashlib

from interaction.action import action_to_id
from interaction.expression import expression_to_id
from interaction.movement import movement_to_id


def timestamp_to_ms(time_str):
    """
    Convert timestamp string to milliseconds.

    Note that `time_str` is in format 'hour:minute:second.10-milliseconds'
    """
    hour, minute, sec_ten_ms = time_str.split(':')
    sec, ten_ms = sec_ten_ms.split('.')
    hour, minute, sec, ms = int(hour), int(minute), int(sec), int(ten_ms) * 10
    total_ms = hour * 3600 * 1000 + minute * 60 * 1000 + sec * 1000 + ms
    return total_ms


def ms_to_timestamp(ms):
    """
    Convert milliseconds to timestamp string.
    """
    hour = ms // (3600 * 1000)
    ms = ms - hour * (3600 * 1000)
    minute = ms // (60 * 1000)
    ms = ms - minute * (60 * 1000)
    sec = ms // 1000
    ms = ms - sec * 1000
    ten_ms = ms // 10
    timestamp_str = '%s:%s:%s.%s' % (int(hour), int(minute), int(sec),
                                     int(ten_ms))
    return timestamp_str


def convert_to_anno_item(item, *argv):
    anno_item = {
        'movement': item['Movement'],
        'action': item['Action'],
        'talk': item['Talk'],
        'expression': item['Expression'],
    }

    if item['Time'] != '':
        anno_item['time'] = timestamp_to_ms(item['Time'])

    if len(argv) > 0:
        offset = argv[0]
        anno_item['offset'] = offset

    return anno_item


def stable_utterance_hash(utterance, length=16):
    """
    Get stable int64 hash code for utterance.

    NOTE: the build-in `hash` is unstable, with different
    random seeds for different runs and env. So use `hashlib.sha1`.
    """
    hex_int = hashlib.sha1(utterance.encode('utf-8')).hexdigest()
    return int(hex_int, 16) % (10 ** length)


def stable_anno_hash(anno, length=8):
    anno_str = '{}-{}-{}-{}'.format(anno['Time'], anno['Talk'],
                                    anno['Action'], anno['Expression'])
    hex_int = hashlib.sha1(anno_str.encode('utf-8')).hexdigest()
    return hex_int[:length]


def is_utterance_cond_model(model_dir):
    for p in os.listdir(model_dir):
        if p.startswith('uc_'):
            return True
    return False


def is_macro_act_model(model_dir):
    for p in os.listdir(model_dir):
        if p.startswith('macro_act_'):
            return True
    return False


def is_attention_model(model_dir):
    for p in os.listdir(model_dir):
        if p.startswith('qkv_'):
            return True
    return False


def has_inst_vt_fc(model_dir):
    for p in os.listdir(model_dir):
        if p.startswith('inst_vt_fc'):
            return True
    return False


def get_macro_act_key(talk, act, exp, move):
    return '{}_{}_{}_{}'.format(
        str(stable_utterance_hash(talk)),
        str(action_to_id(act)),
        str(expression_to_id(exp)),
        str(movement_to_id(move)))


def get_macro_act_set(dataset_pkl, with_null_act=False,
                      null_talk_emb_pkl=None):
    macro_act_set = dict()
    null_talk_emb = None
    with open(dataset_pkl, 'rb') as f:
        dataset = pickle.load(f)

    if null_talk_emb_pkl is not None:
        with open(null_talk_emb_pkl, 'rb') as f:
            null_talk_emb = pickle.load(f)

    for day in dataset:
        for anno in day['annos']:
            key = get_macro_act_key(anno['talk'], anno['action'],
                                    anno['expression'], anno['movement'])

            if anno['talk'] == '':
                null_talk_emb = anno['talk_emb']

            macro_act_set[key] = {
                'talk': anno['talk'],
                'talk_emb': anno['talk_emb'],
                'act': action_to_id(anno['action']),
                'exp': expression_to_id(anno['expression']),
                'move': movement_to_id(anno['movement'])
            }

    if with_null_act:
        assert null_talk_emb is not None
        key = get_null_macro_act_key()
        macro_act_set[key] = {
            'talk': '',
            'talk_emb': null_talk_emb,
            'act': action_to_id('null'),
            'exp': expression_to_id('null'),
            'move': movement_to_id('null')
        }

    return macro_act_set


def get_null_macro_act_key():
    return get_macro_act_key('', 'null', 'null', 'null')


def get_utterance_set(dataset_pkl):
    utterance_set = dict()
    with open(dataset_pkl, 'rb') as f:
        dataset = pickle.load(f)
    for day in dataset:
        for anno in day['annos']:
            key = stable_utterance_hash(anno['talk'])
            if key not in utterance_set:
                utterance_set[key] = (anno['talk'], anno['talk_emb'])

    return utterance_set
