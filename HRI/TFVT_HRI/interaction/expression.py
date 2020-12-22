from interaction.common.discrete_ctrl import DiscreteController
from interaction.common.discrete_uc_ctrl import DiscreteUCController


EXPRESSION_TO_ID = {
    'null': 0,
    'smile': 1,
    'embarrassed': 2,
    'shy': 3,
    'anthomaniac': 4,
    'nervous': 5,
    'shocked': 6,
    'cry': 7,
    'sleepy': 8,
    'blushed': 9,
    'depressed': 10,
    'thinking': 11,
    'blink': 12,
    'concentrated': 13,
    'collapse': 14,
    'despise': 15,
    'angry': 16,
    'watch': 17,
    'cool': 18,
    'desperate': 19,
    'snigger': 20,
    'sharp': 21,
    'think_of': 22,
    'proud': 23,
    'panic': 24,
    'sweat': 25,
    'fighting': 26,
    'confused': 27,
    'dizzy': 28,
    'bah': 29
}

EXPRESSION_TO_ID_V2 = {
    'null': 0,
    'shuangzhayan': 1,
    'xinxin': 2,
    'shy': 3
}


def expression_to_id(exp, version='v1'):
    assert version in ['v1', 'v2']
    if version == 'v1':
        return EXPRESSION_TO_ID[exp]
    elif version == 'v2':
        return EXPRESSION_TO_ID_V2[exp]


def id_to_expression(idx, version='v1'):
    assert version in ['v1', 'v2']
    rev_dict = dict()
    if version == 'v1':
        for k, v in EXPRESSION_TO_ID.items():
            rev_dict[v] = k
    elif version == 'v2':
        for k, v in EXPRESSION_TO_ID_V2.items():
            rev_dict[v] = k
    return rev_dict[idx]


def expression_set_size(version='v1'):
    assert version in ['v1', 'v2']
    if version == 'v1':
        return len(EXPRESSION_TO_ID)
    elif version == 'v2':
        return len(EXPRESSION_TO_ID_V2)


class ExpressionController(DiscreteController):
    def __init__(self, feat, name='expression'):
        exp_n = len(EXPRESSION_TO_ID.keys())
        super(ExpressionController, self).__init__(name, feat, exp_n)


class ExpressionUCController(DiscreteUCController):
    def __init__(self, feat, talk_emb, name='uc_expression'):
        exp_n = len(EXPRESSION_TO_ID.keys())
        super(ExpressionUCController, self).__init__(
            name, feat, talk_emb, exp_n)
