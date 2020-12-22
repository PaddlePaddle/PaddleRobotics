from interaction.common.discrete_ctrl import DiscreteController
from interaction.common.discrete_uc_ctrl import DiscreteUCController


ACTION_TO_ID = {
    'null': 0,
    'shake_hand': 1,
    'raise_hand': 2,
    'raise_left_hand': 3,
    'hug': 4,
    'give_me_five': 5,
    'twist_head': 6,
    'turn_head_to_left': 7,
    'turn_head_to_right': 8,
    'wave': 9,
    'altman': 10,
    'superman': 11
}

ACTION_TO_ID_V2 = {
    'null': 0,
    'shake_hand': 1,
    'hug': 2,
    'wave': 3,
    'altman': 4,
    'superman': 5
}


def action_to_id(act, version='v1'):
    assert version in ['v1', 'v2']
    if version == 'v1':
        return ACTION_TO_ID[act]
    elif version == 'v2':
        return ACTION_TO_ID_V2[act]


def id_to_action(idx, version='v1'):
    assert version in ['v1', 'v2']
    rev_dict = dict()
    if version == 'v1':
        for k, v in ACTION_TO_ID.items():
            rev_dict[v] = k
    elif version == 'v2':
        for k, v in ACTION_TO_ID_V2.items():
            rev_dict[v] = k

    return rev_dict[idx]


def action_set_size(version='v1'):
    assert version in ['v1', 'v2']
    if version == 'v1':
        return len(ACTION_TO_ID)
    elif version == 'v2':
        return len(ACTION_TO_ID_V2)


class ActionController(DiscreteController):
    def __init__(self, feat, name='action'):
        act_n = len(ACTION_TO_ID.keys())
        super(ActionController, self).__init__(name, feat, act_n)


class ActionUCController(DiscreteUCController):
    def __init__(self, feat, talk_emb, name='uc_action'):
        act_n = len(ACTION_TO_ID.keys())
        super(ActionUCController, self).__init__(name, feat, talk_emb, act_n)
