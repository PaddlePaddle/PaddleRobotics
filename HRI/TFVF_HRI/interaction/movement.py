from interaction.common.discrete_ctrl import DiscreteController
from interaction.common.discrete_uc_ctrl import DiscreteUCController


MOVEMENT_TO_ID = {
    'null': 0,
    'move_ahead': 1,
    'move_backward': 2,
    'move_left': 3,
    'move_right': 4,
    'turn_left': 5,
    'turn_right': 6
}


def movement_to_id(move):
    return MOVEMENT_TO_ID[move]


def id_to_movement(idx):
    rev_dict = dict()
    for k, v in MOVEMENT_TO_ID.items():
        rev_dict[v] = k
    return rev_dict[idx]


def movement_set_size():
    return len(MOVEMENT_TO_ID)


class MovementController(DiscreteController):
    def __init__(self, feat, name='movement'):
        move_n = len(MOVEMENT_TO_ID.keys())
        super(MovementController, self).__init__(name, feat, move_n)


class MovementUCController(DiscreteUCController):
    def __init__(self, feat, talk_emb, name='uc_movement'):
        move_n = len(MOVEMENT_TO_ID.keys())
        super(MovementUCController, self).__init__(
            name, feat, talk_emb, move_n)
