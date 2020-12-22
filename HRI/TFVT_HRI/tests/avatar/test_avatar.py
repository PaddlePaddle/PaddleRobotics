import pytest
import os

from avatar import RobotAvatar


class TestRobotAvatar(object):
    def setup_class(self):
        self.avatar = RobotAvatar()

    def test_render(self):
        render_video = 'data/robot_data/render.mp4'
        self.avatar.render('你好，我想你啦！', 'hug', 'smile', 'move_ahead',
                           render_video)
        assert os.path.exists(render_video)
