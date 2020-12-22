import os
from moviepy.editor import VideoFileClip, CompositeVideoClip, \
    ColorClip, TextClip, ImageClip

from interaction.common.utils import get_macro_act_key

ASSERTS_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'robot_data'))


class RobotAvatar(object):
    def __init__(self,
                 assets_path=ASSERTS_PATH,
                 cache_dir=None):
        self._read_act_assets(os.path.join(assets_path, 'action'))
        self._read_exp_assets(os.path.join(assets_path, 'expression'))
        self._read_move_assets(os.path.join(assets_path, 'movement'))
        self._create_green_exp_canvas()

        self.cache_dir = cache_dir

    def _read_act_assets(self, assets_path):
        """
        Read action mp4 files to dict of images `self.act_assets'.
        """
        self.act_assets = dict()
        for k in os.listdir(assets_path):
            assert k.endswith('.mp4')
            video_file = os.path.join(assets_path, k)
            clip = VideoFileClip(video_file)
            self.act_assets[k[:-4]] = clip

    def _read_exp_assets(self, assets_path):
        """
        Read expression gif files to dict of images `self.exp_assets'.
        """
        self.exp_assets = dict()
        for k in os.listdir(assets_path):
            assert k.endswith('.mp4')
            video_file = os.path.join(assets_path, k)
            clip = VideoFileClip(video_file, has_mask=True)
            self.exp_assets[k[:-4]] = clip.resize(width=168)

    def _read_move_assets(self, assets_path):
        self.move_assets = dict()
        for k in os.listdir(assets_path):
            assert k.endswith('.png')
            png_file = os.path.join(assets_path, k)
            clip = ImageClip(png_file)
            self.move_assets[k[:-4]] = clip

    def _create_green_exp_canvas(self, margin=0):
        """
        A green helper canvas to mark the position of robot expression.
        """
        h, w, _ = self.exp_assets['null'].get_frame(0).shape
        h += margin * 2
        w += margin * 2
        self.exp_canvas = ColorClip(size=(w, h), color=[0, 255, 0])

    def render(self, talk, act, exp, move, render_video,
               dft_exp_dt=0.2):
        if self.cache_dir is not None:
            cache_video = '{}.mp4'.format(
                get_macro_act_key(talk, act, exp, move))
            cache_video = os.path.join(self.cache_dir, cache_video)
            if os.path.exists(cache_video):
                clip = VideoFileClip(cache_video)
                clip.write_videofile(render_video)
                return

        act_clip = self.act_assets[act]
        default_exp_clip = self.exp_assets['null']
        exp_clip = self.exp_assets[exp]

        if talk == '':
            clips = [act_clip,
                     default_exp_clip.set_position(lambda t: (291, 160))
                     .set_duration(dft_exp_dt)]
        else:
            talk_clip = TextClip(
                talk, font='data/SimHei.ttf', color='green',
                method='caption', fontsize=30)
            clips = [act_clip,
                     talk_clip.set_position(('center', 50)),
                     default_exp_clip.set_position(lambda t: (291, 160))
                     .set_duration(dft_exp_dt)]

        clips.append(exp_clip.set_position(lambda t: (291, 160))
                     .set_start(dft_exp_dt))
        ts = dft_exp_dt + exp_clip.duration
        if ts < act_clip.duration:
            clips.append(default_exp_clip
                         .set_position(lambda t: (291, 160))
                         .set_duration(act_clip.duration - ts)
                         .set_start(ts))

        if move != 'null':
            move_clip = self.move_assets[move]
            clips.append(move_clip.set_position(('center', 650)))

        final_clip = CompositeVideoClip(clips).set_duration(act_clip.duration)
        final_clip.write_videofile(render_video)
