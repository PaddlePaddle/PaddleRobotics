import os
import cv2
import torch
import pickle
import tempfile
import numpy as np
from utils_cv.action_recognition.dataset import VideoDataset, \
    get_transforms, get_default_tfms_config

from perception.common.video import read_frames_dir, read_all_frames
from interaction.scenario import scenario_to_id, id_to_scenario, \
    scenario_classes
from interaction.common.data_v2 import check_passive_interaction, sample_frames
from interaction.common.utils import timestamp_to_ms


def get_tfms_config(for_train):
    cfg = get_default_tfms_config(for_train)
    cfg.set('random_crop', False)
    cfg.set('input_size', 224)
    cfg.set('im_scale', 224)
    return cfg


def make_boxed_img(img, bg_fill=[128, 128, 128], resize_to=416):
    aspect_ratio = min(resize_to * 1.0 / img.shape[0],
                       resize_to * 1.0 / img.shape[1])
    new_h = int(img.shape[0] * aspect_ratio)
    new_w = int(img.shape[1] * aspect_ratio)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Generate canvas with size (resize_to, resize_to)
    boxed_img = np.zeros((resize_to, resize_to, 3)).astype(np.uint8)
    boxed_img[:, :] = np.array(bg_fill).astype(np.uint8)

    # Paste resized image
    y_offset = (resize_to - new_h) // 2
    x_offset = (resize_to - new_w) // 2
    boxed_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return boxed_img


class FramesDataset(VideoDataset):
    """
    Override some member functions to support v2 dataset
    via reading frames.
    """
    def __init__(self,
                 train_dataset_pkl,
                 test_dataset_pkl,
                 train_full_neg_txt,
                 test_full_neg_txt,
                 eval_txt=None,
                 group_by='Scenario',
                 root='data/xiaodu_clips_v2',
                 neg_root='data/full_neg_data',
                 wae_lst_pkl='data/raw_wae/wae_lst.pkl',
                 seed=None,
                 sample_length=8,
                 sample_interval=100.,
                 batch_size=8,
                 check_passive=False,
                 warning=True):
        assert group_by in ['Scenario', 'WAE_id']
        self.group_by = group_by
        self.root = root
        self.neg_root = neg_root
        self.sample_interval = sample_interval
        self.check_passive = check_passive

        if group_by == 'WAE_id':
            wae_lst_len = self._get_wae_lst_len(wae_lst_pkl)
            self.classes = ['wae_{}'.format(i) for i in range(wae_lst_len)]
        elif group_by == 'Scenario':
            self.classes = scenario_classes(version='v2')

        train_split_file = self._generate_tmp_split_file(
            'train', train_dataset_pkl, train_full_neg_txt)
        if eval_txt is None:
            test_split_file = self._generate_tmp_split_file(
                'test', test_dataset_pkl, test_full_neg_txt)
        else:
            test_split_file = self._generate_eval_tmp_file(eval_txt)

        train_cfg = get_tfms_config(True)
        test_cfg = get_tfms_config(False)

        super(FramesDataset, self).__init__(
            root=root,
            seed=seed,
            num_samples=1,
            sample_length=sample_length,
            train_split_file=train_split_file,
            test_split_file=test_split_file,
            train_transforms=get_transforms(True, train_cfg),
            test_transforms=get_transforms(False, test_cfg),
            batch_size=batch_size,
            warning=warning)

        # os.remove(train_split_file)
        # os.remove(test_split_file)

    def _get_wae_lst_len(self, pkl):
        with open(pkl, 'rb') as f:
            lst = pickle.load(f)
            return len(lst)

    def _generate_tmp_split_file(self, split_type, pkl, txt):
        pos_anno_lst = []
        with open(pkl, 'rb') as f:
            for anno in pickle.load(f):
                video_id = '{}-{}'.format(anno['VideoID'], anno['Time'])
                if self.group_by == 'Scenario':
                    label = scenario_to_id(anno['Scenario'], version='v2')
                    label_name = anno['Scenario'].lower().replace(' ', '_')
                    pos_anno_lst.append([video_id, label, label_name])
                elif self.group_by == 'WAE_id':
                    label = anno['WAE_id']
                    pos_anno_lst.append([video_id, label])

        neg_anno_lst = []
        with open(txt, 'r') as f:
            for line in f.readlines():
                path = line.strip()
                video_id = os.path.basename(path)

                path = os.path.join(self.neg_root, video_id)
                assert os.path.exists(path)
                if self.check_passive and check_passive_interaction(path):
                    continue

                if self.group_by == 'Scenario':
                    label = scenario_to_id('null', version='v2')
                    label_name = 'null'
                    neg_anno_lst.append([video_id, label, label_name])
                elif self.group_by == 'WAE_id':
                    label = 0
                    neg_anno_lst.append([video_id, label])

        records = []
        if split_type == 'test':
            records.extend(neg_anno_lst)
            records.extend(pos_anno_lst)
        else:
            # TODO: rebalance different pos annos
            if len(neg_anno_lst) > len(pos_anno_lst):
                repeats = int(len(neg_anno_lst) / len(pos_anno_lst))
                records.extend(neg_anno_lst)
                for _ in range(repeats):
                    records.extend(pos_anno_lst)

                residual = len(neg_anno_lst) % len(pos_anno_lst)
                ids = np.arange(len(pos_anno_lst))
                np.random.shuffle(ids)
                for i in ids[:residual]:
                    records.append(pos_anno_lst[i])
            else:
                repeats = int(len(pos_anno_lst) / len(neg_anno_lst))
                records.extend(pos_anno_lst)
                for _ in range(repeats):
                    records.extend(neg_anno_lst)

                residual = len(pos_anno_lst) % len(neg_anno_lst)
                ids = np.arange(len(neg_anno_lst))
                np.random.shuffle(ids)
                for i in ids[:residual]:
                    records.append(neg_anno_lst[i])

        ids = np.arange(len(records))
        np.random.shuffle(ids)
        records = [records[i] for i in ids]

        fd, path = tempfile.mkstemp(suffix=split_type)
        with os.fdopen(fd, 'w') as tmp:
            for record in records:
                tmp.write(' '.join([str(e) for e in record]) + '\n')
        print('Wrote {}'.format(path))

        return path

    def _generate_eval_tmp_file(self, eval_txt):
        records = []
        with open(eval_txt, 'r') as f:
            for line in f.readlines():
                info = line.strip().split(' ')
                video_id = '{}-{}'.format(info[1], info[2])
                if self.group_by == 'Scenario':
                    label = int(info[4]) if int(info[0]) == 1 else 0
                    name = id_to_scenario(label, version='v2')
                    label_name = name.lower().replace(' ', '_')
                    records.append([video_id, label, label_name])
                elif self.group_by == 'WAE_id':
                    label = int(info[3]) if int(info[0]) == 1 else 0
                    records.append([video_id, label])

        ids = np.arange(len(records))
        np.random.shuffle(ids)
        records = [records[i] for i in ids]

        fd, path = tempfile.mkstemp(suffix='eval')
        with os.fdopen(fd, 'w') as tmp:
            for record in records:
                tmp.write(' '.join([str(e) for e in record]) + '\n')
        print('Wrote {}'.format(path))

        return path

    def __getitem__(self, idx):
        """
        Return:
            (clips (torch.tensor), label (int))
        """
        record = self.video_records[idx]
        if '-' in record.path and ',' not in record.path:
            sep = record.path.rindex('-')
            video_id = record.path[:sep]
            timestamp = record.path[sep:]
            te = timestamp_to_ms(timestamp)
            ts = te - self.sample_length * self.sample_interval

            frames_dir = os.path.join(self.root, video_id)
            frames = read_frames_dir(frames_dir, max(0.0, ts), te)
            frames = sample_frames(frames, self.sample_length)
        elif '-' in record.path and ',' in record.path:
            sep = record.path.rindex('-')
            frame_dir = record.path[:sep]
            frame_ids = record.path[sep:].split(',')[-self.sample_length:]
            frames = [cv2.imread(os.path.join(frame_dir, '{}.jpg'.format(i)))
                      for i in frame_ids]
        else:
            video_id = record.path
            frames_dir = os.path.join(self.neg_root, video_id)
            frames = read_all_frames(frames_dir)
            frames = sample_frames(frames, self.sample_length)

        frames = [make_boxed_img(i) for i in frames]
        frames = [i[:, :, ::-1] for i in frames]  # utils_cv uses RGB
        clip = np.array(frames)
        return self.transforms(torch.from_numpy(clip)), record.label
