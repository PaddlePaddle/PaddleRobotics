import os
import cv2
import json
import time
import queue
import pickle
import warnings
import numpy as np
import multiprocessing as mp
from paddle import fluid

from config import XiaoduHiConfig
from perception.scene.eval import SceneSensor
from perception.common.video import clip_video_to_frames, \
    get_video_extra_info, read_frames_dir, read_all_frames, \
    read_frames_dir_with_fids
from perception.common.utils import get_bbox_pos_emb, inst_crop_preprocess
from interaction.common.utils import timestamp_to_ms, \
    ms_to_timestamp, stable_anno_hash, stable_utterance_hash
from interaction.scenario import scenario_to_id

Enable_Time_Log = True
Interested_Objs = ['person', 'backpack', 'handbag', 'suitcase',
                   'tie', 'cell phone']
Cfg = XiaoduHiConfig()


class XiaoduHiDataset(object):
    def __init__(self, video_tracking_dir, anno_dir, wae_dir):
        self.video_tracking_dir = video_tracking_dir
        self.anno_dir = anno_dir
        self.wae_dir = wae_dir

        self._collect_annotations()
        self._split_train_test_sets(test_percentage=0.2)

    def _collect_annotations(self):
        wae_dict_pkl = os.path.join(self.wae_dir, 'wae_dict.pkl')
        with open(wae_dict_pkl, 'rb') as f:
            self.wae_dict = pickle.load(f)

        self.annos = []
        for anno_file in os.listdir(self.anno_dir):
            video_id = '_'.join(anno_file.split('_')[:2])
            print(video_id)

            with open(os.path.join(self.anno_dir, anno_file), 'r') as f:
                for line in f.readlines():
                    anno = json.loads(line)
                    anno['VideoID'] = video_id

                    ma_hash = '{}_{}_{}'.format(
                        str(stable_utterance_hash(anno['Talk'])),
                        anno['Action'], anno['Expression'])
                    anno['WAE_id'] = self.wae_dict[ma_hash]['id']
                    self.annos.append(anno)

    def _split_train_test_sets(self, test_percentage=0.2):
        videos = set([anno['VideoID'] for anno in self.annos])
        num_test = int(len(videos) * test_percentage)

        ids = np.arange(len(videos))
        np.random.shuffle(ids)
        videos = [list(videos)[i] for i in ids]
        test_videos = set(videos[:num_test])

        self.test_annos, self.train_annos = [], []
        for anno in self.annos:
            if anno['VideoID'] in test_videos:
                self.test_annos.append(anno)
            else:
                self.train_annos.append(anno)

        ids = np.arange(len(self.train_annos))
        np.random.shuffle(ids)
        self.train_annos = [self.train_annos[i] for i in list(ids)]

    def build_dataset(self, output_dir):
        train_pkl = os.path.join(output_dir, 'train.pkl')
        test_pkl = os.path.join(output_dir, 'test.pkl')

        with open(train_pkl, 'wb') as f:
            pickle.dump(self.train_annos, f)

        with open(test_pkl, 'wb') as f:
            pickle.dump(self.test_annos, f)


class SalutationClsDataset(object):
    def __init__(self, video_tracking_dir, anno_dir,
                 yolov4_model_dir, roi_feat_resolution=5, gpu=0):
        self.video_tracking_dir = video_tracking_dir
        self.anno_dir = anno_dir
        self.yolov4_model_dir = yolov4_model_dir
        self.roi_feat_resolution = roi_feat_resolution
        self.gpu = gpu

        self._collect_annotations()
        self._split_train_test_sets(test_percentage=0.2)

    def _collect_annotations(self):
        self.annos = []
        for anno_file in os.listdir(self.anno_dir):
            video_id = '_'.join(anno_file.split('_')[:2])
            print(video_id)

            with open(os.path.join(self.anno_dir, anno_file), 'r') as f:
                for line in f.readlines():
                    anno = json.loads(line)
                    anno['VideoID'] = video_id
                    if anno['Salutation'] != 'null':
                        self.annos.append(anno)

    def _split_train_test_sets(self, test_percentage=0.2):
        # Copy from XiaoduHiDataloaderv2
        videos = set([anno['VideoID'] for anno in self.annos])
        num_test = int(len(videos) * test_percentage)

        ids = np.arange(len(videos))
        np.random.shuffle(ids)
        videos = [list(videos)[i] for i in ids]
        test_videos = set(videos[:num_test])

        self.test_annos, self.train_annos = [], []
        for anno in self.annos:
            if anno['VideoID'] in test_videos:
                self.test_annos.append(anno)
            else:
                self.train_annos.append(anno)

        ids = np.arange(len(self.train_annos))
        np.random.shuffle(ids)
        self.train_annos = [self.train_annos[i] for i in list(ids)]

    def _process_single_anno(self, idx, anno, txt, data_dir):
        if not hasattr(self, "scene_sensor"):
            self.scene_sensor = SceneSensor(
                self.yolov4_model_dir,
                gpu=self.gpu,
                img_shape=[3, 416, 416],
                roi_feat_resolution=self.roi_feat_resolution,
                algorithm='yolov4')

        # Read annos and data
        track_states_file = os.path.join(
            self.video_tracking_dir, '{}_states.pkl'.format(anno['VideoID']))
        with open(track_states_file, 'rb') as f:
            track_states = pickle.load(f)

        video_file = os.path.join(
            self.video_tracking_dir, '{}.mp4'.format(anno['VideoID']))
        frames = clip_video_to_frames(video_file, 0.0, None)

        # Extract frames
        related_frames, related_tracks = [], []
        for frame, (tracks, bboxes) in zip(frames, track_states):
            if anno['ID'] not in tracks:
                continue

            related_frames.append(frame)
            related_tracks.append(tracks[anno['ID']])

        instances_lst = self.scene_sensor.get_instances_with_feats(
            related_frames, get_full_fm=False)

        for frame, instances, track in zip(
                related_frames, instances_lst, related_tracks):
            _, inst_id = max_iou(track, instances, return_id=True)
            if inst_id == -1:
                warnings.warn(
                    'Cannot find corresponding instance for track in '
                    'anno: {}\n'.format(anno))
                continue

            x1, y1, x2, y2 = instances[inst_id]['bbox']
            cv2.imwrite(os.path.join(data_dir, '{}.jpg'.format(idx)),
                        frame[int(y1):int(y2), int(x1):int(x2)])
            np.save(os.path.join(data_dir, '{}.npy'.format(idx)),
                    instances[inst_id]['fm'])
            with open(txt, 'a') as f:
                if anno['Salutation'] == 'man':
                    tree_mask, cls0, cls1, cls2 = '100', 0, -1, -1
                elif anno['Salutation'] == 'woman':
                    tree_mask, cls0, cls1, cls2 = '100', 1, -1, -1
                elif anno['Salutation'] == 'young_boy':
                    tree_mask, cls0, cls1, cls2 = '110', 0, 0, -1
                elif anno['Salutation'] == 'uncle':
                    tree_mask, cls0, cls1, cls2 = '110', 0, 1, -1
                elif anno['Salutation'] == 'young_girl':
                    tree_mask, cls0, cls1, cls2 = '101', 1, -1, 0
                elif anno['Salutation'] == 'aunt':
                    tree_mask, cls0, cls1, cls2 = '101', 1, -1, 1

                f.write('{} {} {} {} {}\n'.format(
                    idx, tree_mask, cls0, cls1, cls2))

            idx += 1

        return idx

    def build_dataset(self, output_dir):
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        train_txt = os.path.join(output_dir, 'train.txt')
        test_txt = os.path.join(output_dir, 'test.txt')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for txt, data_dir, annos in zip(
                [test_txt, train_txt], [test_dir, train_dir],
                [self.test_annos, self.train_annos]):
            print('Generating {}'.format(txt))
            idx = 0
            for anno in annos:
                idx = self._process_single_anno(idx, anno, txt, data_dir)
                print(idx)


class VideoAugmentorV2(object):
    """A video intensity augmentor that may break detector.

    NOTE:
    Check the whether the augmented video breaks the detector before
    use it!
    """
    def __init__(self,
                 intensity_mul_probs=[0.2, 0.2],
                 intensity_mul_values=[1.1, 0.9],
                 ):
        # TODO: find a better intensity augmentation setting!
        from vidaug import augmentors as va
        aug = []
        for p, v in zip(intensity_mul_probs, intensity_mul_values):
            aug.append(va.Sometimes(p, va.Multiply(value=v)))
        self.video_aug = va.Sequential(aug)

    def __call__(self, frames):
        return self.video_aug(frames)


class DataWorkerV2(object):
    """Preprocess v2 data in another thread"""
    def __init__(self, proc_manager, yolov4_model_dir, video_dir, anno_lst,
                 data_queue, roi_feat_resolution=5, ob_window_len=10,
                 interval=100., tokens_per_frame=20,
                 inst_crop_shape=(128, 128), inst_fm_shape=[512, 5, 5],
                 inst_pos_dim=50, inst_cls_dim=80, visual_token_dim=562,
                 gpu=0, augment=False, use_frames_first=True, read_cache=True):
        self.yolov4_model_dir = yolov4_model_dir
        self.video_dir = video_dir
        self.anno_lst = anno_lst
        self.data_queue = data_queue
        self.conf_dict = {
            'gpu': gpu,
            'roi_feat_resolution': roi_feat_resolution,
            'ob_window_len': ob_window_len,
            'interval': interval,
            'tokens_per_frame': tokens_per_frame,
            'augment': augment,
            'inst_crop_shape': inst_crop_shape,
            'inst_fm_shape': inst_fm_shape,
            'inst_pos_dim': inst_pos_dim,
            'inst_cls_dim': inst_cls_dim,
            'visual_token_dim': visual_token_dim,
            'use_frames_first': use_frames_first,
            'read_cache': read_cache
        }

        self.msg_queue = proc_manager.Queue()

    def start(self):
        self.proc = mp.Process(
            target=DataWorkerV2.worker_func,
            args=(self.yolov4_model_dir, self.video_dir, self.anno_lst,
                  self.data_queue, self.msg_queue, self.conf_dict))
        self.proc.start()

    def stop(self):
        self.msg_queue.put('stop')

    def update(self, anno_lst):
        self.msg_queue.put(('update', anno_lst))

    def next_epoch(self):
        self.msg_queue.put('new_epoch')

    @staticmethod
    def worker_func(yolov4_model_dir, video_dir, anno_lst, data_queue,
                    msg_queue, conf_dict):
        video_aug = VideoAugmentorV2()
        scene_sensor = SceneSensor(
            yolov4_model_dir,
            gpu=conf_dict['gpu'],
            img_shape=[3, 416, 416],
            roi_feat_resolution=conf_dict['roi_feat_resolution'],
            algorithm='yolov4')

        def _process_neg_frames(anno):
            if check_passive_interaction(anno['Path']):
                # Ignore examples in which someone is interacting the robot
                return

            try:
                frames = read_all_frames(anno['Path'])
            except Exception:
                warnings.warn('OpenCV IO error. Reading {}'.format(
                    anno['Path']))
                return

            frames = sample_frames(frames, conf_dict['ob_window_len'])
            h, w, _ = frames[0].shape
            if h / w == 480 / 640:
                frames = [cv2.resize(i, (640, 480)) for i in frames]
            elif h / w == 720 / 1280:
                frames = [cv2.resize(i, (1280, 720)) for i in frames]

            instances_lst = scene_sensor.get_instances_with_feats(
                frames, get_full_fm=False)
            success, data = convert_instances_lst_to_data(
                instances_lst, conf_dict['tokens_per_frame'],
                {}, [], anno['WAE_id'],
                conf_dict['inst_crop_shape'], conf_dict['inst_fm_shape'],
                conf_dict['inst_pos_dim'], conf_dict['inst_cls_dim'],
                conf_dict['visual_token_dim'])
            if success:
                data_queue.put(data)
            else:
                warnings.warn(
                    'Failed to process annotation: {}\n'.format(anno))

        def _process_single_anno(anno):
            if anno['VideoType'] == 'neg_frames':
                _process_neg_frames(anno)
                return

            te = timestamp_to_ms(anno['Time'])
            ts = te - conf_dict['ob_window_len'] * conf_dict['interval']
            frames_dir = os.path.join(video_dir, anno['VideoID'])
            # print('=================', frames_dir, anno['VideoType'])
            if conf_dict['use_frames_first'] and os.path.isdir(frames_dir):
                # Read images
                try:
                    frames = read_frames_dir(frames_dir, max(0.0, ts), te)
                    frames = sample_frames(frames, conf_dict['ob_window_len'])
                    ctx_frames = read_frames_dir(frames_dir, 0.0, te)
                except Exception:
                    warnings.warn('OpenCV IO error. Reading {}'.format(
                        frames_dir))
                    return

                h, w, _ = frames[0].shape
                if h / w == 480 / 640:
                    frames = [cv2.resize(i, (640, 480)) for i in frames]
                elif h / w == 720 / 1280:
                    frames = [cv2.resize(i, (1280, 720)) for i in frames]
            else:
                # Read video
                video_file = os.path.join(
                    video_dir, '{}.mp4'.format(anno['VideoID']))
                frames = clip_video_to_frames(video_file, max(0.0, ts), te)
                frames = sample_frames(frames, conf_dict['ob_window_len'])
                ctx_frames = clip_video_to_frames(video_file, 0.0, te)

            track_states_file = os.path.join(
                video_dir, '{}_states.pkl'.format(anno['VideoID']))
            with open(track_states_file, 'rb') as f:
                track_states = pickle.load(f)
            last_frame_tracks = track_states[len(ctx_frames)-1][0]
            obj_ids = anno['ID'].split(',') if anno['ID'] != '' else []
            check_passed = True
            for idx in obj_ids:
                check_passed = check_passed and idx in last_frame_tracks
            if not check_passed:
                warnings.warn(
                    'Failed to process annotation: {}\n'.format(anno))
                return

            if conf_dict['augment']:
                while True:
                    aug_frames = video_aug(frames)
                    instances = scene_sensor.get_instances(aug_frames[-1:])[0]
                    iou_lst = [max_iou(last_frame_tracks[idx], instances)
                               for idx in obj_ids]
                    if len(iou_lst) == 0 or min(iou_lst) > 0.5:
                        break
                frames = aug_frames

            instances_lst = scene_sensor.get_instances_with_feats(
                frames, get_full_fm=False)
            success, data = convert_instances_lst_to_data(
                instances_lst, conf_dict['tokens_per_frame'],
                last_frame_tracks, obj_ids, anno['WAE_id'],
                conf_dict['inst_crop_shape'], conf_dict['inst_fm_shape'],
                conf_dict['inst_pos_dim'], conf_dict['inst_cls_dim'],
                conf_dict['visual_token_dim'])
            if success:
                data_queue.put(data)
            else:
                warnings.warn(
                    'Failed to process annotation: {}\n'.format(anno))

        while True:
            msg = msg_queue.get()

            if msg == 'stop':
                break
            elif msg == 'new_epoch':
                for anno in anno_lst:
                    if conf_dict['read_cache'] and 'VideoID' in anno:
                        cache_file = '{}_{}_cache.pkl'.format(
                            anno['VideoID'], stable_anno_hash(anno))
                        cache_file = os.path.join(video_dir, cache_file)
                        if os.path.exists(cache_file):
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                                data_queue.put(data)
                        else:
                            _process_single_anno(anno)
                    else:
                        _process_single_anno(anno)
            elif len(msg) == 2 and msg[0] == 'update':
                anno_lst = msg[1]


def sample_frames(frames, ob_window_len):
    ids = np.linspace(0, len(frames)-1, num=ob_window_len).astype(int)
    return [frames[i] for i in ids]


def iou(box0, box1):
    xmin_0, ymin_0, xmax_0, ymax_0 = box0
    xmin_1, ymin_1, xmax_1, ymax_1 = box1

    x_overlap = max(0, min(xmax_0, xmax_1) - max(xmin_0, xmin_1))
    y_overlap = max(0, min(ymax_0, ymax_1) - max(ymin_0, ymin_1))

    intersection = x_overlap * y_overlap
    area_0 = (xmax_0 - xmin_0) * (ymax_0 - ymin_0)
    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
    union = area_0 + area_1 - intersection
    return intersection / union


def max_iou(track, instances, return_id=False):
    iou_lst = [iou(track, i['bbox']) for i in instances]
    max_id, max_value = -1, 0.0
    for idx, v in enumerate(iou_lst):
        if v > max_value:
            max_value = v
            max_id = idx

    if return_id:
        return max_value, max_id
    else:
        return max_value


def check_passive_interaction(path, min_iou=0.8, min_size=0.1):
    # TODO: check these default values using real cases
    neg_id = os.path.basename(path)
    state_pkl = os.path.join(
        os.path.dirname(path), '{}_states.pkl'.format(neg_id))
    if not os.path.exists(state_pkl):
        return True

    try:
        im = cv2.imread(os.path.join(path, '0.jpg'))
    except Exception:
        warnings.warn('OpenCV IO error. Reading {}'.format(path))
        return True

    h, w, _ = im.shape
    size = h * w

    with open(state_pkl, 'rb') as f:
        states = pickle.load(f)

    persons = dict()
    for tracks, dets in states:
        for p in tracks.keys():
            if p not in persons:
                persons[p] = {'x_start': tracks[p], 'x_end': tracks[p]}
                persons[p]['size'] = (tracks[p][2] - tracks[p][0]) * \
                    (tracks[p][3] - tracks[p][1]) / size
            else:
                persons[p]['x_end'] = tracks[p]

    has_active_person = False
    for p in persons.keys():
        if iou(persons[p]['x_start'], persons[p]['x_end']) > min_iou and \
           persons[p]['size'] > min_size:
            has_active_person = True
            break
    return has_active_person


def filter_instances(instances, size, keeped_ids=[],
                     categories=Interested_Objs):
    # NOTE: put the `keeped_ids' at the front
    assert len(keeped_ids) < size
    keep, maybe_keep = set(keeped_ids), set()
    for i in range(len(instances)):
        if i not in keep and instances[i]['category'] in categories:
            maybe_keep.add(i)

    if len(keep) + len(maybe_keep) < size:
        sub_instances = [instances[i] for i in list(keep) + list(maybe_keep)]
        return sub_instances

    persons = [i for i in maybe_keep if instances[i]['category'] == 'person']
    persons = set(persons)

    # Keep most persons
    if len(persons) + len(keep) < size:
        mis = size - len(keep) - len(persons)
        maybe_keep = list(maybe_keep - persons)
        random_ids = np.arange(len(maybe_keep))
        np.random.shuffle(random_ids)
        rand_keep = [maybe_keep[i] for i in random_ids[:mis]]
        sub_instances = [instances[i] for i in
                         list(keep) + list(persons) + rand_keep]
    else:
        mis = size - len(keep)
        # order persons by view size
        persons = list(persons)
        persons = sorted(
            persons, key=lambda i: bbox_to_area_size(instances[i]['bbox']))
        sub_instances = [instances[i] for i in list(keep) + persons[:mis]]

    return sub_instances


def bbox_to_area_size(bbox):
    xmin, ymin, xmax, ymax = bbox
    return (ymax - ymin) * (xmax - xmin)


def convert_instances_to_data(instances, classes=80, crop_shape=(128, 128)):
    if len(instances) == 0:
        return [], [], [], [], []

    h, w, _ = Cfg.frame_shape
    c, emb_h, emb_w = instances[0]['fm'].shape

    inst_crop, inst_fm, inst_cls, inst_pos_emb, visual_tokens = \
        [], [], [], [], []
    for i in instances:
        inst_crop.append(inst_crop_preprocess(i['crop'], crop_shape))
        inst_fm.append(i['fm'])

        cls = np.zeros(classes)
        cls[i['cid']] = 1.0
        inst_cls.append(cls)

        pos_emb = get_bbox_pos_emb(i['bbox'], h, w, emb_h, emb_w)
        pos_emb = np.reshape(pos_emb, [-1])
        inst_pos_emb.append(pos_emb)

        gap = np.reshape(i['fm'], [c, -1])
        gap = np.sum(gap, axis=1) / (emb_h * emb_w)

        token = np.concatenate([pos_emb, gap])
        visual_tokens.append(token)

    return inst_crop, inst_fm, inst_cls, inst_pos_emb, visual_tokens


def match_objs_ids_to_instance_ids(obj_ids, tracks, instances):
    res = []
    for idx in obj_ids:
        _, inst_id = max_iou(tracks[idx], instances, return_id=True)
        res.append(inst_id)
    return res


def convert_instances_lst_to_data(instances_lst, tokens_per_frame,
                                  last_frame_tracks, obj_ids,
                                  wae_id, inst_crop_shape, inst_fm_shape,
                                  inst_pos_dim, inst_cls_dim,
                                  visual_token_dim):
    inst_crop, inst_fm, inst_cls, inst_pos_emb = [], [], [], []
    visual_tokens, frame_ids, padding_mask, is_obj = [], [], [], []
    act_ids, has_act = [], []

    for frame_id, instances in enumerate(instances_lst):
        take_photo_hints = [i for i in range(len(instances))
                            if instances[i]['category'] == 'cell phone']
        if frame_id == len(instances_lst) - 1:
            objs_matched_instance_ids = match_objs_ids_to_instance_ids(
                obj_ids, last_frame_tracks, instances)
            if -1 in objs_matched_instance_ids:
                # Has obj not match any instances
                return False, dict()

            sub_instances = filter_instances(
                instances, tokens_per_frame,
                keeped_ids=objs_matched_instance_ids + take_photo_hints)

            is_obj.extend([1.0 for _ in range(len(objs_matched_instance_ids))])
            is_obj.extend([0.0 for _ in range(
                tokens_per_frame - len(objs_matched_instance_ids))])

            has_act.append(1.0 if wae_id != 0 else 0.0)
            act_ids.append(wae_id)
        else:
            sub_instances = filter_instances(
                instances, tokens_per_frame, keeped_ids=take_photo_hints)
            is_obj.extend([0.0 for _ in range(tokens_per_frame)])

            has_act.append(0.0)
            act_ids.append(0)

        sub_inst_crop, sub_inst_fm, sub_inst_cls, sub_inst_pos, \
            sub_vtokens = convert_instances_to_data(
                sub_instances, crop_shape=inst_crop_shape)

        inst_crop.extend(sub_inst_crop)
        inst_fm.extend(sub_inst_fm)
        inst_cls.extend(sub_inst_cls)
        inst_pos_emb.extend(sub_inst_pos)
        visual_tokens.extend(sub_vtokens)
        padding_mask.extend([1.0 for _ in range(len(sub_instances))])
        if len(sub_instances) < tokens_per_frame:
            # Add zero pad token
            mis = tokens_per_frame - len(sub_instances)

            zero_pad = np.zeros([3] + list(inst_crop_shape)).astype(np.float32)
            inst_crop.extend([zero_pad for _ in range(mis)])

            zero_pad = np.zeros(inst_fm_shape).astype(np.float32)
            inst_fm.extend([zero_pad for _ in range(mis)])

            zero_pad = np.zeros([inst_pos_dim]).astype(np.float32)
            inst_pos_emb.extend([zero_pad for _ in range(mis)])

            zero_pad = np.zeros([inst_cls_dim]).astype(np.float32)
            inst_cls.extend([zero_pad for _ in range(mis)])

            zero_pad = np.zeros([visual_token_dim]).astype(np.float32)
            visual_tokens.extend([zero_pad for _ in range(mis)])
            padding_mask.extend([0.0 for _ in range(mis)])

        frame_ids.extend([frame_id+1 for _ in range(tokens_per_frame)])

    data = {
        'inst_crop': np.stack(inst_crop).astype(np.float32),
        'inst_fm': np.stack(inst_fm).astype(np.float32),
        'inst_cls': np.stack(inst_cls).astype(np.float32),
        'inst_pos_emb': np.stack(inst_pos_emb).astype(np.float32),
        'visual_tokens': np.stack(visual_tokens).astype(np.float32),
        'frame_ids': np.array(frame_ids).astype(np.int64),
        'padding_mask': np.array(padding_mask).astype(np.float32),
        'act_ids': np.array(act_ids).astype(np.int64),
        'has_act': np.array(has_act).astype(np.float32),
        'is_obj': np.array(is_obj).astype(np.float32)
    }
    return True, data


class XiaoduHiDataloaderv2(object):
    def __init__(self,
                 feed_list,
                 places,
                 yolov4_model_dir,
                 video_dir,
                 dataset_pkl,
                 full_neg_txt=None,
                 batch_size=8,
                 capacity=16,
                 num_workers=1,
                 worker_gpus=[0],
                 roi_feat_resolution=5,
                 ob_window_len=10,
                 interval=100.,
                 tokens_per_frame=20,
                 visual_token_dim=562,
                 augment=False,
                 resample_negs_per_epoch=True,
                 data_queue_timeout=30,
                 bad_vids=['1470_15'],
                 for_test=False):
        self.yolov4_model_dir = yolov4_model_dir
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.worker_gpus = worker_gpus
        self.roi_feat_resolution = roi_feat_resolution
        self.ob_window_len = ob_window_len
        self.interval = interval
        self.tokens_per_frame = tokens_per_frame
        self.visual_token_dim = visual_token_dim
        self.augment = augment
        self.resample_negs_per_epoch = resample_negs_per_epoch
        self.data_queue_timeout = data_queue_timeout
        self.bad_vids = bad_vids
        self.for_test = for_test

        self._load_dataset_pkl(dataset_pkl)
        if full_neg_txt is not None:
            self._load_full_neg_data(full_neg_txt)
            print('Pos : Neg = {} : {}'.format(len(self.pos_anno_lst),
                                               len(self.neg_frames_dir)))
            print('Attention: training requires at least {} epochs'.format(
                int(len(self.neg_frames_dir) / len(self.pos_anno_lst) + 1)))
        self._sample_neg_annos()

        self.loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list, capacity=capacity, iterable=True)
        self.feed_list = [i.name for i in feed_list]

        self._create_data_workers()
        self.loader.set_sample_list_generator(
            self._sample_list_generator_creator(), places=places)

    def _load_dataset_pkl(self, dataset_pkl):
        with open(dataset_pkl, 'rb') as f:
            self.pos_anno_lst = pickle.load(f)
        self.pos_anno_lst = [anno for anno in self.pos_anno_lst
                             if anno['VideoID'] not in self.bad_vids]

    def _load_full_neg_data(self, full_neg_txt):
        self.neg_frames_dir = []
        with open(full_neg_txt, 'r') as f:
            for path in f.readlines():
                path = os.path.realpath(path.strip())
                if os.path.isdir(path):
                    self.neg_frames_dir.append(path)

        np.random.seed(0)
        ids = np.arange(len(self.neg_frames_dir))
        for _ in range(10):
            np.random.shuffle(ids)
        self.neg_frames_dir = [self.neg_frames_dir[i] for i in ids]
        self.neg_frames_idx = 0

    def _sample_neg_annos(self):
        if hasattr(self, 'neg_frames_dir'):
            self._sample_neg_annos_with_full_neg()
        else:
            self._sample_neg_annos_without_full_neg()

    def _sample_neg_annos_with_full_neg(self):
        # NOTE: for training, each epoch uses same amount as pos examples
        # for testing, each epoch uses full neg examples
        neg_annos = []
        neg_count = len(self.neg_frames_dir) if self.for_test else \
            len(self.pos_anno_lst)
        for _ in range(neg_count):
            null_anno = {'WAE_id': 0, 'VideoType': 'neg_frames'}
            idx = self.neg_frames_idx % len(self.neg_frames_dir)
            null_anno['Path'] = self.neg_frames_dir[idx]
            self.neg_frames_idx += 1
            neg_annos.append(null_anno)

        ids = np.arange(len(neg_annos))
        np.random.shuffle(ids)
        self.neg_anno_lst = [neg_annos[i] for i in ids]

    def _sample_neg_annos_without_full_neg(self):
        videos = set([anno['VideoID'] for anno in self.pos_anno_lst])
        ob_window_ms = self.ob_window_len * self.interval

        neg_annos = []
        for vid in videos:
            annos = [anno for anno in self.pos_anno_lst
                     if anno['VideoID'] == vid]
            null_anno = {k: v for k, v in annos[0].items()}
            null_anno['WAE_id'] = 0
            null_anno['ID'] = ''

            annos_ms = [timestamp_to_ms(anno['Time']) for anno in annos]

            free_tseg = []
            prev_ms = 0
            for ms in annos_ms:
                if ms - prev_ms < ob_window_ms * 2:
                    continue
                free_tseg.append((prev_ms + ob_window_ms, ms - ob_window_ms))
                prev_ms = ms

            neg_ms = []
            for tseg in free_tseg:
                ms = tseg[0]
                while ms < tseg[1]:
                    ms += self.interval
                    neg_ms.append(ms)

            for ms in neg_ms:
                anno_copy = {k: v for k, v in null_anno.items()}
                anno_copy['Time'] = ms_to_timestamp(ms)
                neg_annos.append(anno_copy)

        if len(neg_annos) < len(self.pos_anno_lst) or self.for_test:
            self.neg_anno_lst = [i for i in neg_annos]
        else:
            ids = np.arange(len(neg_annos))
            if not self.resample_negs_per_epoch:
                np.random.seed(0)
            np.random.shuffle(ids)
            # neg : pos = 1 : 1
            self.neg_anno_lst = [neg_annos[i] for i in
                                 ids[:len(self.pos_anno_lst)]]

    def _create_data_workers(self):
        proc_manager = mp.Manager()
        self.data_queue = proc_manager.Queue()
        self.data_workers = []
        for wid in range(self.num_workers):
            gpu = wid % len(self.worker_gpus)
            worker = DataWorkerV2(
                proc_manager, self.yolov4_model_dir, self.video_dir,
                self._get_annos_per_worker(wid), self.data_queue,
                roi_feat_resolution=self.roi_feat_resolution,
                ob_window_len=self.ob_window_len,
                interval=self.interval,
                tokens_per_frame=self.tokens_per_frame,
                visual_token_dim=self.visual_token_dim,
                gpu=self.worker_gpus[gpu],
                augment=self.augment)
            self.data_workers.append(worker)

    def _get_annos_per_worker(self, worker_id):
        num_pos = len(self.pos_anno_lst)
        num_neg = len(self.neg_anno_lst)
        pos_anno_lst = [self.pos_anno_lst[i] for i in range(num_pos)
                        if i % self.num_workers == worker_id]
        neg_anno_lst = [self.neg_anno_lst[i] for i in range(num_neg)
                        if i % self.num_workers == worker_id]
        anno_lst = pos_anno_lst + neg_anno_lst
        ids = np.arange(len(anno_lst))
        np.random.shuffle(ids)
        anno_lst = [anno_lst[i] for i in ids]
        return anno_lst

    def _sample_list_generator_creator(self):
        def __reader__():
            if self.resample_negs_per_epoch:
                self._sample_neg_annos()
                self._worker_update_anno_lst()
            self._worker_prepare_next_epoch()
            num_annos = len(self.pos_anno_lst) + len(self.neg_anno_lst)

            idx, batch = 0, []
            while idx < num_annos:
                # NOTE: some annos may cannot be processed correctly,
                # max idx is less than num_annos, so here add timeout
                try:
                    data = self.data_queue.get(timeout=self.data_queue_timeout)
                except queue.Empty:
                    break

                data = [data[k] for k in self.feed_list]
                batch.append(data)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

                idx += 1

            if len(batch) > 0:
                yield batch

        return __reader__

    def _worker_update_anno_lst(self):
        for wid in range(self.num_workers):
            anno_lst = self._get_annos_per_worker(wid)
            self.data_workers[wid].update(anno_lst)

    def _worker_prepare_next_epoch(self):
        for worker in self.data_workers:
            worker.next_epoch()

    def start_workers(self):
        for worker in self.data_workers:
            worker.start()

    def stop_workers(self):
        for worker in self.data_workers:
            worker.stop()

    def save_to_txt(self, txt, dt=0):
        assert self.for_test and hasattr(self, 'neg_frames_dir')

        def _cvt_obj_bbox(tracks, obj_ids):
            bbox_lst = []
            for idx in obj_ids:
                bbox = ['{:.2f}'.format(i) for i in list(tracks[idx])]
                bbox_lst.append(','.join(bbox))

            return ' '.join(bbox_lst)

        anno_lst = []
        anno_lst.extend(self.neg_anno_lst)
        anno_lst.extend(self.pos_anno_lst)

        ids = np.arange(len(anno_lst))
        np.random.shuffle(ids)
        anno_lst = [anno_lst[i] for i in ids]

        with open(txt, 'w') as f:
            for anno in anno_lst:
                if anno['VideoType'] == 'neg_frames':
                    frames_ids = read_all_frames(anno['Path'], return_id=True)
                    frames_ids = sample_frames(frames_ids, self.ob_window_len)
                    # e.g. 0 path_a/path_b 0,2,4,8
                    f.write('0 {} {}\n'.format(
                        os.path.realpath(anno['Path']), ','.join(frames_ids)))
                elif anno['VideoType'] == 'r2':
                    te = timestamp_to_ms(anno['Time']) + dt
                    ts = te - self.ob_window_len * self.interval
                    frames_dir = os.path.join(self.video_dir, anno['VideoID'])
                    frames_ids = read_frames_dir(frames_dir, max(0.0, ts), te,
                                                 return_id=True)
                    frames_ids = sample_frames(frames_ids, self.ob_window_len)
                    ctx_frames_ids = read_frames_dir(frames_dir, 0.0, te,
                                                     return_id=True)

                    track_states_file = os.path.join(
                        self.video_dir,
                        '{}_states.pkl'.format(anno['VideoID']))
                    with open(track_states_file, 'rb') as fs:
                        track_states = pickle.load(fs)
                    last_frame_tracks = track_states[len(ctx_frames_ids)-1][0]
                    obj_ids = anno['ID'].split(',') if anno['ID'] != '' else []
                    check_passed = True
                    for idx in obj_ids:
                        check_passed = check_passed and idx in last_frame_tracks
                    if not check_passed:
                        warnings.warn(
                            'Failed to process annotation: {}\n'.format(anno))
                        continue

                    # e.g. 1 path_a/path_b 0,2,4 15 5,5,233,233 9,9,333,444
                    sid = scenario_to_id(anno['Scenario'], version='v2')
                    f.write('1 {} {} {} {}'.format(
                        os.path.realpath(frames_dir), ','.join(frames_ids),
                        anno['WAE_id'], sid))

                    if anno['ID'] != '':
                        f.write(' {}\n'.format(
                            _cvt_obj_bbox(last_frame_tracks, obj_ids)))
                    else:
                        f.write('\n')

    def __call__(self):
        return self.loader()


class CV2Reader(object):
    """A cv2 reader compatible with DecordReader,
    so that it can worker with other workers in data_via_decord.py.
    """
    def __init__(self,
                 pos_video_dir,
                 reader_id,
                 proc_manager,
                 data_queue,
                 anno_lst,
                 ob_window_len=10,
                 null_wae_id=0,
                 interval=250,
                 read_cache=True,
                 use_frames_first=True):
        self.reader_id = reader_id
        self.data_queue = data_queue
        self.msg_queue = proc_manager.Queue()
        self.anno_lst = anno_lst
        self.conf_dict = {
            'pos_video_dir': pos_video_dir,
            'ob_window_len': ob_window_len,
            'null_wae_id': null_wae_id,
            'interval': interval,
            'read_cache': read_cache,
            'use_frames_first': use_frames_first
        }

    def start(self):
        self.proc = mp.Process(
            target=CV2Reader.worker_func,
            args=(self.reader_id, self.data_queue,
                  self.msg_queue, self.anno_lst, self.conf_dict))
        self.proc.start()

    def stop(self):
        self.msg_queue.put('stop')

    def update(self, anno_lst):
        self.msg_queue.put(('update', anno_lst))

    def next_epoch(self):
        self.msg_queue.put('new_epoch')

    @staticmethod
    def worker_func(idx, data_queue, msg_queue, anno_lst, conf_dict):
        def _process_neg_frames(anno):
            if Enable_Time_Log:
                t1 = time.time()

            if check_passive_interaction(anno['Path']):
                return

            frame_ids = read_all_frames(anno['Path'], return_id=True)
            frame_ids = sample_frames(frame_ids, conf_dict['ob_window_len'])
            frames = read_frames_dir_with_fids(anno['Path'], frame_ids)
            h, w, _ = frames[0].shape
            if h / w == 480 / 640:
                frames = [cv2.resize(i, (640, 480)) for i in frames]
            elif h / w == 720 / 1280:
                frames = [cv2.resize(i, (640, 360)) for i in frames]

            anno_copy = {k: v for k, v in anno.items()}
            anno_copy['Frames'] = [pickle.dumps(img) for img in frames]
            anno_copy['WAE_id'] = conf_dict['null_wae_id']
            data_queue.put(anno_copy)
            if Enable_Time_Log:
                t2 = time.time()
                print('CV2 reader takes {:.3f}s'.format(t2 - t1))

        def _process_single_anno(anno):
            if anno['VideoType'] == 'neg_frames':
                _process_neg_frames(anno)
                return

            if Enable_Time_Log:
                t1 = time.time()

            te = timestamp_to_ms(anno['Time'])
            ts = te - conf_dict['ob_window_len'] * conf_dict['interval']
            frames_dir = os.path.join(
                conf_dict['pos_video_dir'], anno['VideoID'])

            if conf_dict['use_frames_first'] and os.path.isdir(frames_dir):
                frame_ids = read_frames_dir(
                    frames_dir, max(0.0, ts), te, return_id=True)
                frame_ids = sample_frames(frame_ids, conf_dict['ob_window_len'])
                frames = read_frames_dir_with_fids(frames_dir, frame_ids)
                ctx_frames = read_frames_dir(frames_dir, 0.0, te, return_id=True)
            else:
                video_file = os.path.join(
                    conf_dict['pos_video_dir'],
                    '{}.mp4'.format(anno['VideoID']))
                frames = clip_video_to_frames(video_file, max(0.0, ts), te)
                frames = sample_frames(frames, conf_dict['ob_window_len'])
                ctx_frames = clip_video_to_frames(video_file, 0.0, te)

            anno_copy = {k: v for k, v in anno.items()}
            anno_copy['Frames'] = [pickle.dumps(img) for img in frames]
            anno_copy['FrameIDs'] = [len(ctx_frames) - 1]
            data_queue.put(anno_copy)

            if Enable_Time_Log:
                t2 = time.time()
                print('CV2 reader takes {:.3f}s'.format(t2 - t1))

        while True:
            msg = msg_queue.get()

            if msg == 'stop':
                break

            elif msg == 'new_epoch':
                for anno in anno_lst:
                    anno_copy = {k: v for k, v in anno.items()}
                    if conf_dict['read_cache'] and 'VideoID' in anno:
                        cache_file = '{}_{}_cache.pkl'.format(
                            anno['VideoID'], stable_anno_hash(anno))
                        cache_file = os.path.join(
                            conf_dict['pos_video_dir'], cache_file)
                        if os.path.exists(cache_file):
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                                anno_copy['Cache'] = pickle.dumps(data)
                                data_queue.put(anno_copy)
                        else:
                            _process_single_anno(anno_copy)
                    else:
                        _process_single_anno(anno_copy)

            elif len(msg) == 2 and msg[0] == 'update':
                anno_lst = msg[1]


class SalutationClsDataloader(object):
    def __init__(self,
                 feed_list,
                 places,
                 dataset_dir,
                 dataset_txt,
                 batch_size=8,
                 capacity=16,
                 rebalance=True):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.rebalance = rebalance

        self._load_dataset_txt(dataset_txt)
        self.loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list, capacity=capacity, iterable=True)
        self.feed_list = [i.name for i in feed_list]

        self.loader.set_sample_list_generator(
            self._sample_list_generator_creator(), places=places)

    def _load_dataset_txt(self, dataset_txt):
        self.annos = []
        self.man, self.woman = [], []
        with open(dataset_txt, 'r') as f:
            for line in f.readlines():
                idx, tree_mask, cls0, cls1, cls2 = line.strip().split(' ')
                tree_mask = self._convert_tree_mask(tree_mask)
                root_cls, left_cls, right_cls = int(cls0), int(cls1), int(cls2)
                # NOTE: this modification makes it suitable for one-hot op,
                # its contribution is masked out using `tree_mask`
                left_cls = 0 if left_cls == -1 else left_cls
                right_cls = 0 if right_cls == -1 else right_cls

                if root_cls == 0:
                    self.man.append(
                        [idx, tree_mask, root_cls, left_cls, right_cls])
                elif root_cls == 1:
                    self.woman.append(
                        [idx, tree_mask, root_cls, left_cls, right_cls])

        if self.rebalance:
            mis = len(self.man) - len(self.woman)
            if mis > 0:
                ids = np.random.randint(0, len(self.woman), mis)
                self.annos.extend([self.woman[i] for i in ids])
            else:
                ids = np.random.randint(0, len(self.man), -mis)
                self.annos.extend([self.man[i] for i in ids])

        self.annos.extend(self.man)
        self.annos.extend(self.woman)

        ids = np.arange(len(self.annos))
        np.random.shuffle(ids)
        self.annos = [self.annos[i] for i in ids]

    def _convert_tree_mask(self, tree_mask):
        mask = np.zeros(len(tree_mask), dtype=np.float32)
        for i in range(len(tree_mask)):
            if tree_mask[i] == '1':
                mask[i] = 1.0
        return mask

    def _sample_list_generator_creator(self):
        def __reader__():
            feed_mapping = SalutationClsDataloader.create_feed_mapping()
            ids = np.arange(len(self.annos))
            np.random.shuffle(ids)
            self.annos = [self.annos[i] for i in ids]

            for idx in range(0, len(self.annos), self.batch_size):
                batch_annos = self.annos[idx:idx+self.batch_size]
                yield_data = []
                for aid, mask, root_cls, left_cls, right_cls in batch_annos:
                    fm_npy = os.path.join(
                        self.dataset_dir, '{}.npy'.format(aid))
                    feed_mapping['fm'] = np.load(fm_npy)
                    feed_mapping['tree_mask'] = mask
                    feed_mapping['root_cls'] = root_cls
                    feed_mapping['left_cls'] = left_cls
                    feed_mapping['right_cls'] = right_cls

                    yield_data.append(
                        [feed_mapping[k] for k in self.feed_list])

                yield yield_data

        return __reader__

    def __call__(self):
        return self.loader()

    @staticmethod
    def create_feed_mapping():
        return {
            'fm': None,
            'tree_mask': None,
            'root_cls': None,
            'left_cls': None,
            'right_cls': None
        }
