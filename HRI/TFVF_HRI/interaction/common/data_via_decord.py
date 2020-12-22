import os
import cv2
import time
import json
import pickle
import warnings
import threading
import numpy as np
from queue import Queue, Empty
import multiprocessing as mp
from functools import lru_cache
from decord import VideoReader, cpu
from paddle import fluid

from config import XiaoduHiConfig
from perception.scene.eval import SceneSensor
from perception.common.video import get_video_extra_info, get_video_length
from interaction.common.utils import timestamp_to_ms
from interaction.common.data import convert_instances_lst_to_data

Cfg = XiaoduHiConfig()
Enable_Time_Log = True


@lru_cache(maxsize=128)
def get_frame_ts_table(video):
    vr = VideoReader(video, ctx=cpu(0))
    return [int(vr.get_frame_timestamp(i)[1] * 1000) for i in range(len(vr))]


class XiaoduHiDecord(object):
    """Re-organize v2 dataset to use faster decord reading."""
    def __init__(self, video_tracking_dir, train_pkl, test_pkl,
                 anno_dir, dt=200):
        self.video_tracking_dir = video_tracking_dir
        self.train = self._load_pkl(train_pkl)
        self.test = self._load_pkl(test_pkl)
        self.anno_dir = anno_dir
        self.dt = dt

    def _load_pkl(self, pkl):
        with open(pkl, 'rb') as f:
            return pickle.load(f)

    def _collect_neg_annos(self):
        neg_annos = []
        for anno_file in os.listdir(self.anno_dir):
            video_id = '_'.join(anno_file.split('_')[:2])
            with open(os.path.join(self.anno_dir, anno_file), 'r') as f:
                annos = []
                for line in f.readlines():
                    anno = json.loads(line)
                    anno['Video'] = os.path.join(
                        self.video_tracking_dir, '{}.mp4'.format(video_id))
                    anno['Time'] = timestamp_to_ms(anno['Time'])
                    annos.append(anno)

                if len(annos) == 0:
                    total_ms, _ = get_video_length(os.path.join(
                        self.video_tracking_dir, '{}.mp4'.format(video_id)))
                    anno = dict()
                    anno['Time'] = total_ms
                    anno['Video'] = os.path.join(
                        self.video_tracking_dir, '{}.mp4'.format(video_id))
                    neg_annos.append([anno])
                else:
                    annos = sorted(annos, key=lambda a: a['Time'])
                    neg_annos.append(annos)

        return neg_annos

    def _process_pos_data(self, origin_data, workers):
        pos_data = []
        data_queue = Queue(1000)
        for idx in range(workers):
            annos = [origin_data[i] for i in range(len(origin_data))
                     if i % workers == idx]
            w = threading.Thread(
                target=XiaoduHiDecord.worker_ts2fids,
                args=(data_queue, self.video_tracking_dir, annos, self.dt))
            w.setDaemon = True
            w.start()

        collected = 0
        while collected < len(origin_data):
            new_anno = data_queue.get()
            pos_data.append(new_anno)
            collected += 1

            if collected % 1000 == 0:
                print(collected)

        print(collected)
        return pos_data

    def _process_neg_data(self, origin_data, workers):
        neg_data, worker_threads = [], []
        data_queue = Queue(1000)
        for idx in range(workers):
            splits = [origin_data[i] for i in range(len(origin_data))
                      if i % workers == idx]
            w = threading.Thread(
                target=XiaoduHiDecord.worker_negfids,
                args=(data_queue, splits))
            w.setDaemon = True
            w.start()
            worker_threads.append(w)

        collected = 0
        is_alive = [w.is_alive() for w in worker_threads]
        while True in is_alive:
            new_anno = data_queue.get()
            neg_data.append(new_anno)
            collected += 1

            if collected % 1000 == 0:
                print(collected)

            is_alive = [w.is_alive() for w in worker_threads]

        print(collected)
        return neg_data

    def build_dataset(self, output_dir, workers=8):
        print('Collecting pos_train...')
        pos_train = self._process_pos_data(self.train, workers)
        print('Collecting pos_test...')
        pos_test = self._process_pos_data(self.test, workers)
        print('Collecting neg...')
        # TODO: update
        neg_annos = self._collect_neg_annos()
        neg = self._process_neg_data(neg_annos, workers)

        dataset = {
            'pos_train': pos_train,
            'pos_test': pos_test,
            'neg': neg
        }
        with open(os.path.join(output_dir, 'decord.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

            print('Pos_train: ', len(pos_train))
            print('Pos test: ', len(pos_test))
            print('Neg: ', len(neg))

    @staticmethod
    def worker_ts2fids(data_queue, video_dir, annos, dt, nframes=10,
                       interval=250):
        # Reader worker that convert timestamp to frame IDs
        assert nframes > 1
        for anno in annos:
            video = os.path.join(video_dir, '{}.mp4'.format(anno['VideoID']))
            frame_ts_table = get_frame_ts_table(video)
            frame_ids = []
            end_ts = timestamp_to_ms(anno['Time']) + dt
            start_ts = max(0, end_ts - (nframes - 1) * interval)

            for fid, ms in enumerate(frame_ts_table):
                if ms < start_ts:
                    continue
                if ms > end_ts:
                    break

                if len(frame_ids) >= nframes:
                    frame_ids.pop(0)
                frame_ids.append(fid)

                start_ts += interval

            anno['Video'] = video
            if len(frame_ids) < nframes:
                last_fid, cur_nframes = frame_ids[-1], len(frame_ids)
                for _ in range(nframes - cur_nframes):
                    frame_ids.append(last_fid)
            anno['FrameIDs'] = frame_ids
            data_queue.put(anno)

    @staticmethod
    def worker_negfids(data_queue, splits, nframes=10, interval=250,
                       safe_gap=2000, null_wae_id=0):
        assert nframes > 1
        for annos in splits:
            video = annos[0]['Video']
            frame_ts_table = get_frame_ts_table(video)

            frame_ids = []
            fid, start_ts, end_ts = 0, 0, -1
            for anno in annos:
                end_ts = anno['Time'] - safe_gap
                if end_ts - start_ts > nframes * interval:
                    while fid < len(frame_ts_table):
                        ms = frame_ts_table[fid]
                        fid += 1
                        if ms < start_ts:
                            continue
                        if ms > end_ts:
                            break

                        # start_ts < ms < end_ts
                        if len(frame_ids) > nframes:
                            frame_ids.pop(0)
                        frame_ids.append(fid)

                        if len(frame_ids) == nframes:
                            v2_anno = {
                                'FrameIDs': frame_ids,
                                'Video': video,
                                'WAE_id': null_wae_id
                            }
                            data_queue.put(v2_anno)

                            # Comment follow line to use sliding negs
                            frame_ids = []

                        start_ts += interval

                start_ts = anno['Time'] + safe_gap


class DecordReader(object):
    def __init__(self,
                 reader_id,
                 proc_manager,
                 data_queue,
                 anno_lst):
        self.reader_id = reader_id
        self.data_queue = data_queue
        self.msg_queue = proc_manager.Queue()
        self.anno_lst = anno_lst

    def start(self):
        self.proc = mp.Process(
            target=DecordReader.worker_func,
            args=(self.reader_id, self.data_queue,
                  self.msg_queue, self.anno_lst))
        self.proc.start()

    def stop(self):
        self.msg_queue.put('stop')

    def update(self, anno_lst):
        self.msg_queue.put(('update', anno_lst))

    def next_epoch(self):
        self.msg_queue.put('new_epoch')

    @staticmethod
    def worker_func(idx, data_queue, msg_queue, anno_lst):
        while True:
            msg = msg_queue.get()

            if msg == 'stop':
                break

            elif msg == 'new_epoch':
                for anno in anno_lst:
                    if Enable_Time_Log:
                        t1 = time.time()
                    anno_copy = {k: v for k, v in anno.items()}
                    vr = VideoReader(anno['Video'], ctx=cpu(idx))
                    h, w, _ = Cfg.input_frame_shape

                    anno_copy['Frames'] = [
                        pickle.dumps(cv2.resize(img[:, :, ::-1], (w, h))) \
                        for img in \
                        list(vr.get_batch(anno['FrameIDs']).asnumpy())]
                    data_queue.put(anno_copy)
                    if Enable_Time_Log:
                        t2 = time.time()
                        print('Decord reader takes {:.3f}s'.format(t2 - t1))

            elif len(msg) == 2 and msg[0] == 'update':
                anno_lst = msg[1]


class Detector(object):
    def __init__(self,
                 gpu,
                 yolov4_model_dir,
                 proc_manager,
                 in_queue,
                 out_queue,
                 roi_feat_resolution=5):
        self.conf_dict = {
            'gpu': gpu,
            'yolov4_model_dir': yolov4_model_dir,
            'roi_feat_resolution': roi_feat_resolution
        }
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.msg_queue = proc_manager.Queue()

    def start(self):
        self.proc = mp.Process(
            target=Detector.worker_func,
            args=(self.in_queue, self.out_queue,
                  self.msg_queue, self.conf_dict))
        self.proc.start()

    def stop(self):
        self.msg_queue.put('stop')

    @staticmethod
    def worker_func(in_queue, out_queue, msg_queue, conf_dict):
        scene_sensor = SceneSensor(
            conf_dict['yolov4_model_dir'],
            gpu=conf_dict['gpu'],
            img_shape=[3, 416, 416],
            roi_feat_resolution=conf_dict['roi_feat_resolution'],
            algorithm='yolov4')

        while True:
            try:
                msg = msg_queue.get_nowait()
            except Empty:
                msg = ''

            if msg == 'stop':
                break

            try:
                anno = in_queue.get(timeout=5)
            except Empty:
                anno = None

            if anno is not None:
                if Enable_Time_Log:
                    t1 = time.time()

                if 'Cache' not in anno:
                    frames = [pickle.loads(i) for i in anno['Frames']]
                    anno['Instances'] = scene_sensor.get_instances_with_feats(
                        frames, get_full_fm=False)
                    del anno['Frames']  # to save memory!

                out_queue.put(anno)
                if Enable_Time_Log:
                    t2 = time.time()
                    print('Detector takes {:.3f}s'.format(t2 - t1))


class PostWorker(object):
    def __init__(self,
                 proc_manager,
                 in_queue,
                 out_queue,
                 tokens_per_frame=20,
                 inst_crop_shape=(128, 128),
                 inst_fm_shape=[512, 5, 5],
                 inst_pos_dim=50,
                 inst_cls_dim=80,
                 visual_token_dim=562,
                 null_wae_id=0):
        self.conf_dict = {
            'tokens_per_frame': tokens_per_frame,
            'inst_crop_shape': inst_crop_shape,
            'inst_fm_shape': inst_fm_shape,
            'inst_pos_dim': inst_pos_dim,
            'inst_cls_dim': inst_cls_dim,
            'visual_token_dim': visual_token_dim,
            'null_wae_id': null_wae_id
        }
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.msg_queue = proc_manager.Queue()

    def start(self):
        self.proc = mp.Process(
            target=PostWorker.worker_func,
            args=(self.in_queue, self.out_queue,
                  self.msg_queue, self.conf_dict))
        self.proc.start()

    def stop(self):
        self.msg_queue.put('stop')

    @staticmethod
    def worker_func(in_queue, out_queue, msg_queue, conf_dict):
        while True:
            try:
                msg = msg_queue.get_nowait()
            except Empty:
                msg = ''

            if msg == 'stop':
                break

            try:
                anno = in_queue.get(timeout=5)
            except Empty:
                anno = None

            if anno is not None:
                if Enable_Time_Log:
                    t1 = time.time()

                if 'Cache' in anno:
                    out_queue.put(pickle.loads(anno['Cache']))
                    continue

                if anno['WAE_id'] == conf_dict['null_wae_id']:
                    last_frame_tracks, obj_ids = {}, []
                else:
                    video_dir = os.path.dirname(anno['Video'])
                    track_states_file = os.path.join(
                        video_dir, '{}_states.pkl'.format(anno['VideoID']))
                    with open(track_states_file, 'rb') as f:
                        track_states = pickle.load(f)
                    last_frame_tracks = track_states[anno['FrameIDs'][-1]][0]
                    obj_ids = anno['ID'].split(',') if anno['ID'] != '' \
                        else []

                check_passed = True
                for idx in obj_ids:
                    check_passed = check_passed and idx in last_frame_tracks
                if not check_passed:
                    anno_slim = {k: anno[k] for k in
                                 ['Time', 'ID', 'Video', 'WAE_id']}
                    warnings.warn(
                        'Failed to process annotation: {}\n'.format(anno_slim))
                    continue

                success, data = convert_instances_lst_to_data(
                    anno['Instances'], conf_dict['tokens_per_frame'],
                    last_frame_tracks, obj_ids, anno['WAE_id'],
                    conf_dict['inst_crop_shape'],
                    conf_dict['inst_fm_shape'], conf_dict['inst_pos_dim'],
                    conf_dict['inst_cls_dim'], conf_dict['visual_token_dim'])
                if success:
                    out_queue.put(data)
                else:
                    anno_slim = {k: anno[k] for k in
                                 ['Time', 'ID', 'Video', 'WAE_id']}
                    warnings.warn(
                        'Failed to process annotation: {}\n'.format(anno_slim))

                if Enable_Time_Log:
                    t2 = time.time()
                    print('Postprocessing takes {:.3f}s'.format(t2 - t1))


class XiaoduHiDecordLoader(object):
    def __init__(self,
                 feed_list,
                 places,
                 yolov4_model_dir,
                 dataset_pkl,
                 batch_size=8,
                 capacity=16,
                 decord_readers=8,
                 yolov4_detectors=1,
                 post_workers=8,
                 detector_gpus=[0],
                 roi_feat_resolution=5,
                 tokens_per_frame=20,
                 visual_token_dim=562,
                 dataloader_timeout=30,
                 for_test=False):
        self.yolov4_model_dir = yolov4_model_dir
        self.batch_size = batch_size
        self.detector_gpus = detector_gpus
        self.roi_feat_resolution = roi_feat_resolution
        self.tokens_per_frame = tokens_per_frame
        self.visual_token_dim = visual_token_dim
        self.dataloader_timeout = dataloader_timeout
        self.for_test = for_test

        self._load_dataset_pkl(dataset_pkl)

        self.loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list, capacity=capacity, iterable=True)
        self.feed_list = [i.name for i in feed_list]

        self._create_workers(decord_readers, yolov4_detectors, post_workers)
        self.loader.set_sample_list_generator(
            self._sample_list_generator_creator(), places=places)

    def _load_dataset_pkl(self, dataset_pkl):
        with open(dataset_pkl, 'rb') as f:
            self.data = pickle.load(f)

        np.random.seed(0)
        ids = np.arange(len(self.data['neg']))
        np.random.shuffle(ids)

        num_test = len(self.data['pos_test'])
        num_train = len(self.data['pos_train'])
        n = int(ids.shape[0] * num_test / (num_test + num_train + 0.0))
        self.neg_test_ids = [i for i in list(ids[:n])]
        self.neg_train_ids = [i for i in list(ids[n:])]

        ids = np.arange(len(self.neg_train_ids))
        np.random.shuffle(ids)
        self.neg_train_ids = [self.neg_train_ids[i] for i in list(ids)]
        self.neg_train_offset = 0

    def _get_annos_per_reader(self, reader_id, readers):
        if self.for_test:
            num_pos = len(self.data['pos_test'])
            num_neg = len(self.neg_test_ids)
            pos_anno_lst = [self.data['pos_test'][i] for i in range(num_pos)
                            if i % readers == reader_id]
            neg_anno_lst = [self.data['neg'][self.neg_test_ids[i]]
                            for i in range(num_neg)
                            if i % readers == reader_id]
        else:
            num_pos = len(self.data['pos_train'])
            num_neg = num_pos  # rebalance pos and neg for training
            n = len(self.neg_train_ids)
            o = self.neg_train_offset
            pos_anno_lst = [self.data['pos_train'][i] for i in range(num_pos)
                            if i % readers == reader_id]
            neg_anno_lst = [self.data['neg'][self.neg_train_ids[(i + o) % n]]
                            for i in range(num_neg)
                            if i % readers == reader_id]

        anno_lst = pos_anno_lst + neg_anno_lst
        ids = np.arange(len(anno_lst))
        np.random.shuffle(ids)
        anno_lst = [anno_lst[i] for i in ids]
        return anno_lst

    def _create_workers(self, readers, detectors, post_workers,
                        queue_max_size=100):
        proc_manager = mp.Manager()
        read_frame_queue = proc_manager.Queue(queue_max_size)
        process_inst_queue = proc_manager.Queue(queue_max_size)
        self.dataloader_queue = proc_manager.Queue(queue_max_size)

        self.reader_lst, self.detector_lst, self.pw_lst = [], [], []
        for idx in range(readers):
            self.reader_lst.append(
                DecordReader(
                    idx, proc_manager, read_frame_queue, []))

        for idx in range(detectors):
            gpu = self.detector_gpus[idx % len(self.detector_gpus)]
            self.detector_lst.append(
                Detector(gpu, self.yolov4_model_dir, proc_manager,
                         read_frame_queue, process_inst_queue))

        for idx in range(post_workers):
            self.pw_lst.append(
                PostWorker(proc_manager, process_inst_queue,
                           self.dataloader_queue))

    def _sample_list_generator_creator(self):
        def __reader__():
            self._update_anno_lst()
            self._prepare_next_epoch()

            batch = []
            while True:
                try:
                    data = self.dataloader_queue.get(
                        timeout=self.dataloader_timeout)
                except Empty:
                    break

                data = [data[k] for k in self.feed_list]
                batch.append(data)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            if len(batch) > 0:
                yield batch

        return __reader__

    def _update_anno_lst(self):
        readers = len(self.reader_lst)
        for idx, reader in enumerate(self.reader_lst):
            reader.update(self._get_annos_per_reader(idx, readers))

        if not self.for_test:
            self.neg_train_offset += len(self.data['pos_train'])

    def _prepare_next_epoch(self):
        for reader in self.reader_lst:
            reader.next_epoch()

    def start_workers(self):
        for detector in self.detector_lst:
            detector.start()
            time.sleep(5)  # wait for model loading

        for post_worker in self.pw_lst:
            post_worker.start()

        for reader in self.reader_lst:
            reader.start()

    def stop_workers(self):
        for reader in self.reader_lst:
            reader.stop()

        for detector in self.detector_lst:
            detector.stop()

        for post_worker in self.pw_lst:
            post_worker.stop()

    def __call__(self):
        return self.loader()
