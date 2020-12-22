import os
import sys
import cv2
import glob
import pickle
import argparse
import numpy as np
from moviepy.editor import VideoFileClip

sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from perception.common.video import VideoWriter, convert_to_h264


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect video with person tracking information '
        'for the dataset-v2 annotation.')
    parser.add_argument(
        '--video_dir', '-d', default='data/clips',
        help='Path to collected video clips.')
    parser.add_argument(
        '--encoder_model', type=str,
        default='pretrain_models/mars-small128.pb',
        help='Path to encoder model for ReID feature extraction.')
    parser.add_argument(
        '--yolov4_model', type=str,
        default='tools/yolov4_paddle/inference_model',
        help='Path to yolov4 model.')
    parser.add_argument(
        '--max_cosine_distance', type=float, default=0.3,
        help='Maximum cosine distance for Nearest Neightbor Metric in ReID')
    parser.add_argument(
        '--gpu', '-g', type=str, default='0', help='GPU card')
    parser.add_argument(
        '--workers', '-w', type=int, default=1,
        help='Number of workers to split tasks.')
    parser.add_argument(
        '--current_worker', '-c', type=int, default=1,
        help='The worker id for this running.')
    parser.add_argument(
        '--resume', type=int, default=None,
        help='The name of video file to resume the stopped worker.')

    return parser.parse_args()


def run_worker(tasks, gpu_id, encoder_model, yolov4_model,
               output_dir, max_cosine_distance, resume):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    from perception.tracker.re_id import create_box_encoder, \
        NearestNeighborDistanceMetric
    from perception.tracker.tracker import Tracker, Detection
    from perception.scene.eval import SceneSensor

    encoder = create_box_encoder(encoder_model, batch_size=8)
    metric = NearestNeighborDistanceMetric(
        'cosine', max_cosine_distance, None)
    tracker = Tracker(metric)
    detector = SceneSensor(yolov4_model,
                           gpu=0,
                           img_shape=[3, 416, 416],
                           algorithm='yolov4')

    for video_file in tasks:
        task_id = os.path.basename(video_file)[:-len('.mp4')]
        if resume is not None:
            if resume != task_id:
                continue
            else:
                resume = None

        clip = VideoFileClip(video_file)
        track_video = os.path.join(
            output_dir, '{}_track.mp4'.format(task_id))
        video_writer = VideoWriter(
            track_video, (clip.w, clip.h), clip.fps)

        tracker_logs = []
        for frame in clip.iter_frames():
            frame = frame[:, :, ::-1]
            instances = detector.get_instances(frame)[0]
            boxes = [ins['bbox'] for ins in instances]
            features = encoder(frame, boxes)

            detections = [Detection(ins, feat) for ins, feat in
                          zip(instances, features)]

            tracker.predict()
            tracker.update(detections)

            track_log = dict()
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                track_log[str(track.track_id)] = bbox

                # NOTE: https://github.com/opencv/opencv/issues/14866
                # We have to add this line
                frame = np.array(frame)
                cv2.rectangle(
                    frame, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(track.track_id),
                            (int(bbox[0]), int(bbox[1] + 23)),
                            0, 5e-3 * 100, (0, 255, 0), 2)

            det_log = []
            for det in detections:
                if str(det.cls) != 'person':
                    continue

                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
                det_log.append(bbox)

                # NOTE: https://github.com/opencv/opencv/issues/14866
                # We have to add this line
                frame = np.array(frame)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, score,
                            (int(bbox[0]), int(bbox[3])),
                            0, 5e-3 * 100, (0, 255, 0), 2)

            tracker_logs.append((track_log, det_log))
            video_writer.add_frame(frame)

        video_writer.close()
        convert_to_h264(track_video)
        print('Saved {}'.format(track_video))

        tracker_logs_file = os.path.join(
            output_dir, '{}_states.pkl'.format(task_id))
        with open(tracker_logs_file, 'wb') as f:
            pickle.dump(tracker_logs, f)
        print('Saved {}'.format(tracker_logs_file))

        tracker.reset()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('-h')
    args = parse_args()
    assert args.workers > 0 and args.current_worker <= args.workers

    tasks = glob.glob(os.path.join(args.video_dir, '*.mp4'))
    tasks = [i for i in tasks if not
             os.path.basename(i).endswith('_track.mp4')]
    tasks = sorted(tasks)

    tasks_split = [[] for _ in range(args.workers)]
    for i, task in enumerate(tasks):
        tasks_split[i % args.workers].append(task)

    run_worker(tasks_split[args.current_worker - 1], args.gpu,
               args.encoder_model, args.yolov4_model,
               args.video_dir, args.max_cosine_distance,
               args.resume)
