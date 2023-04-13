import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2
import threading
import time

from visual_api.handlers import SyncExecutor
from visual_api.models.detection import Detection
import visual_api.launchers as launchers
from visual_api.common import NetworkInfo, open_images_capture, PerformanceMetrics
from visual_api.common.colors import get_color

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

import pandas as pd
import numpy as np
from time import perf_counter

labels_list = pd.read_csv('labels.csv', delimiter=';', usecols=['label']).to_numpy().squeeze()



class UsingVisualAPI(object):
    def __init__(self):
        self.image = None
        self.thread_stop = False

    def set_config(self, cfg):
        self.config = cfg

    def stop_cycle(self):
        self.thread_stop = True

    def fast_start(self, path_model, path_input):

        loop = True

        cap = open_images_capture(path_input, loop)
        #cap = CapruteVideoStream(src=path_input, loop=loop).start()
        delay = int(cap.get_type() in {'VIDEO', 'CAMERA'})

        # 1 create launcher
        launcher = launchers.create_launcher_by_model_path(path_model)

        # 2 create model
        config = {
            'mean_values': None,
            'scale_values': None,
            'reverse_input_channels': False,
            'topk': 5,
            'path_to_labels': 'c:/Users/user/PycharmProjects/computer_vision_faster/datasets/coco128/coco128.yaml'
        }
        model = Detection(NetworkInfo(launcher.get_input_layers(), launcher.get_output_layers()), config)
        model.log_layers_info()

        # 3 create handler-executor
        executor = SyncExecutor(model, launcher)

        # 4 Inference part
        next_frame_id = 0
        video_writer = cv2.VideoWriter()
        ESC_KEY = 27
        key = -1

        THR_SCORE = 0.8

        metrics = PerformanceMetrics()

        while True:
            current_time = perf_counter()

            # Get new image/frame
            frame = cap.read()
            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                pass
                #args.output name file output
                #out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))
                #if args.output and not video_writer.open(args.output,
                #                                         cv2.VideoWriter_fourcc(*'MJPG'),
                #                                         cap.fps(),
                #                                         (frame.shape[1], frame.shape[0])):
                #    raise RuntimeError("Can't open video writer")

            # Inference current frame
            detections, _ = executor.run(frame)

            #if args.raw_output_message:
            #    print_raw_results(detections, next_frame_id)

            #frame = self.draw_boxes(frame, detections, labels_list, THR_SCORE)
            frame = self.draw(frame, detections, model.labels)

            metrics.update(current_time, frame)

            #if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id <= args.output_limit - 1):
            #    video_writer.write(frame)

            # Visualization
            #if not args.no_show:
            #if True:
                #это если мы будем выводить тут.
                #cv2.imshow('Detection Results', frame)
            #    key = cv2.waitKey(1)
                #key = cv2.waitKey(delay)
                # Quit.
            #    if key in {ord('q'), ord('Q'), ESC_KEY}:
            #        cap.stop()
            #        cv2.destroyAllWindows()
            #        break

            self.image = frame
            next_frame_id += 1

            if self.thread_stop:
                self.thread_stop = False
                return


    def draw(self, frame, detections, labels):
        image_draw = frame.copy()
        for (bbox, score, label) in detections:
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = labels[cls_id]
            color = (0, 255, 0)
            cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            cv2.putText(image_draw,
                        f'{cls}:{int(score * 100)}',
                        (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.80,
                        [225, 255, 255],
                        thickness=1)
        return image_draw


    def draw_boxes(self, frame, boxes, labels, obj_thresh):
        for box in boxes:

            label = int(box[1]) - 1
            if box[2] > obj_thresh and label < len(labels):
                label_str = f"{labels[label]}, {box[2]:.2f}"

                h = frame.shape[0]
                w = frame.shape[1]

                text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * frame.shape[1], 5)
                width, height = text_size[0][0], text_size[0][1]

                b_xmin, b_ymin, b_xmax, b_ymax = int(box[0][1] * w), int(box[0][0] * h), int(box[0][3] * w), int(
                    box[0][2] * h)

                region = np.array([[b_xmin, b_ymin],
                                   [b_xmin, b_ymin - height - 15],
                                   [b_xmin + width + 10, b_ymin - height - 15],
                                   [b_xmin + width + 10, b_ymin]], dtype='int32')

                cv2.rectangle(img=frame, pt1=(b_xmin, b_ymin), pt2=(b_xmax, b_ymax), color=get_color(label),
                              thickness=2)  #
                cv2.fillPoly(img=frame, pts=[region], color=get_color(label))
                cv2.putText(img=frame,
                            text=label_str,
                            org=(b_xmin + 13, b_ymin - 13),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1e-3 * frame.shape[0],
                            color=(0, 0, 0),
                            thickness=1)


class CapruteVideoStream:
    def __init__(self, src=0, loop=True):
        self.stream = open_images_capture(src, loop)
        self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True