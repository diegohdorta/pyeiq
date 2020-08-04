# Copyright 2018 The TensorFlow Authors
# Copyright 2020 NXP Semiconductors
#
# This file was copied from TensorFlow respecting its rights. All the modified
# parts below are according to TensorFlow's LICENSE terms.
#
# SPDX-License-Identifier:    Apache-2.0

import collections
import colorsys
import os
import random
import sys
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.opencv.inference import OpenCVDNN
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import OBJ_DETECTION, OBJ_DETECTION_DNN, OBJ_DETECTION_GST
from eiq.modules.utils import DemoBase
from eiq.multimedia import gstreamer

try:
    import svgwrite
    has_svgwrite = True
except ImportError:
    has_svgwrite = False


class eIQObjectDetection(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_DETECTION)

        self.colors = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * Single Shot Detection as default algorithm.\n")

    def usage(self, name=None, labels=True, model=True):
        super().usage(name=name, labels=labels, model=model)

    def generate_colors(self):
        hsv_tuples = [(x / len(self.labels), 1., 1.)
                      for x in range(len(self.labels))]

        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255),
                                     int(x[2] * 255)), colors))
        random.seed(10101)
        random.shuffle(colors)
        random.seed(None)

        self.colors = colors

    def process_image(self, image):
        self.interpreter.set_tensor(np.expand_dims(image, axis=0))
        self.interpreter.run_inference()

        positions = self.interpreter.get_tensor(0, squeeze=True)
        classes = self.interpreter.get_tensor(1, squeeze=True)
        scores = self.interpreter.get_tensor(2, squeeze=True)

        result = []
        for idx, score in enumerate(scores):
            if score > 0.5:
                result.append({'pos': positions[idx], '_id': classes[idx]})
        return result

    def detect_object(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((self.interpreter.width(), self.interpreter.height()))
        top_result = self.process_image(image)
        frame = self.overlay.display_result(frame, self.model, self.media_src,
                                            self.interpreter.inference_time,
                                            top_result, self.labels, self.colors)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.labels, self.model)
        self.interpreter = TFLiteInterpreter(self.model)
        self.labels = self.load_labels(self.labels)
        self.generate_colors()

    def run(self):
        self.start()
        self.run_inference(self.detect_object)


class eIQObjectDetectionDNN(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_DETECTION_DNN)
        self.config = self.data['config']

        self.caffe = None
        self.proto = None
        self.nn = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * OpenCV DNN as inference engine."
                "\n   * Deep Neural Networks as default algorithm.\n")

    def usage(self, name=None, labels=True, model=True):
        super().usage(name=name, labels=labels, model=model)

    def gather_data(self):
        super().gather_data()

        self.caffe = os.path.join(self.model_dir, self.data['caffe'])
        self.proto = os.path.join(self.model_dir, self.data['proto'])

    def detect_object(self, frame):
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (self.config['dims'],
                                   self.config['dims']))
        self.nn.create_blob(image, self.config['dims'],
                            self.config['scale'],
                            self.config['normalize'])

        self.nn.run_inference()

        for i in range(self.nn.det.shape[2]):
            confidence = self.nn.det[0, 0, i, 2]

            if confidence > self.config['threshold']:
                index = int(self.nn.det[0, 0, i, 1])
                left = int(width * self.nn.det[0, 0, i, 3])
                top = int(height * self.nn.det[0, 0, i, 4])
                right = int(width * self.nn.det[0, 0, i, 5])
                bottom = int(height * self.nn.det[0, 0, i, 6])
                cv2.rectangle(frame, (left, top), (right, bottom),
                              FONT['color']['black'], FONT['thickness'])

                if index in self.labels:
                    label = ("{0}: {1:.3f}".format(self.labels[index],
                                                   confidence))
                    label_size = cv2.getTextSize(label, FONT['hershey'],
                                                 FONT['size'],
                                                 FONT['thickness'] + 1)[0]
                    top = max(top, label_size[1])
                    cv2.rectangle(frame, (left, top - label_size[1]),
                                  (left + label_size[0], top),
                                  FONT['color']['black'], cv2.FILLED)
                    cv2.putText(frame, label, (left, top), FONT['hershey'],
                                FONT['size'], (255, 255, 255),
                                FONT['thickness'] - 1)

        self.overlay.draw_info(frame, self.caffe, self.media_src,
                               self.nn.inference_time)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.caffe, self.image, self.labels, self.proto)
        self.labels = self.load_labels(self.labels)
        self.nn = OpenCVDNN(self.caffe, self.proto)

    def run(self):
        self.start()
        self.run_inference(self.detect_object)


class eIQObjectDetectionGStreamer(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_DETECTION_GST)

        self.tensor = None

        self.videosrc = None
        self.videofile = None
        self.videofmt = "raw"
        self.src_width = 640
        self.src_height = 480

    def video_config(self):
        if self.args.video_src and self.args.video_src.startswith("/dev/video"):
            self.videosrc = self.args.video_src
        elif self.args.video_src and os.path.exists(self.args.video_src):
            self.videofile = self.args.video_src
            self.src_width = 1920
            self.src_height = 1080

    def input_image_size(self):
        return self.interpreter.input_details[0]['shape'][1:]

    def input_tensor(self):
        return self.tensor(self.interpreter.input_details[0]['index'])()[0]

    def set_input(self, buf):
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            np_buffer = np.reshape(np.frombuffer(mapinfo.data, dtype=np.uint8),
                                   self.input_image_size())
            self.input_tensor()[:, :] = np_buffer
            buf.unmap(mapinfo)

    def output_tensor(self, i):
        output_data = np.squeeze(self.tensor(
                                 self.interpreter.output_details[i]['index'])())
        if 'quantization' not in self.interpreter.output_details:
            return output_data
        scale, zero_point = self.interpreter.output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    @staticmethod
    def avg_fps_counter(window_size):
        window = collections.deque(maxlen=window_size)
        prev = time.monotonic()
        yield 0.0

        while True:
            curr = time.monotonic()
            window.append(curr - prev)
            prev = curr
            yield len(window) / sum(window)

    @staticmethod
    def shadow_text(dwg, x, y, text, font_size=20):
        dwg.add(dwg.text(text, insert=(x + 1, y + 1),
                         fill='black', font_size=font_size))
        dwg.add(dwg.text(text, insert=(x, y),
                fill='white', font_size=font_size))

    def generate_svg(self, src_size, inference_size,
                     inference_box, objs, labels, text_lines):
        dwg = svgwrite.Drawing('', size=src_size)
        src_w, src_h = src_size
        inf_w, inf_h = inference_size
        box_x, box_y, box_w, box_h = inference_box
        scale_x, scale_y = src_w / box_w, src_h / box_h

        for y, line in enumerate(text_lines, start=1):
            self.shadow_text(dwg, 10, y * 20, line)
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            x, y, w, h = int(x * inf_w), int(y * inf_h),\
                         int(w * inf_w), int(h * inf_h)
            x, y = x - box_x, y - box_y
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            self.shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill='none',
                             stroke='red', stroke_width='2'))
        return dwg.tostring()

    def get_output(self, score_threshold=0.1, top_k=3):
        boxes = self.output_tensor(0)
        category = self.output_tensor(1)
        scores = self.output_tensor(2)

        return [gstreamer.make_boxes(i, boxes, category, scores)
                for i in range(top_k) if scores[i] >= score_threshold]

    def start(self):
        self.video_config()
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)
        self.tensor = self.interpreter.interpreter.tensor

    def run(self):
        if not has_svgwrite:
            sys.exit("The module svgwrite needed to run this demo was not "
                     "found. If you want to install it type 'pip3 install "
                     "svgwrite' at your terminal. Exiting...")

        self.start()
        labels = self.load_labels(self.labels)
        w, h, _ = self.input_image_size()
        inference_size = (w, h)
        fps_counter = self.avg_fps_counter(30)

        def user_callback(input_tensor, src_size, inference_box):
            nonlocal fps_counter
            start_time = time.monotonic()
            self.set_input(input_tensor)
            self.interpreter.run_inference()
            objs = self.get_output()
            end_time = time.monotonic()
            text_lines = ['Inference: {:.2f} ms'.format((end_time-start_time)
                                                        * 1000),
                          'FPS: {} fps'.format(round(next(fps_counter)))]
            return self.generate_svg(src_size, inference_size, inference_box,
                                     objs, labels, text_lines)

        gstreamer.run_pipeline(user_callback,
                               src_size=(self.src_width,
                                         self.src_height),
                               appsink_size=inference_size,
                               videosrc=self.videosrc,
                               videofile=self.videofile,
                               videofmt=self.videofmt)
