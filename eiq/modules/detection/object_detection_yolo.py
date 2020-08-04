# Copyright 2020 Michael Kuo
# Copyright 2020 NXP Semiconductors
#
# Parts of this file were copied from Michael Kuo's github repository 
# respecting its rights. All the modified parts below are according to
# MIT LICENSE terms.
#
# SPDX-License-Identifier:    MIT

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import LEFT, TOP, RIGHT, BOTTOM, CONFIDENCE, CLASSES, OBJ_DETECTION_YOLOV3
from eiq.modules.utils import DemoBase


class eIQObjectDetectionYOLOV3(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_DETECTION_YOLOV3)
        self.config = self.data['config']

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * YOLO as default algorithm.\n")

    def usage(self, name=None, labels=True, model=True):
        super().usage(name=name, labels=labels, model=model)

    @staticmethod
    def expit(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def sort_results(self, e):
        return e[CONFIDENCE]

    def get_input_data(self, image):
        image = image.resize((self.interpreter.width(),
                              self.interpreter.height()))

        n = np.array(image, dtype='float32')
        n = n / 255.0

        return np.array([n], dtype='float32')

    @staticmethod
    def load_labels(labels_path):
        labels = []

        with open(labels_path, 'r') as lines:
            for line in lines.readlines():
                labels.append(line.rstrip())

        return labels

    def non_maximal_suppression(self, results):
        predictions = []

        if len(results):
            results.sort(reverse=True, key=self.sort_results)
            best_prediction = results.pop(0)
            predictions.append(best_prediction)

            while len(results) != 0:
                prediction = results.pop(0)
                overlaps = False

                for j in range(len(predictions)):
                    previous_prediction = predictions[j]

                    intersect_proportion = 0.0
                    primary = previous_prediction
                    secondary = prediction

                    if (primary[LEFT] < secondary[RIGHT]) \
                            and (primary[RIGHT] > secondary[LEFT]) \
                            and (primary[TOP] < secondary[BOTTOM]) \
                            and (primary[BOTTOM] > secondary[TOP]):
                        intersection = max(0, min(primary[RIGHT],
                                                  secondary[RIGHT])
                                                  - max(primary[LEFT],
                                           secondary[LEFT])) \
                                       * max(0, min(primary[BOTTOM],
                                                    secondary[BOTTOM])
                                                    - max(primary[TOP],
                                             secondary[TOP]))

                        main = np.abs(primary[RIGHT] - primary[LEFT]) \
                               * np.abs(primary[BOTTOM] - primary[TOP])
                        intersect_proportion = intersection / main

                    overlaps = overlaps or (intersect_proportion >
                                            self.config['overlap_threshold'])

                if not overlaps:
                    predictions.append(prediction)

        return predictions

    def check_result(self, data):
        results = []

        for row in range(self.config['grid_h']):
            for column in range(self.config['grid_w']):
                for box in range(self.config['boxes']):
                    item = data[row][column]
                    offset = (len(self.labels) + 5) * box

                    confidence = self.expit(item[offset + 4])

                    classes = item[offset + 5: offset + 5 + len(self.labels)]
                    classes = self.softmax(classes)

                    detected_class = np.argmax(classes)
                    max_class = classes[detected_class]

                    confidence_in_class = max_class * confidence

                    if confidence_in_class >= self.config['threshold']:
                        x_pos = (column + self.expit(item[offset])) \
                                * self.config['block_size']
                        y_pos = (row + self.expit(item[offset + 1])) \
                                * self.config['block_size']
                        w = (np.exp(item[offset + 2])
                             * self.config['anchors'][box][0]) \
                            * self.config['block_size']
                        h = (np.exp(item[offset + 3])
                             * self.config['anchors'][box][1]) \
                            * self.config['block_size']

                        left = max(0, x_pos - w / 2)
                        top = max(0, y_pos - h / 2)
                        right = min(self.interpreter.width() - 1,
                                    x_pos + w / 2)
                        bottom = min(self.interpreter.height() - 1,
                                     y_pos + h / 2)

                        results.append([left, top, right, bottom,
                                        confidence_in_class,
                                        self.labels[detected_class]])

        return self.non_maximal_suppression(results)

    def detect_object(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.get_input_data(image)
        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()
        data = self.interpreter.get_tensor(0)[0]
        self.draw_rectangles(frame, self.check_result(data))
        self.overlay.draw_info(frame, self.model, self.media_src,
                               self.interpreter.inference_time)

        return frame

    def draw_rectangles(self, frame, predictions):
        width = frame.shape[1]
        height = frame.shape[0]
        width_ratio = width / self.interpreter.width()
        height_ratio = height / self.interpreter.height()

        for element in predictions:
            x1 = int(element[LEFT] * width_ratio)
            x2 = int(element[RIGHT] * width_ratio)
            y1 = int(element[TOP] * height_ratio)
            y2 = int(element[BOTTOM] * height_ratio)

            top = int(max(0, np.floor(y1 + 0.5).astype('int32')))
            left = int(max(0, np.floor(x1 + 0.5).astype('int32')))
            bottom = int(min(height, np.floor(y2 + 0.5).astype('int32')))
            right = int(min(width, np.floor(x2 + 0.5).astype('int32')))

            label_size = cv2.getTextSize(element[5], FONT['hershey'],
                                         FONT['size'], FONT['thickness'])[0]
            label_left = int(left - 3)
            label_top = int(top - 3)
            label_right = int(left + 3 + label_size[0])
            label_bottom = int(top - 5 - label_size[1])

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.rectangle(frame, (label_left, label_top),
                          (label_right, label_bottom),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, element[CLASSES], (left, top - 4),
                        FONT['hershey'], FONT['size'], FONT['color']['black'],
                        FONT['thickness'])

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.labels, self.model)
        self.labels = self.load_labels(self.labels)
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()
        self.run_inference(self.detect_object)
