# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import FACIAL_EXPRESSION_DETECTION, FACE_EYES_DETECTION
from eiq.modules.utils import DemoBase
from eiq.utils import Timer


class eIQFacialExpressionDetection(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__,
                         FACIAL_EXPRESSION_DETECTION)

        self.face_cascade = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite and OpenCV as inference engines."
                "\n   * CNN and Haar Cascade as default algorithms.\n")

    def usage(self, name=None, labels=False, model=True):
        super().usage(name=name, labels=labels, model=model)

    def gather_data(self):
        super().gather_data()

        self.face_cascade = os.path.join(self.model_dir,
                                         self.data['face_cascade'])

    @staticmethod
    def preprocess_image(image, x, y, w, h):
        image = image[y:y+h, x:x+w]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = image.resize((200, 200))
        image = np.expand_dims(image, axis=0)

        if np.max(image) > 1:
            image = image / 255.0

        return image.astype(np.float32)

    def detect_facial_expression(self, image, x, y, w, h):
        image = self.preprocess_image(image, x, y, w, h)

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        results = self.interpreter.get_tensor(0)
        classes = int(np.argmax(results, axis=1))

        if 0 <= classes <= 5:
            return self.data['expressions'][classes]
        else:
            return 'surprise'

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=5,
                                                   minSize=(150, 150))

        for (x, y, w, h) in faces:
            expression = self.detect_facial_expression(gray, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, expression, (x, y-5), FONT['hershey'],
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        self.overlay.draw_info(frame, self.model, self.media_src,
                               self.interpreter.inference_time)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.model, self.face_cascade)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()
        self.run_inference(self.detect_face)


class eIQFaceAndEyesDetection(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, FACE_EYES_DETECTION)
        self.timer = Timer()

        self.eye_cascade = None
        self.face_cascade = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * OpenCV as inference engine."
                "\n   * Haar Cascade as default algorithm.\n")

    def usage(self, name=None, labels=False, model=False):
        super().usage(name=name, labels=labels, model=model)

    def gather_data(self):
        super().gather_data()

        self.eye_cascade = os.path.join(self.model_dir,
                                        self.data['eye_cascade'])
        self.face_cascade = os.path.join(self.model_dir,
                                         self.data['face_cascade'])

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with self.timer.timeit():
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                              (0, 255, 0), 2)

        self.overlay.draw_info(frame, "haarcascade_frontalface_default.xml",
                               self.media_src, self.timer.time)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.model,
                           self.eye_cascade, self.face_cascade)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)

    def run(self):
        self.start()
        self.run_inference(self.detect_face)
