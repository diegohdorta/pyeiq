# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys
import threading

import cv2

from eiq.config import BASE_DIR, ZIP
from eiq.multimedia.overlay import OpenCVOverlay
from eiq.multimedia.utils import GstVideo, save_image, VideoConfig
from eiq.utils import args_parser, check_data, colored, Downloader, file_type, Framerate


class DemoBase:
    def __init__(self, args=None, class_name=None, data=None):
        if args:
            self. args = args
        else:
            self.args = args_parser()

        self.class_name = class_name
        self.framerate = Framerate()
        self.overlay = OpenCVOverlay()

        self.base_dir = os.path.join(BASE_DIR, self.class_name)
        self.save_dir = os.path.join(BASE_DIR, "media")
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.data = data
        self.image = None
        self.labels = None
        self.media_src = None
        self.model = None

        self.interpreter = None

    def usage(self, name=None, labels=False, model=False):
        print("Available parameters:\n")
        print(colored("-d / --download", "yellow") + ": Chooses from which"
              " server the models are going to be downloaded.\n"
              "Options: drive, github, wget. Default: Finds the fastest one.\n"
              "# pyeiq --run {} --download drive\n".format(name))
        print(colored("-i / --image", "yellow") + ": Run the demo on an image"
              " of your choice.\n"
              "# pyeiq --run {} --image /home/root/image.jpg\n".format(name))
        if labels:
            print(colored("-l / --labels", "yellow")+ ": Uses a labels file"
                  " of your choice within the demo.\n"
                  "Labels and models must be compatible for proper outputs.\n"
                  "# pyeiq --run {} --labels /home/root/labels.txt\n".format(name))
        if model:
            print(colored("-m / --model", "yellow") + ": Uses a model file of"
                  " your choice within the demo.\n"
                  "Labels and models must be compatible for proper outputs.\n"
                  "# pyeiq --run {} --model /home/root/model.tflite\n".format(name))
        print(colored("-r / --res", "yellow") + ": Choose the resolution of"
              " your video capture device.\n"
              "Options: full_hd (1920x1080), hd (1280x720), vga (640x480).\n"
              "Default: hd if supported, otherwise uses the best one available.\n"
              "# pyeiq --run {} --res vga\n".format(name))
        print(colored("-f / --video_fwk", "yellow") + ": Choose which framework"
              " is used to display the video.\n"
              "Options: opencv, v4l2, gstreamer (experimental). Default: v4l2.\n"
              "# pyeiq --run {} --video_fwk opencv\n".format(name))
        print(colored("-v / --video_src", "yellow") + ": Run inference in videos."
              "It can be from a video capture device or a video file.\n"
              "Options: True (Find a video capture device automatically),"
              " /dev/video<x>, path_to_video_file.\n"
              "# pyeiq --run {} --video_src /dev/video3\n".format(name))
        print(colored("Multiple parameters can be used at once:\n", "yellow") +
              "# pyeiq --run {} -d drive -v True -f opencv -res vga\n".format(name))

    def validate_data(self, *args):
        if not check_data(os.path.join(self.base_dir, self.class_name + ZIP),
                                       self.data['sha1'], *args):
            self.gather_data()

    def gather_data(self):
        if self.data and 'src' in self.data:
            downloader = Downloader(self.args)
            downloader.retrieve_data(self.data['src'], self.class_name + ZIP,
                                     self.base_dir, self.data['sha1'], True)

        self.validate_media()

        if hasattr(self.args, 'image'):
            if self.args.image and (file_type(self.args.image) == 'image'):
                self.image = self.args.image
        if not self.image and self.data and 'image' in self.data:
            self.image = os.path.join(self.media_dir, self.data['image'])
        if self.image:
            self.media_src = os.path.basename(self.image)

        if hasattr(self.args, 'labels'):
            if self.args.labels and os.path.isfile(self.args.labels):
                self.labels = self.args.labels
        if not self.labels and self.data and 'labels' in self.data:
            self.labels = os.path.join(self.model_dir, self.data['labels'])

        if hasattr(self.args, 'model'):
            if self.args.model and os.path.isfile(self.args.model):
                self.model = self.args.model
        if not self.model and self.data and 'model' in self.data:
            self.model = os.path.join(self.model_dir, self.data['model'])

    def validate_media(self):
        if self.args.image and (not self.args.video_src):
            media = file_type(self.args.image)
            if media == "video":
                sys.exit("Your image parameter seems to be a video file. "
                         "Try to use -v instead.")
            elif (not media) or (media != "image"):
                print("Your image parameter is not a valid image file. "
                      "Using default image...")

        if self.args.video_src:
            media = file_type(self.args.video_src)
            if media == "image":
                sys.exit("Your video_src parameter seems to be an image file. "
                         "Try to use -i instead.")
            elif media and (media != "video"):
                sys.exit("Your video parameter is not a valid video file.")

    @staticmethod
    def load_labels(path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def run_inference(self, inference_func):
        if self.args.video_src:
            video_config = VideoConfig(self.args)
            sink, src = video_config.get_config()

            if os.path.isfile(self.args.video_src):
                self.media_src = os.path.basename(self.args.video_src)
            else:
                self.media_src = video_config.dev.name

            if not src:
                if (not sink) or (not sink.isOpened()):
                    sys.exit("Your video device could not be initialized. Exiting...")
                while sink.isOpened():
                    ret, frame = sink.read()
                    if ret:
                        self.overlay.draw_fps(frame, round(self.framerate.fps))
                        with self.framerate.fpsit():
                            cv2.imshow(self.data['window_title'], inference_func(frame))
                    else:
                        print("Your video device could not capture any image.")
                        break
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                sink.release()
            else:
                with self.framerate.fpsit():
                    gst_video = GstVideo(sink, src, inference_func, self.framerate.fps)
                    gst_video.run()
        else:
            try:
                print("Running inference...")
                inference_func(cv2.imread(self.image, cv2.IMREAD_COLOR))
                frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
                thread = threading.Thread(target=display_image,
                                          args=(self.data['window_title'],
                                                inference_func(frame),
                                                self.save_dir,
                                                self.class_name))
                thread.daemon = True
                thread.start()
                thread.join()
            except KeyboardInterrupt:
                sys.exit("")

        cv2.destroyAllWindows()


def display_image(window_title, image, dest, name):
    save_image(image, dest, name)
    cv2.imshow(window_title, image)
    print("Done.")
    cv2.waitKey()
