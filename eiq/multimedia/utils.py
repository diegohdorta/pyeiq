# Copyright 2020 NXP Semiconductors
#
# This file was partial copied from the link below and modified according to
# its license terms.
#
# Reference:
# https://github.com/toradex/torizon-samples/blob/master/dlr-gstreamer/inference.py
#
# SPDX-License-Identifier: BSD-3-Clause

import atexit
import os
import pathlib
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np

from eiq.config import PNG
from eiq.multimedia.gstreamer import set_appsink_pipeline, set_appsink_video_pipeline, set_appsrc_pipeline
from eiq.multimedia.v4l2 import v4l2_camera_pipeline, v4l2_video_pipeline
from eiq.multimedia.overlay import OpenCVOverlay


class VideoDevice:
    def __init__(self):
        self.name = None
        self.caps = None
        self.default_caps = None
        self.full_hd_caps = None
        self.hd_caps = None
        self.vga_caps = None


class Caps:
    def __init__(self):
        self.name = None
        self.format = None
        self.width = None
        self.height = None
        self.framerate = None


class Devices:
    def __init__(self):
        self.devices = []

    def get_video_devices(self):
        Gst.init()
        dev_monitor = Gst.DeviceMonitor()
        dev_monitor.add_filter("Video/Source")
        dev_monitor.start()

        for dev in dev_monitor.get_devices():
            video_dev = VideoDevice()
            dev_props = dev.get_properties()
            dev_caps = dev.get_caps()

            name = dev_props.get_string("device.path")
            caps = self.get_device_caps(dev_caps.normalize())
            full_hd_caps, hd_caps, vga_caps = self.get_std_caps(caps)
            default_caps = hd_caps
            if (not default_caps) and caps:
                default_caps = caps[0]

            video_dev.name = name
            video_dev.caps = caps
            video_dev.default_caps = default_caps
            video_dev.full_hd_caps = full_hd_caps
            video_dev.hd_caps = hd_caps
            video_dev.vga_caps = vga_caps
            self.devices.append(video_dev)

        dev_monitor.stop()

    @staticmethod
    def get_device_caps(dev_caps):
        caps_list = []

        for i in range(dev_caps.get_size()):
            if dev_caps.get_structure(i).get_name() != "video/x-raw":
                continue

            caps = Caps()
            caps_struct = dev_caps.get_structure(i)
            caps.name = caps_struct.get_name()
            caps.format = caps_struct.get_string("format")
            caps.width = caps_struct.get_int("width")[1]
            caps.height = caps_struct.get_int("height")[1]
            framerate = ("{}/{}".format(*caps_struct.get_fraction(
                                        "framerate")[1:]))
            caps.framerate = framerate
            caps_list.append(caps)

        return caps_list

    @staticmethod
    def get_std_caps(dev_caps):
        full_hd_caps = None
        hd_caps = None
        vga_caps = None

        for caps in dev_caps:
            if  (caps.width == 1920) and (caps.height == 1080):
                full_hd_caps = caps
            elif (caps.width == 1280) and (caps.height == 720):
                hd_caps = caps
            elif (caps.width == 640) and (caps.height == 480):
                vga_caps = caps

        return full_hd_caps, hd_caps, vga_caps

    def search_device(self, dev_name):
        dev = None

        if dev_name.startswith("/dev/video"):
            for device in self.devices:
                if device.name == dev_name:
                    dev = device

                if not dev:
                    print("The specified video_src was not found.\n"
                          "Searching for default video device...")

        if not dev and self.devices:
            dev = self.devices[0]
        elif not dev:
            sys.exit("No video device found. Exiting...")

        print("Using {} as video device".format(dev.name))
        return dev


class GstVideo:
    def __init__(self, sink, src, inference_func, framerate):
        self.sink = sink
        self.src = src
        self.inference_func = inference_func
        self.fps = framerate
        self.appsource = None
        self.sink_pipeline = None
        self.src_pipeline = None
        self.overlay = OpenCVOverlay()

        atexit.register(self.exit_handler)

    def exit_handler(self):
        self.sink_pipeline.set_state(Gst.State.NULL)
        self.src_pipeline.set_state(Gst.State.NULL)

    def run(self):
        self.sink_pipeline = Gst.parse_launch(self.sink)
        appsink = self.sink_pipeline.get_by_name('sink')
        appsink.connect("new-sample", self.on_new_frame)

        self.src_pipeline = Gst.parse_launch(self.src)
        self.appsource = self.src_pipeline.get_by_name('src')

        self.sink_pipeline.set_state(Gst.State.PLAYING)
        bus1 = self.sink_pipeline.get_bus()
        self.src_pipeline.set_state(Gst.State.PLAYING)
        bus2 = self.src_pipeline.get_bus()

        while True:
            message = bus1.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    self.sink_pipeline.set_state(Gst.State.NULL)
                    self.src_pipeline.set_state(Gst.State.NULL)
                    sys.exit("ERROR bus 1: {}: {}".format(err, debug))

                if message.type == Gst.MessageType.WARNING:
                    err, debug = message.parse_warning()
                    print("WARNING bus 1: {}: {}".format(err, debug))

            message = bus2.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    self.sink_pipeline.set_state(Gst.State.NULL)
                    self.src_pipeline.set_state(Gst.State.NULL)
                    sys.exit("ERROR bus 2: {}: {}".format(err, debug))

                if message.type == Gst.MessageType.WARNING:
                    err, debug = message.parse_warning()
                    print("WARNING bus 2: {}: {}".format(err, debug))

    def on_new_frame(self, sink):
        sample = sink.emit("pull-sample")
        caps = sample.get_caps().get_structure(0)
        resize = (caps.get_value('height'), caps.get_value('width'), 3)

        mem = sample.get_buffer()
        _, arr = mem.map(Gst.MapFlags.READ)
        img = np.ndarray(resize, buffer=arr.data, dtype=np.uint8)

        self.overlay.draw_fps(img, round(self.fps))

        img = self.inference_func(img)
        self.appsource.emit("push-buffer", Gst.Buffer.new_wrapped(img.tobytes()))
        mem.unmap(arr)

        return Gst.FlowReturn.OK


class VideoConfig:
    def __init__(self, args):
        self.res = args.res
        self.video_fwk = args.video_fwk
        self.video_src = args.video_src
        self.devices = Devices()
        self.devices.get_video_devices()

        if not os.path.isfile(self.video_src):
            self.dev = self.devices.search_device(self.video_src)
            self.dev_caps = self.get_caps()

    def get_caps(self):
        if self.res == "full_hd":
                return self.validate_caps(self.dev.full_hd_caps)
        elif self.res == "vga":
                return self.validate_caps(self.dev.vga_caps)
        else:
            if self.res != "hd":
                print("Invalid resolution, trying to use hd instead.")

            return self.validate_caps(self.dev.hd_caps)

    def validate_caps(self, caps):
            if caps:
                return caps
            else:
                print("Resolution not supported. Using "
                      "{}x{} instead.".format(self.dev.default_caps.width,
                                              self.dev.default_caps.height))
                return self.dev.default_caps

    def get_config(self):
        if self.video_fwk == "gstreamer":
            return self.gstreamer_config()
        elif self.video_fwk == "opencv":
            return self.opencv_config()
        else:
            if self.video_fwk != "v4l2":
                print("Invalid video framework. Using v4l2 instead.")

            return self.v4l2_config()

    def gstreamer_config(self):
        if self.video_src and os.path.isfile(self.video_src):
            sink_pipeline = set_appsink_video_pipeline(self.video_src)
            src_pipeline = set_appsrc_pipeline()
            return sink_pipeline, src_pipeline
        else:
            sink_pipeline = set_appsink_pipeline(device=self.dev.name)
            src_pipeline = set_appsrc_pipeline(width=self.dev_caps.width,
                                               height=self.dev_caps.height)
            return sink_pipeline, src_pipeline

    def opencv_config(self):
        if self.video_src and os.path.isfile(self.video_src):
            return cv2.VideoCapture(self.video_src), None
        else:
            dev = int(self.dev.name[10:])
            video = cv2.VideoCapture(dev)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, self.dev_caps.width)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.dev_caps.height)

            return video, None

    def v4l2_config(self):
        if self.video_src and os.path.isfile(self.video_src):
            pipeline = v4l2_video_pipeline(self.video_src)
        else:
            pipeline = v4l2_camera_pipeline(width=self.dev_caps.width,
                                            height=self.dev_caps.height,
                                            device=self.dev.name,
                                            frate=self.dev_caps.framerate)

        return cv2.VideoCapture(pipeline), None


def save_image(image, dest, name):
    if not os.path.exists(dest):
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

    count = 1
    img_name = os.path.join(dest, name+PNG)

    if os.path.exists(img_name):
        img_name = "{}({}){}".format(name, count, PNG)
        img_name = os.path.join(dest, img_name)

        while os.path.exists(img_name):
            count += 1
            img_name = "{}({}){}".format(name, count, PNG)
            img_name = os.path.join(dest, img_name)

    cv2.imwrite(img_name, image)
    print("Image saved as {}".format(img_name))
