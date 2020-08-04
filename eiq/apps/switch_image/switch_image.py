# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
from random import randint
from socket import gethostname
from stat import S_IEXEC
import sys
import threading

import gi
gi.require_versions({'GdkPixbuf': "2.0", 'Gtk': "3.0"})
from gi.repository.GdkPixbuf import Colorspace, Pixbuf
from gi.repository import GLib, Gtk

import cv2
import numpy as np

from eiq.apps.config import SWITCH_IMAGE
from eiq.apps.utils import run_label_image
from eiq.config import BASE_DIR, ZIP
from eiq.utils import args_parser, check_data, Downloader


class eIQSwitchLabelImage(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title=SWITCH_IMAGE['window_title'])
        self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.args = args_parser()
        self.data = SWITCH_IMAGE
        self.msg = self.data['msg']
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.bin_dir = os.path.join(self.base_dir, "bin")
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.binary = os.path.join(self.bin_dir, self.data['bin'])
        self.model = os.path.join(self.model_dir, self.data['model'])
        self.labels = os.path.join(self.model_dir, self.data['labels'])

        grid = Gtk.Grid(row_spacing=10, column_spacing=10, border_width=18)
        self.add(grid)
        self.hw_accel = self.get_hw_accel()
        self.value_returned = []
        self.label_returned = []
        self.value_returned_box = []
        self.label_returned_box = []
        self.displayed_image = Gtk.Image()
        self.image_map = Gtk.ListStore(str)

        download = Downloader(self.args)
        if not check_data(os.path.join(self.base_dir,
                                       self.__class__.__name__ + ZIP),
                          self.data['sha1'], self.binary, self.model,
                          self.labels):
            download.retrieve_data(self.data['src'],
                                   self.__class__.__name__ + ZIP,
                                   self.base_dir, self.data['sha1'], True)
        os.chmod(self.binary, S_IEXEC)

        self.get_bmp_images()

        grid.set_column_homogeneous(True)
        grid.set_row_homogeneous(True)

        model_box = Gtk.Box()
        model_name_box = Gtk.Box()
        result_box = Gtk.Box()
        percentage_box = Gtk.Box()
        inference_box = Gtk.Box()
        inference_value_box = Gtk.Box()
        image_label_box = Gtk.Box()
        image_map_box = Gtk.Box()
        image_box = Gtk.Box()

        self.image_combo_box = Gtk.ComboBox.new_with_model(self.image_map)
        self.image_combo_box.connect("changed", self.on_combo_image_changed)
        image_rendered_list = Gtk.CellRendererText()
        self.image_combo_box.pack_start(image_rendered_list, True)
        self.image_combo_box.add_attribute(image_rendered_list, "text", 0)

        for i in range(5):
            self.value_returned.append(Gtk.Entry())
            self.label_returned.append(Gtk.Label())
            self.value_returned_box.append(Gtk.Box())
            self.label_returned_box.append(Gtk.Box())

        self.image = self.get_random_image()
        self.set_displayed_image(self.image)
        self.set_initial_entrys()

        model_label = Gtk.Label()
        model_label.set_markup(self.msg['model_name'])
        self.model_name_label = Gtk.Label.new(None)
        result_label = Gtk.Label()
        result_label.set_markup(self.msg['labels'])
        percentage_label = Gtk.Label()
        percentage_label.set_markup(self.msg['confidence'])
        inference_label = Gtk.Label()
        inference_label.set_markup(self.msg['inf_time'])
        self.inference_value_label = Gtk.Label.new(None)
        image_label = Gtk.Label()
        image_label.set_markup(self.msg['select_img'])
        image_label.set_xalign(0.0)

        model_box.pack_start(model_label, True, True, 0)
        model_name_box.pack_start(self.model_name_label, True, True, 0)
        result_box.pack_start(result_label, True, True, 0)
        percentage_box.pack_start(percentage_label, True, True, 0)
        inference_box.pack_start(inference_label, True, True, 0)
        inference_value_box.pack_start(self.inference_value_label, True, True, 0)
        image_label_box.pack_start(image_label, True, True, 0)
        image_map_box.pack_start(self.image_combo_box, True, True, 0)

        for i in range(5):
            self.label_returned_box[i].pack_start(self.label_returned[i], True, True, 0)
            self.value_returned_box[i].pack_start(self.value_returned[i], True, True, 0)

        image_box.pack_start(self.displayed_image, True, True, 0)

        self.cpu_button = Gtk.Button.new_with_label("CPU")
        self.cpu_button.connect("clicked", self.run_cpu_inference)
        grid.attach(self.cpu_button, 3, 0, 1, 1)
        self.npu_button = Gtk.Button.new_with_label(self.hw_accel)
        self.npu_button.connect("clicked", self.run_npu_inference)
        grid.attach(self.npu_button, 4, 0, 1, 1)

        grid.attach(model_box, 0, 5, 2, 1)
        grid.attach(model_name_box, 0, 6, 2, 1)
        grid.attach(inference_box, 0, 7, 2, 1)
        grid.attach(inference_value_box, 0, 8, 2, 1)
        grid.attach(result_box, 6, 3, 1, 1)
        grid.attach(percentage_box, 7, 3, 1, 1)

        for i in range(5):
            grid.attach(self.label_returned_box[i], 6, (4+i), 1, 1)
            grid.attach(self.value_returned_box[i], 7, (4+i), 1, 1)

        grid.attach(image_label_box, 0, 2, 2, 1)
        grid.attach(image_map_box, 0, 3, 2, 1)
        grid.attach(image_box, 2, 1, 4, 10)

    def set_displayed_image(self, image):
        img = cv2.imread(image)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        arr = np.ndarray.tostring(img)
        pixbuf = Pixbuf.new_from_data(arr, Colorspace.RGB, False, 8, width,
                                      height, width * 3, None, None)
        self.displayed_image.set_from_pixbuf(pixbuf)

    def on_combo_image_changed(self, combo):
        iterr = combo.get_active_iter()
        if iterr is not None:
            model = combo.get_model()
            image_name = model[iterr][0]
            print("Selected image: {}".format(image_name))
            self.image = os.path.join(self.media_dir, image_name)
            self.set_displayed_image(self.image)

    def get_bmp_images(self):
        if not os.path.exists(self.media_dir) or not os.listdir(self.media_dir):
            sys.exit(f"{self.media_dir} does not exists, try to run"
                     " 'pyeiq --clear-cache' and run this application again.")

        for file in os.listdir(self.media_dir):
            self.image_map.append([file])

    def get_random_image(self):
        images = []

        for file in os.listdir(self.media_dir):
            file = os.path.join(self.media_dir, file)
            images.append(file)

        index = randint(0, (len(images) - 1))
        return images[index]

    @staticmethod
    def get_hw_accel():
        if gethostname() == "imx8mpevk":
            return "NPU"

        return "GPU"

    def set_initial_entrys(self):
        for i in range(5):
            self.label_returned[i].set_text("")
            self.value_returned[i].set_editable(False)
            self.value_returned[i].set_can_focus(False)
            self.value_returned[i].set_text("0%")
            self.value_returned[i].set_alignment(xalign=0)
            self.value_returned[i].set_progress_fraction(-1)

    def set_returned_entrys(self, value):
        x = 0
        for i in value[2:]:
            self.label_returned[x].set_text(str(i[2]))
            self.value_returned[x].set_progress_fraction(float(i[0]))
            self.value_returned[x].set_text("{0:.2f}%".format(float(i[0])*100))
            x = x + 1

    def set_pre_inference(self):
        self.set_initial_entrys()
        self.model_name_label .set_text("")
        self.inference_value_label.set_text("Running...")
        self.image_combo_box.set_sensitive(False)
        self.cpu_button.set_sensitive(False)
        self.npu_button.set_sensitive(False)

    def set_post_inference(self, x):
        self.model_name_label.set_text(x[0])
        self.inference_value_label.set_text("{0:.2f} ms".format(float(x[1])))
        self.set_returned_entrys(x)
        self.image_combo_box.set_sensitive(True)
        self.cpu_button.set_sensitive(True)
        self.npu_button.set_sensitive(True)

        for i in x:
            print(i)

    def run_cpu_inference(self, window):
        self.set_pre_inference()
        thread = threading.Thread(target=self.run_inference, args=(False,))
        thread.daemon = True
        thread.start()

    def run_inference(self, accel):
        if accel:
            print("Running Inference on {0}".format(self.hw_accel))
        else:
            print("Running Inference on CPU")

        x = run_label_image(accel, self.binary, self.model,
                            self.image, self.labels)

        GLib.idle_add(self.set_post_inference, x)

    def run_npu_inference(self, window):
        self.set_pre_inference()
        thread = threading.Thread(target=self.run_inference, args=(True,))
        thread.daemon = True
        thread.start()

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * CNN as default algorithm.\n\n"
                "This application offers a graphical interface for users to run\n"
                "an object classification demo using either CPU or GPU/NPU to\n"
                "perform inference on a list of available images.")

    @staticmethod
    def run():
        app = eIQSwitchLabelImage()
        app.connect("destroy", Gtk.main_quit)
        app.show_all()
        Gtk.main()
