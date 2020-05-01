import os

import gi
gi.require_versions({'GdkPixbuf': '2.0', 'Gtk': '3.0'})
from gi.repository import GdkPixbuf, Gtk

import config


class MLPlayer(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title=config.MAIN_WINDOW_TITLE)
        self.set_default_size(640, 480)
        self.set_position(Gtk.WindowPosition.CENTER)

        self.demo_to_run = None
        self.demos_list = self.get_demos()
        self.description = Gtk.Label.new(config.DEFAULT_DEMOS_DESCRIPTION)
        self.image = config.DEFAULT_IMAGE

        self.displayed_image = Gtk.Image()
        self.grid = Gtk.Grid(
            row_spacing = 10, column_spacing = 10,
            border_width = 18, expand=True
        )
        self.add(self.grid)

        self.add_demo_box(0,0,1,1)
        self.add_image_box(6, 0, 3, 3)


    def add_demo_box(self, col=0, row=0, width=1, height=1):
        demos_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=10, expand=True
        )

        demos_label = Gtk.Label.new(None)
        demos_label.set_markup("<b>Select a demo</b>")
        demos_label.set_xalign(config.ALIGN_LEFT)
        demos_box.pack_start(demos_label, False, False, True)

        # Create ComboBox to select demo
        demos_combo = Gtk.ComboBoxText()
        demos_combo.set_entry_text_column(0)
        demos_combo.connect("changed", self.on_demos_combo_changed)
        for demo in self.demos_list:
            demos_combo.append_text(demo)
        demos_box.pack_start(demos_combo, False, False, True)

        demos_description_frame = Gtk.Frame.new("Demo Description")
        demos_description_frame.set_label_align(
            config.ALIGN_CENTER, config.ALIGN_CENTER
        )
        demos_description_frame.add(self.description)
        self.description.set_xalign(0.05)
        demos_box.pack_start(demos_description_frame, False, False, True)

        self.grid.attach(demos_box, col, row, width, height)

    def add_filters(self, dialog):
        filter_images = Gtk.FileFilter()
        filter_images.set_name("Image files")
        for ext in GdkPixbuf.Pixbuf.get_formats():
            filter_images.add_pattern("*." + str(ext.get_name()))
        dialog.add_filter(filter_images)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)

    def add_image_box(self, col=0, row=0, width=1, height=1):
        image_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=10, expand=True
        )

        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(
            config.DEFAULT_IMAGE,
            config.DEFAULT_IMAGE_HEIGHT,
            config.DEFAULT_IMAGE_WIDTH,
        )
        self.displayed_image.set_from_pixbuf(pixbuf)

        choose_image_button = Gtk.Button.new_with_label("Choose an image")
        choose_image_button.connect("clicked", self.on_choose_image_clicked)
        image_box.pack_start(choose_image_button, False, False, True)
        image_box.pack_start(self.displayed_image, False, False, True)

        self.grid.attach(image_box, col, row, width, height)

    def get_demos(self):
        demos_list = []

        if not os.path.isdir(config.DEFAULT_DEMOS_DIR):
            demos_list.append("No PyeIQ demo found")
        else:
            for demo in os.listdir(config.DEFAULT_DEMOS_DIR):
                if "image" in demo:
                    demos_list.append(demo)

        return demos_list

    def on_choose_image_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            "Please choose an image",
            self,
            Gtk.FileChooserAction.OPEN,
            (
                Gtk.STOCK_CANCEL,
                Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN,
                Gtk.ResponseType.OK,
            ),
        )
        dialog.set_current_folder(os.getenv('HOME'))
        self.add_filters(dialog)
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            self.image = dialog.get_filename()

            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(
                self.image,
                config.DEFAULT_IMAGE_HEIGHT,
                config.DEFAULT_IMAGE_WIDTH,
            )
            self.displayed_image.set_from_pixbuf(pixbuf)
        elif response == Gtk.ResponseType.CANCEL:
            pass

        dialog.destroy()

    def on_demos_combo_changed(self, combo):
        demo = combo.get_active_text()

        if demo is not None:
            self.demo_to_run = demo

def main():
    app = MLPlayer()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()