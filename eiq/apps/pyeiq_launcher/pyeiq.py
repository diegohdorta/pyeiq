#!/usr/bin/env python3
# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
from random import randint
import shutil
from subprocess import Popen

from eiq import __version__ as version
from eiq.apps.pyeiq_launcher.config import APPS, DEMOS
from eiq.config import BASE_DIR
from eiq.utils import args_parser, colored


class PyeIQ:
    def __init__(self):
        self.args = args_parser()
        self.apps = [key for key in APPS.keys()]
        self.demos = [key for key in DEMOS.keys()]
        self.pyeiq_set = {**APPS, **DEMOS}

    def pyeiq_info(self):
        msg = "Welcome to PyeIQ - v{}".format(version)
        self.main_msg(msg)

        print("Available commands:\n")
        print("# pyeiq --clear-cache: Clear cached data generated by demos.")
        print("# pyeiq --info app/demo: app/demo short description and usage."
              "(e.g., pyeiq --run {}).".format(self.randomize(self.demos)))
        print("# pyeiq --list-apps: List the available applications.")
        print("# pyeiq --list-demos: List the available demos.")
        print("# pyeiq --run app/demo: Run the app/demo "
              "(e.g., pyeiq --run {}).\n".format(self.randomize(self.demos)))

    @staticmethod
    def main_msg(msg):
        os.system("clear")
        msg = "   {}   ".format(msg)
        print(colored((len(msg)+4) * "#", "blue"))
        print(colored("#" + (len(msg)+2) * " " + "#", "blue"))
        print(colored("#", "blue"), colored(msg, "green"), colored("#", "blue"))
        print(colored("#" + (len(msg)+2) * " " + "#", "blue"))
        print(colored((len(msg)+4) * "#" + "\n", "blue"))

    def randomize(self, arr):
        return arr[randint(0, (len(arr) - 1))]

    def clear_cache(self):
        if os.path.exists(BASE_DIR):
            print("Removing {}...".format(BASE_DIR))
            try:
                shutil.rmtree(BASE_DIR)
                print("{} has been removed.".format(BASE_DIR))
            except:
                print("Failed to remove {}.".format(BASE_DIR))
            print("")
        else:
            print(colored("No data to be removed.\n", "yellow"))

    def install(self):
        if os.path.isfile(self.args.install):
            print(f"Installing {self.args.install}...")
            pathlib.Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
            file_name = os.path.join(BASE_DIR, os.path.basename(self.args.install))
            shutil.move(self.args.install, file_name)
            shutil.unpack_archive(file_name, BASE_DIR)
            os.remove(file_name)

            for file in os.listdir(BASE_DIR):
                name = file.split(".")[0]
                path = os.path.join(BASE_DIR, name)
                new_file = os.path.join(path, file)
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                shutil.move(os.path.join(BASE_DIR, file), new_file)
                shutil.unpack_archive(new_file, path)

    def print_info(self, target):
        if target in self.pyeiq_set:
            msg = "PyeIQ - {}".format(target)
            self.main_msg(msg)
            print(self.pyeiq_set[target]().description())

            if target in self.demos:
                self.pyeiq_set[target]().usage(target)
        else:
            print(colored("Invalid PyeIQ app/demo.\n"
                          "Type pyeiq --list-demos or --list-apps to "
                          "list the available apps/demos", "yellow"))
        print("")

    def list_apps(self):
        self.main_msg("PyeIQ - Available Applications")

        for app in self.apps:
            print(">>> {}".format(app))

        print("\nFor more details about an application use --info (e.g.,"
              "pyeiq --info {}).".format(self.randomize(self.apps)))
        print("To run an application use --run (e.g., pyeiq --run "
              "{}).\n".format(self.randomize(self.apps)))

    def list_demos(self):
        self.main_msg("PyeIQ - Available Demos")

        for demo in self.demos:
            print(">>> {}".format(demo))

        print("\nFor more details about a demo use --info (e.g.,"
              " pyeiq --info {}).".format(self.randomize(self.demos)))
        print("To run a demo use --run (e.g., pyeiq --run "
              "{}).\n".format(self.randomize(self.demos)))

    def run(self, target):
            if target in self.pyeiq_set:
                msg = "PyeIQ - {}".format(target)
                self.main_msg(msg)
                self.pyeiq_set[target]().run()
            else:
                print(colored("Invalid PyeIQ app/demo.\n"
                              "Type pyeiq --list-demos or --list-apps to "
                              "list the available apps/demos", "yellow"))
            print("")

    def main(self):
        if self.args.install:
            self.install()
        if self.args.clear_cache:
            self.clear_cache()
        elif self.args.info:
            self.print_info(self.args.info)
        elif self.args.list_apps:
            self.list_apps()
        elif self.args.list_demos:
            self.list_demos()
        elif self.args.run:
            self.run(self.args.run)
        else:
            self.pyeiq_info()


if __name__ == '__main__':
    PyeIQ().main()
