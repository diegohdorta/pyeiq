---
layout: default
title: Installation for Developers
parent: Getting Started
nav_order: 2
---

# Installation for Developers
{: .no_toc }

1. TOC
{:toc}
---

## Docker Container

1. Replace the **latest-branch-tag** according to the Official Releases:
```console
$ git clone https://source.codeaurora.org/external/imxsupport/pyeiq && cd pyeiq/
$ git checkout <latest-branch-tag>
$ docker build --build-arg BRANCH=<latest-branch-tag> -t package-latest -f Dockerfile.latest .
```

2. Start the container and copy the generated package:
```console
$ docker run package-latest
$ docker cp $(docker ps -alq):/pyeiq/dist/ latest-package
$ scp latest-package/eiq-<version>.tar.gz root@<ip_address>:/home/root
```

3. Follow the **Deploying the Package** section below.

## Manually

### Software Requirements

1. Install the following packages in the GNU/Linux system:
```console
# apt install python3 python3-pip
```
2. Install required packages:
```console
# pip3 install requests opencv-python
```

### Building the Package

1. Clone the repository and replace the **latest-branch-tag** according to the Official Releases:
```console
$ git clone https://source.codeaurora.org/external/imxsupport/pyeiq && cd pyeiq/
$ git checkout <latest-branch-tag>
```
2. Generate the PyeIQ package:
```console
# python3 setup.py sdist
```
3. Copy the package to the board:
```console
$ scp dist/eiq-<version>.tar.gz root@<ip_address>:/home/root
```

### Deploying the Package

1. Install the PyeIQ package in the board GNU/Linux system:
```console
# pip3 install eiq-<version>.tar.gz
```

2. Check the installation:
```console
# pip3 freeze | grep eiq
```

### Downloading Data on Host Machine (Optional)

When you a run a PyeIQ demo/application, it automatically gathers all the data
required, but it relies on an internet connection. You can also get all that
data on your host machine and send it to the board.

1. Generate the package containing the data:
```console
$ git clone https://source.codeaurora.org/external/imxsupport/pyeiq && cd pyeiq/
$ git checkout <latest-branch-tag>
$ python3 offline_installer.py <boards_ip_address>
```
This is going to generate a package named eiq.zip and send it to the board.

2. Install the data on your board:
```console
# pyeiq --install $HOME/eiq.zip
```

[pypirepo]: https://pypi.org/project/eiq/#description
[pypicaf]: https://source.codeaurora.org/external/imxsupport/pyeiq/
[eiqpackage]: https://img.shields.io/badge/pip3%20install-eiq-green

