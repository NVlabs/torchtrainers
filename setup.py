#!/usr/bin/python3
#
# Copyright (c) 2013-2019 Thomas Breuel. All rights reserved.
# This file is part of torchtrainers (see unknown).
# See the LICENSE file for licensing terms (BSD-style).
#

from __future__ import print_function

import sys
import glob
from distutils.core import setup  # , Extension, Command

#scripts = glob.glob("wl-*[a-z]")

setup(
    name='torchtrainers',
    version='v0.0',
    author="Thomas Breuel",
    description="Simple training tools for PyTorch.",
    packages=["torchtrainers"],
    #scripts=scripts,
)
