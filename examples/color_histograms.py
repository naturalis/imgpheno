#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This program is a demonstration of feature extraction function color_histograms.

The histograms for the BGR color space are printed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import features as ft

MAXDIM = 500

def main():
    print __doc__

    parser = argparse.ArgumentParser(description='Get image color statistics')
    parser.add_argument('image', metavar='FILE', help='Input image')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img == None or img.size == 0:
        sys.stderr.write("Failed to read %s\n" % args.image)
        return 1

    sys.stderr.write("Processing %s...\n" % args.image)

    # Scale the image down if necessary.
    max_px = max(img.shape[:2])
    if max_px > MAXDIM:
        rf = float(MAXDIM) / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)

    cs_str = "BGR"
    hists = ft.color_histograms(img, [10,10,10])
    for i, hist in enumerate(hists):
        print "%s: %s" % (cs_str[i], hist.astype(int).ravel())

    return 0

if __name__ == "__main__":
    main()

