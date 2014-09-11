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

import common
import imgpheno as ft

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

    # Scale the image down if its perimeter exceeds the maximum.
    img = common.scale_max_perimeter(img, 1000)

    cs_str = "BGR"
    hists = ft.color_histograms(img)
    for i, hist in enumerate(hists):
        print "%s: %s" % (cs_str[i], hist.astype(int).ravel())
    print "BGR ranges:", get_min_max(img)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cs_str = "HSV"
    hists = ft.color_histograms(img_hsv)
    for i, hist in enumerate(hists):
        print "%s: %s" % (cs_str[i], hist.astype(int).ravel())
    print "HSV ranges:", get_min_max(img_hsv)

    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    cs_str = "Luv"
    hists = ft.color_histograms(img_luv)
    for i, hist in enumerate(hists):
        print "%s: %s" % (cs_str[i], hist.astype(int).ravel())
    print "LUV ranges:", get_min_max(img_luv)

    return 0

def get_min_max(img):
    """Returns the color space ranges for an image."""
    minval = np.amin(img, (1,0))
    maxval = np.amax(img, (1,0))
    return (tuple(minval), tuple(maxval))

if __name__ == "__main__":
    main()

