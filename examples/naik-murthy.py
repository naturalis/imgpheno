#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This program is a demonstration of Hue-preserving color image enhancement. [1]

1. Naik, S. K. & Murthy, C. A. Hue-preserving color image enhancement
   without gamut problem. IEEE Trans. Image Process. 12, 1591–8 (2003).

The following key bindings are available:
  0 - Original image
  1 - Linearly enhanced
  2 - Enhanced using S-type function (delta1=0, delta2=3, m=0.5, n=2)
    W - Linearly enhanced + enhanced using S-type function (delta1=0, delta2=3, m=0.5, n=2)
    S - Linearly enhanced + enhanced using S-type function (delta1=0, delta2=3, m=1.5, n=2)
  3 - Enhanced using histogram equalization
  ESC - exit
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import common
import features as ft

def main():
    print __doc__

    parser = argparse.ArgumentParser(description='Test image segmentation')
    parser.add_argument('image', metavar='FILE', help='Input image')
    parser.add_argument('--max-size', metavar='N', type=float, help="Scale the input image down if its perimeter exceeds N. Default is no scaling.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img == None or img.size == 0:
        sys.stderr.write("Failed to read %s\n" % args.image)
        return -1

    sys.stderr.write("Processing %s...\n" % args.image)

    # Scale the image down if its perimeter exceeds the maximum (if set).
    img = common.scale_max_perimeter(img, args.max_size)

    # Linear enhancement
    enhanced1 = ft.naik_murthy_linear(img)

    # Nonlinear enhancement using S-type function
    enhanced2a = ft.naik_murthy_nonlinear(img, ft.s_type_enhancement, 0, 3, 0.5, 2)
    enhanced2b = ft.naik_murthy_nonlinear(enhanced1, ft.s_type_enhancement, 0, 3, 0.5, 2)
    enhanced2c = ft.naik_murthy_nonlinear(enhanced1, ft.s_type_enhancement, 0, 3, 1.5, 2)

    # Enhancement using histogram equalization
    sums = bgr_to_sums(img) # Range 0..(255*3-1)
    sums = cv2.normalize(sums, None, 0, 255, cv2.NORM_MINMAX) # Range 0..255
    sums = cv2.equalizeHist(np.uint8(sums))
    sums = cv2.normalize(np.float32(sums), None, 0, 3, cv2.NORM_MINMAX) # Range 0..3
    enhanced3 = ft.naik_murthy_nonlinear(img, sums, fmap=True)

    # Display the image in a window.
    cv2.namedWindow('image')
    cv2.imshow('image', img)

    while True:
        k = cv2.waitKey(0) & 0xFF

        if k == ord('0'):
            cv2.imshow('image', img)
        elif k == ord('1'):
            cv2.imshow('image', enhanced1)
        elif k == ord('2'):
            cv2.imshow('image', enhanced2a)
        elif k == ord('w'):
            cv2.imshow('image', enhanced2b)
        elif k == ord('s'):
            cv2.imshow('image', enhanced2c)
        elif k == ord('3'):
            cv2.imshow('image', enhanced3)
        elif k == 27:
            break

    cv2.destroyAllWindows()
    return 0

def bgr_to_sums(img):
    """Returns a 2d array with sums of the pixel BGR values."""
    y = np.zeros(img.shape[:2], dtype=np.float32)
    itemset = y.itemset
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            l = img[i,j].sum()
            itemset((i,j), l)
    return y

if __name__ == "__main__":
    main()
