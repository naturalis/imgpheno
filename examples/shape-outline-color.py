#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This program is a demonstration of the shape description function shape_outline.

The following key bindings are available:
  Q - exit
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import common
import imgpheno as ft

img = None
img_src = None
bin_mask = None
box = None
res = None

def main():
    global res

    print __doc__

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Aprse arguments
    parser = argparse.ArgumentParser(description='Get the rough shape from the main object')
    parser.add_argument('path', metavar='PATH', help='Path to image file')
    parser.add_argument('--max-size', metavar='N', type=float, help="Scale the input image down if its perimeter exceeds N. Default is no scaling.")
    parser.add_argument('--iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser.add_argument('--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    parser.add_argument('-k', metavar='N', type=int, default=20, help="The resolution for the outline feature. Default is 20.")
    args = parser.parse_args()
    res = args.k

    # Create UI
    cv2.namedWindow('image')
    cv2.createTrackbar('Position', 'image', 0, args.k-2, set_position)

    process_image(args, args.path)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    return 0

def set_position(x):
    global img, res
    if img == None:
        return
    draw_outline(x, res)

def process_image(args, path):
    global img, img_src, outline, box, bin_mask

    img = cv2.imread(path)
    if img == None or img.size == 0:
        logging.error("Failed to read %s" % path)
        exit(1)

    logging.info("Processing %s..." % path)

    # Scale the image down if its perimeter exceeds the maximum (if set).
    img = common.scale_max_perimeter(img, args.max_size)
    img_src = img.copy()

    # Perform segmentation.
    logging.info("- Segmenting...")
    mask = common.grabcut(img, args.iters, None, args.margin)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Obtain contours (all points) from the mask.
    contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get bounding rectange of the largest contour.
    box = cv2.boundingRect(contour)

    # And draw it.
    logging.info("- Done")
    draw_outline(0, args.k)

def draw_outline(i, k):
    global img, img_src, box, bin_mask

    img = img_src.copy()
    im_x, im_y, im_w, im_h = box

    img = cv2.bitwise_and(img, img, mask=bin_mask)

    # Calculate the points for the horizontal outline.
    step = float(im_h) / (k - 1)
    y = int((i * step) + im_y)
    y2 = int(((i+1) * step) + im_y)
    pt1 = (im_x, y)
    pt2 = (im_x+im_w, y2)

    cv2.rectangle(img, pt1, pt2, common.COLOR['green'], 1)

    # Calculate the points for the vertical outline.
    step = float(im_w) / (k - 1)
    x = int((i * step) + im_x)
    x2 = int(((i+1) * step) + im_x)
    pt1 = (x, im_y)
    pt2 = (x2, im_y+im_h)

    cv2.rectangle(img, pt1, pt2, common.COLOR['red'], 1)

    cv2.imshow('image', img)

if __name__ == "__main__":
    main()
