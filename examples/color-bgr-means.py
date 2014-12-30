#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This program is a demonstration of the shape description function color_bgr_means.

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
    parser = argparse.ArgumentParser(
        description="Get the rough shape from the main object"
    )
    parser.add_argument(
        'path',
        metavar='PATH',
        help="Path to image file")
    parser.add_argument(
        '--max-size',
        metavar='N',
        type=float,
        help="Scale the input image down if its perimeter exceeds N. "\
        "Default is no scaling.")
    parser.add_argument(
        '--iters',
        metavar='N',
        type=int,
        default=5,
        help="The number of segmentation iterations. Default is 5.")
    parser.add_argument(
        '--margin',
        metavar='N',
        type=int,
        default=1,
        help="The margin of the foreground rectangle from the edges. "\
        "Default is 1.")
    parser.add_argument(
        '--bins',
        metavar='N',
        type=int,
        default=21,
        help="The number of horizontal and vertical bins. Default is 21.")
    args = parser.parse_args()
    res = args.bins

    # Create UI
    cv2.namedWindow('BGR Means')
    cv2.createTrackbar('Bin', 'BGR Means', 0, args.bins-1, set_position)

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
    draw_sections(x, res)

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
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD),
        255, 0).astype('uint8')

    # Obtain contours (all points) from the mask.
    contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)

    # Get bounding rectange of the largest contour.
    props = ft.contour_properties([contour], 'BoundingRect')
    box = props[0]['BoundingRect']

    # And draw it.
    logging.info("- Done")
    draw_sections(0, args.bins)

def draw_sections(i, bins):
    global img, img_src, box, bin_mask

    img = img_src.copy()
    img = cv2.bitwise_and(img, img, mask=bin_mask)

    rect_x, rect_y, width, height = box
    centroid = (width/2+rect_x, height/2+rect_y)
    longest = max([width, height])
    incr = float(longest) / bins

    # Calculate X and Y starting points.
    x_start = centroid[0] - (longest / 2)
    y_start = centroid[1] - (longest / 2)

    x = (incr * i) + x_start
    y = (incr * i) + y_start

    x_incr = x + incr
    y_incr = y + incr
    x_end = x_start + longest
    y_end = y_start + longest

    # Remove negative values, which otherwise result in reverse indexing.
    if x_start < 0: x_start = 0
    if y_start < 0: y_start = 0
    if x < 0: x = 0
    if y < 0: y = 0
    if x_incr < 0: x_incr = 0
    if y_incr < 0: y_incr = 0
    if x_end < 0: x_end = 0
    if y_end < 0: y_end = 0

    # Convert back to integers.
    y = int(y)
    y_start = int(y_start)
    y_incr = int(y_incr)
    y_end = int(y_end)
    x = int(x)
    x_start = int(x_start)
    x_incr = int(x_incr)
    x_end = int(x_end)

    # Draw the horizontal section.
    cv2.rectangle(img, (x_start,y), (x_end,y_incr), common.COLOR['green'], 1)

    # Draw the vertical section.
    cv2.rectangle(img, (x,y_start), (x_incr,y_end), common.COLOR['red'], 1)

    cv2.imshow('BGR Means', img)

if __name__ == "__main__":
    main()
