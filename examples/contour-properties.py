#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This program is a demonstration of the shape description function contour_properties.

The following key bindings are available:
  Q - exit
"""

import argparse
import logging
import math
import mimetypes
import os
import sys
import pprint

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import common
import imgpheno as ft

BLACK = (0,0,0)
GRAY = (105,105,105)
BLUE = (255,0,0)
CYAN = (255,255,0)
GREEN = (0,255,0)
RED = (0,0,255)

img = None
img_src = None

def main():
    print __doc__

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Aprse arguments
    parser = argparse.ArgumentParser(description='Get the rough shape from the main object')
    parser.add_argument('path', metavar='PATH', help='Path to image file')
    parser.add_argument('--max-size', metavar='N', type=float, help="Scale the input image down if its perimeter exceeds N. Default is no scaling.")
    parser.add_argument('--iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser.add_argument('--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    args = parser.parse_args()

    # Create UI
    cv2.namedWindow('image')

    process_image(args, args.path)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    return 0

def process_image(args, path):
    global img, img_src

    img = cv2.imread(path)
    if img == None or img.size == 0:
        logging.error("Failed to read %s" % path)
        exit(1)

    logging.info("Processing %s..." % path)

    # Scale the image down if its perimeter exceeds the maximum (if set).
    img = common.scale_max_perimeter(img, args.max_size)
    img_src = img.copy()

    # Perform segmentation
    logging.info("- Segmenting...")
    mask = common.grabcut(img, args.iters, None, args.margin)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate contour properties.
    logging.info("- Computing contour properties...")
    props = ft.contour_properties(contours, 'all')

    # Print properties.
    pprint.pprint(props)

    # Draw some properties.
    draw_props(props)

def draw_props(properties):
    global img, img_src

    img = img_src.copy()

    for props in properties:
        if props['Ellipse'] != None:
            cv2.circle(img, props['Centroid'], 5, BLUE, 1)
            cv2.ellipse(img, props['Ellipse'], CYAN)
            cv2.drawContours(img, [props['ConvexHull']], 0, RED, 1)

            major_axis = ft.angled_line(props['Centroid'], props['Orientation'], props['MajorAxisLength']/2)
            cv2.line(img, tuple(major_axis[0]), tuple(major_axis[1]), RED)

            minor_axis = ft.angled_line(props['Centroid'], props['Orientation'] + 90, props['MinorAxisLength']/2)
            cv2.line(img, tuple(minor_axis[0]), tuple(minor_axis[1]), BLUE)

        box = cv2.cv.BoxPoints(props['BoundingBox'])
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, CYAN, 1)

        for p in props['Extrema']:
            cv2.circle(img, p, 5, CYAN, 1)

    cv2.imshow('image', img)

if __name__ == "__main__":
    main()
