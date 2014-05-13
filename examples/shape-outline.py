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

import features as ft
from common import COLOR

img = None
img_src = None
outline_hor = None
outline_ver = None
box = None
res = None

def main():
    global res

    print __doc__

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Aprse arguments
    parser = argparse.ArgumentParser(description='Get the rough shape from the main object')
    parser.add_argument('path', metavar='PATH', help='Path to image file')
    parser.add_argument('--maxdim', metavar='N', type=float, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is no limit.")
    parser.add_argument('--iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser.add_argument('--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    parser.add_argument('--res', metavar='N', type=int, default=20, help="The resolution for the outline feature. Default is 20.")
    args = parser.parse_args()
    res = args.res

    # Create UI
    cv2.namedWindow('image')
    cv2.createTrackbar('Position', 'image', 0, args.res-1, set_position)

    process_image(args, args.path)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    return 0

def set_position(x):
    global img, outline_hor, outline_ver, res
    if img == None:
        return
    draw_outline(x, outline_hor, outline_ver, res)

def process_image(args, path):
    global img, img_src, outline_hor, outline_ver, box

    img = cv2.imread(path)
    if img == None or img.size == 0:
        logging.error("Failed to read %s" % path)
        exit(1)

    logging.info("Processing %s..." % path)

    # Resize the image if it is larger then the threshold.
    max_px = max(img.shape[:2])
    if args.maxdim and max_px > args.maxdim:
        logging.info("- Scaling image down...")
        rf = float(args.maxdim) / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)
    img_src = img.copy()

    # Perform segmentation.
    logging.info("- Segmenting...")
    mask = ft.segment(img, args.iters, args.margin)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Obtain contours (all points) from the mask.
    contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get bounding rectange of the largest contour.
    box = cv2.boundingRect(contour)

    # Get the outline.
    logging.info("- Obtaining shape...")
    outline_hor, outline_ver = ft.shape_outline(bin_mask, args.res)

    # And draw it.
    logging.info("- Done")
    draw_outline(0, outline_hor, outline_ver, args.res)

def draw_outline(i, hor, ver, res):
    global img, img_src, box

    img = img_src.copy()
    im_x, im_y, im_w, im_h = box

    # Calculate the points for the horizontal outline.
    step = float(im_w) / (res - 1)
    x = int((i * step) + im_x)
    y1, y2 = hor[i]
    p1 = (x, im_y+y1)
    p2 = (x, im_y+y2)

    # Draw the points.
    cv2.circle(img, p1, 5, COLOR['red'])
    cv2.circle(img, p2, 5, COLOR['red'])
    cv2.line(img, p1, p2, COLOR['red'])

    # Calculate the points for the vertical outline.
    step = float(im_h) / (res - 1)
    y = int((i * step) + im_y)
    x1, x2 = ver[i]
    p1 = (im_x+x1, y)
    p2 = (im_x+x2, y)

    # Draw the points.
    cv2.circle(img, p1, 5, COLOR['green'])
    cv2.circle(img, p2, 5, COLOR['green'])
    cv2.line(img, p1, p2, COLOR['green'])

    cv2.imshow('image', img)

if __name__ == "__main__":
    main()
