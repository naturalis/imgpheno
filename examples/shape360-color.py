#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This program is a demonstration of feature extraction function shape:360.

In this example the function is used to extract the colors for transects for
given angles.

The following key bindings are available:
  N - load next image
  P - load previous image
  Q - exit
"""

import argparse
import logging
import math
import mimetypes
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import common
import features as ft

BLACK = (0,0,0)
GRAY = (105,105,105)
BLUE = (255,0,0)
CYAN = (255,255,0)
GREEN = (0,255,0)
RED = (0,0,255)

bin_mask = None
img = None
img_src = None
rotation = 0
intersects = None

def main():
    print __doc__

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Aprse arguments
    parser = argparse.ArgumentParser(description='Get the rough shape from the main object')
    parser.add_argument('path', metavar='PATH', help='Path to image folder')
    parser.add_argument('--max-size', metavar='N', type=float, help="Scale the input image down if its perimeter exceeds N. Default is no scaling.")
    parser.add_argument('--iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser.add_argument('--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    args = parser.parse_args()

    # Get list of images from the specified directory
    if args.path:
        images = get_image_files(args.path)
        if len(images) < 1:
            logging.info("No images found")
            return 0

    # Create UI
    cv2.namedWindow('image')
    cv2.createTrackbar('Angle', 'image', 0, 359, set_angle_shape)

    i = 0
    process_image(args, images[i])
    while True:
        k = cv2.waitKey(0) & 0xFF

        if k == ord('n'):
            i += 1
            if i >= len(images):
                i = 0
            process_image(args, images[i])
        elif k == ord('p'):
            i -= 1
            if i < 0:
                i = len(images) - 1
            process_image(args, images[i])
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()

    return 0

def set_angle_shape(x):
    global img

    if img == None:
        return
    draw_axis()
    draw_angle_shape(x)

def process_image(args, path):
    global bin_mask, intersects, rotation, img, img_src, center

    img = cv2.imread(path)
    if img == None or img.size == 0:
        logging.info("Failed to read %s" % path)
        return

    logging.info("Processing %s..." % path)

    # Scale the image down if its perimeter exceeds the maximum (if set).
    img = common.scale_max_perimeter(img, args.max_size)
    img_src = img.copy()

    # Perform segmentation
    logging.info("- Segmenting...")
    mask = ft.segment(img, args.iters, args.margin)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Obtain contours (all points) from the mask.
    contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Fit an ellipse on the contour to get the rotation angle.
    box = cv2.fitEllipse(contour)
    rotation = int(box[2])

    # Get the shape360 feature.
    logging.info("- Obtaining shape...")
    intersects, center = ft.shape_360(contour, rotation)

    logging.info("- Done")

    draw_axis()

def draw_axis():
    """Draw horizontal, vertical, and symmetry axis."""
    global center, img, img_src

    # Redraw the image.
    img = img_src.copy()

    cv2.imshow('image', img)

def draw_angle_shape(angle):
    global bin_mask, rotation, img, center, intersects

    # Get a line from the center to the outer intersection point.
    line = None
    if len(intersects[angle]) > 0:
        line = ft.extreme_points([center] + intersects[angle])

    # Create a mask for the line, where the line is foreground.
    line_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if line != None:
        cv2.line(line_mask, tuple(line[0]), tuple(line[1]), 255, 1)

    # Merge the binary mask with the image.
    img_masked = cv2.bitwise_and(img, img, mask=bin_mask)
    img_masked = cv2.bitwise_and(img, img_masked, mask=line_mask)

    # Draw main intersections.
    for x,y in intersects[angle]:
        cv2.circle(img, (x,y), 5, RED)

    cv2.imshow('image', img_masked)

def get_image_files(path):
    fl = []
    for item in os.listdir(path):
        im_path = os.path.join(path, item)
        if os.path.isdir(im_path):
            fl.extend( get_image_files(im_path) )
        elif os.path.isfile(im_path):
            mime = mimetypes.guess_type(im_path)[0]
            if mime and mime.startswith('image'):
                fl.append(im_path)
    return fl

if __name__ == "__main__":
    main()
