#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import features as ft

INTERSECT_DIST_MAX = 30
BLACK = (0,0,0)
GRAY = (105,105,105)
BLUE = (255,0,0)
CYAN = (255,255,0)
GREEN = (0,255,0)
RED = (0,0,255)

img_src = None
img = None
no_image = True
angle_sym = 0
angle_shape = 0
radius = 0
landmarks = None

def main():
    global args

    if sys.flags.debug:
        # Print debug messages if the -d flag is set for the Python interpreter.
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        # Otherwise just show log messages of type INFO.
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Aprse arguments
    parser = argparse.ArgumentParser(description='Get the rough shape from the main object')
    parser.add_argument('--path', metavar='PATH', required=False, help='Path to image folder')
    parser.add_argument('--maxdim', metavar='N', type=float, default=500.0, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is 500.")
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
    cv2.createTrackbar('Symmetry Angle', 'image', 0, 180, set_angle_symmetry)
    cv2.createTrackbar('Shape Angle', 'image', 0, 180, set_angle_shape)
    cv2.createTrackbar('Radius', 'image', 0, 400, set_radius)
    cv2.setTrackbarPos('Radius', 'image', 200)

    i = 0
    while True:
        if args.path:
            # Load images from path if PATH is given.
            process_image(args, images[i])
        else:
            # Otherwise just draw the angle on a black background.
            draw_axis()

        k = cv2.waitKey(0) & 0xFF

        if k == ord('n'):
            i += 1
            if i >= len(images):
                i = 0
        elif k == ord('p'):
            i -= 1
            if i < 0:
                i = len(images) - 1
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()

    return 0

def set_angle_symmetry(x):
    global angle_sym
    angle_sym = x
    draw_axis()

def set_angle_shape(x):
    global angle_shape
    draw_angle_shape(x)

def set_radius(x):
    global radius
    radius = x
    draw_axis()

def process_image(args, path):
    global img, img_src, no_image, landmarks

    no_image = False

    img = cv2.imread(path)
    if img == None or img.size == 0:
        logging.info("Failed to read %s" % path)
        return

    logging.info("Processing %s..." % path)

    # Resize the image if it is larger then the threshold.
    max_px = max(img.shape[:2])
    if args.maxdim and max_px > args.maxdim:
        rf = float(args.maxdim) / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)
    img_src = img.copy()

    # Perform segmentation
    mask = ft.segment(img, args.iters, args.margin)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Get landmarks
    landmarks = ft.shape360(img, bin_mask)
    contour, center, defects, start = landmarks

    # Fit an ellipse (and draw it)
    if len(contour) >= 6:
        box = cv2.fitEllipse(contour)
        #cv2.ellipse(img, box, GREEN)
        angle_sym = box[2]

        # This indirectly calls set_angle_symmetry(angle_sym)
        cv2.setTrackbarPos('Symmetry Angle', 'image', int(angle_sym))

    draw_axis()

def draw_axis():
    """Draw horizontal, vertical, and symmetry axis."""
    global angle_sym, img, no_image, radius, landmarks

    if no_image:
        img = np.zeros((500, 500, 3), np.uint8)
        center = (250,250)
    else:
        img = img_src.copy()

        # Draw landmarks
        contour, center, defects, start = landmarks

        for d, point in defects:
            if d < 10:
                break
            cv2.circle(img, point, 5, BLUE, -1)
        cv2.circle(img, center, 5, GREEN, -1)
        cv2.circle(img, start, 5, CYAN, -1)

    # Draw x and y axis
    cv2.line(img, (0, center[1]), (img.shape[1], center[1]), BLACK)
    cv2.line(img, (center[0], 0), (center[0], img.shape[0]), BLACK)

    # Draw the symmetry angle.
    angle_line = ft.angled_line(center, angle_sym, radius)
    if angle_sym > 90:
        color = BLUE
    else:
        color = RED
    cv2.line(img, angle_line[0], angle_line[1], color)

    cv2.imshow('image', img)

def draw_angle_shape(angle):
    global angle_sym, img, radius, landmarks

    draw_axis()
    contour, center, defects, start = landmarks

    # Shift the angle by the object angle.
    angle += angle_sym

    # Draw the angle.
    angle_line = ft.angled_line(center, angle, radius)
    cv2.line(img, angle_line[0], angle_line[1], GREEN)

    # Get the slope for the linear function.
    slope = ft.slope_from_angle(angle, inverse=True)
    logging.info("Slope for angle %d is %f" % (angle, slope))

    # Find all contour points that somewhat fit the linear function.
    weighted_points = []
    for p in contour:
        p = np.array(p[0])
        p_norm = p - center
        if abs(slope) == float("inf"):
            if p_norm[0] == 0:
                weighted_points.append((0.0, tuple(p)))
        else:
            # Only save points for which the distance to the expected point
            # is less than or equal to the threshold. Save the points with
            # a weight value.
            y = slope * p_norm[0]
            p_exp = (p_norm[0], y)
            d = ft.point_dist(p_norm, p_exp)

            # The threshold depends on the slope of the linear function.
            threshold = math.ceil(abs(slope))
            if d <= threshold:
                w = 1 / (d+1)
                weighted_points.append((w, tuple(p)))

    assert len(weighted_points) != 0, "No intersections found"

    # Cluster the points.
    weighted_points = ft.weighted_points_nearest(weighted_points, t=8)
    weights, points = zip(*weighted_points)
    points = np.array(points, dtype=np.float32)

    # Draw main intersections.
    for x,y in points:
        cv2.circle(img, (x,y), 5, RED)

    cv2.imshow('image', img)

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
