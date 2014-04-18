#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import features as ft

def main():
    parser = argparse.ArgumentParser(description='Get a simplified outline from the main object')
    parser.add_argument('image', metavar='FILE', help='Input image')
    parser.add_argument('--maxdim', metavar='N', type=float, default=500.0, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is 500.")
    parser.add_argument('--iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser.add_argument('--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    parser.add_argument('--outline-res', metavar='N', type=int, default=10, help="The resolution for the outline description. Default is 10.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img == None or img.size == 0:
        sys.stderr.write("Failed to read %s\n" % args.image)
        return 1

    sys.stderr.write("Processing %s...\n" % args.image)

    # Resize the image if it is larger then the threshold.
    max_px = max(img.shape[:2])
    if args.maxdim and max_px > args.maxdim:
        rf = args.maxdim / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)

    # Perform segmentation.
    mask = ft.segment(img, args.iters, args.margin)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Get outline.
    outline = ft.simple_outline(bin_mask, args.outline_res)
    outline = cv2.normalize(outline, None, -1, 1, cv2.NORM_MINMAX)

    print outline

    return 0

if __name__ == "__main__":
    main()
