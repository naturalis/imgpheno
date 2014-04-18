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
    parser = argparse.ArgumentParser(description='Test image segmentation')
    parser.add_argument('image', metavar='FILE', help='Input image')
    parser.add_argument('-i', '--iters', metavar='N', type=int, default=5, help="The number of grabCut iterations. Default is 5.")
    parser.add_argument('-m', '--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    parser.add_argument('--maxdim', metavar='N', type=float, default=None, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is no limit.")
    parser.add_argument('--mindim', metavar='N', type=int, default=100, help="Limit the minimum dimension for input and output images. Images with a smaller width or height are ignored. Default is 100.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img == None or img.size == 0:
        sys.stderr.write("Failed to read %s\n" % args.image)
        return -1

    sys.stderr.write("Processing %s...\n" % args.image)

    # Resize the image if it is larger then the threshold.
    max_px = max(img.shape[:2])
    if args.maxdim and max_px > args.maxdim:
        rf = args.maxdim / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)

    # Perform segmentation.
    mask = ft.segment(img, args.iters, args.margin)

    # Create a binary mask. Foreground is made white, background black.
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Merge the binary mask with the image.
    img_masked = cv2.bitwise_and(img, img, mask=bin_mask)

    # Display the image in a window.
    cv2.namedWindow('image')
    cv2.imshow('image', img_masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
