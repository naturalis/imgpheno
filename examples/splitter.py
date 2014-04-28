#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import features as ft

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Test image segmentation and splitting')
    parser.add_argument('files', metavar='FILE', nargs='+', help='Input images')
    parser.add_argument('-o', '--output', metavar='PATH', default=".", help='Path for output files.')
    parser.add_argument('-i', '--iters', metavar='N', type=int, default=5, help="The number of grabCut iterations. Default is 5.")
    parser.add_argument('-m', '--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    parser.add_argument('--maxdim', metavar='N', type=float, default=None, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is no limit.")
    parser.add_argument('--mindim', metavar='N', type=int, default=100, help="Limit the minimum dimension for input and output images. Images with a smaller width or height are ignored. Default is 100.")
    args = parser.parse_args()

    for f in args.files:
        split_image(f, args)

    sys.stderr.write("Output was saved to %s\n" % args.output)

    return 0

def split_image(path, args):
    img = cv2.imread(path)
    if img == None or img.size == 0:
        sys.stderr.write("Failed to read %s. Skipping.\n" % path)
        return -1

    logging.info("Processing %s ..." % path)

    # Resize the image if it is larger than the threshold.
    max_px = max(img.shape[:2])
    if args.maxdim and max_px > args.maxdim:
        rf = float(args.maxdim) / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)

    logging.info("Segmenting...")

    # Perform segmentation.
    mask = ft.segment(img, args.iters, args.margin)

    # Create a binary mask. Foreground is made white, background black.
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Split the image into segments.
    segments = ft.split_by_mask(img, bin_mask)

    logging.info("Exporting segments...")
    for i, im in enumerate(segments):
        if im.shape[0] < args.mindim or im.shape[1] < args.mindim:
            continue

        name = os.path.basename(path)
        name = os.path.splitext(name)
        out_path = "%s_%d%s" % (name[0], i, name[1])
        out_path = os.path.join(args.output, out_path)
        logging.info("\t%s" % out_path)
        cv2.imwrite(out_path, im)

    return 0

if __name__ == "__main__":
    main()
