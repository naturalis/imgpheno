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

import common
import features as ft

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Test image segmentation and splitting')
    parser.add_argument('files', metavar='FILE', nargs='+', help='Input images')
    parser.add_argument('-o', '--output', metavar='PATH', default=".", help='Path for output files.')
    parser.add_argument('-i', '--iters', metavar='N', type=int, default=5, help="The number of grabCut iterations. Default is 5.")
    parser.add_argument('-m', '--margin', metavar='N', type=int, default=1, help="The margin of the foreground rectangle from the edges. Default is 1.")
    parser.add_argument('--max-size', metavar='N', type=float, help="Scale the input image down if its perimeter exceeds N. Default is no scaling.")
    parser.add_argument('--min-size-out', metavar='N', type=int, default=200, help="Set the minimum perimeter for output images. Smaller images are ignored. Default is 200.")
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

    # Scale the image down if its perimeter exceeds the maximum (if set).
    img = common.scale_max_perimeter(img, args.max_size)

    logging.info("Segmenting...")

    # Perform segmentation.
    mask = ft.segment(img, args.iters, args.margin)

    # Create a binary mask. Foreground is made white, background black.
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Split the image into segments.
    segments = ft.split_by_mask(img, bin_mask)

    logging.info("Exporting segments...")
    for i, im in enumerate(segments):
        if sum(im.shape[:2]) < args.min_size_out:
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
