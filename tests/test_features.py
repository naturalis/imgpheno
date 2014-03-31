#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import cv2

from context import features as ft

IMAGE_SLIPPER = "data/slipper.jpg"
MAXDIM = 500

class TestFeatures(unittest.TestCase):
    """Unit tests for the features module."""

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)

    def test_point_dist(self):
        """Test point distance calculation."""
        data = (
            ((0,0),         (8,9),      12.0416),
            ((1,3),         (8,9),      9.2195),
            ((-3,34),       (8,-99),    133.4541),
            ((3.5,34.1),    (8.0,99.6), 65.6544)
        )

        for p1, p2, out in data:
            self.assertEqual( round(ft.point_dist(p1, p2), 4), out )

    def test_segment_split(self):
        """Test image segmentation and splitting."""
        im_path = os.path.join(self.base_dir, IMAGE_SLIPPER)
        img = cv2.imread(im_path)
        if img == None or img.size == 0:
            raise SystemError("Failed to read %s" % im_path)

        # Resize the image if it is larger then the threshold.
        max_px = max(img.shape[:2])
        if max_px > MAXDIM:
            rf = float(MAXDIM) / max_px
            img = cv2.resize(img, None, fx=rf, fy=rf)

        # Perform segmentation.
        mask = ft.segment(img, 5, 1)

        # Create a binary mask. Foreground is made white, background black.
        bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # Split the image into segments.
        segments = ft.split_by_mask(img, bin_mask)

        # Two flowers in image should produce two segments.
        self.assertEqual( len(list(segments)), 2 )

if __name__ == '__main__':
    unittest.main()
