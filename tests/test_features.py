#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import cv2

from context import features as ft

IMAGE_SLIPPER = "../examples/images/slipper.jpg"
IMAGES_ERYCINA = ("../examples/images/erycina/1.jpg", "../examples/images/erycina/2.jpg")
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

    def test_shape_360(self):
        """Test the shape:360 feature.

        Extracts the shape from two slightly rotated versions of the same
        image. Then the medium square error between the two extracted shapes is
        calculated and checked.
        """
        shape = []
        for i, path in enumerate(IMAGES_ERYCINA):
            im_path = os.path.join(self.base_dir, path)
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

            # Obtain contours (all points) from the mask.
            contour = ft.get_largest_countour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Fit an ellipse on the contour to get the rotation angle.
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])

            # Extract the shape.
            intersects, center = ft.shape_360(contour, rotation)

            # For each angle save the mean distance from center to contour and
            # the standard deviation for the distances.
            means = []
            sds = []
            for angle in range(360):
                distances = []
                for p in intersects[angle]:
                    d = ft.point_dist(center, p)
                    distances.append(d)

                if len(distances) == 0:
                    mean = 0
                    sd = 0
                else:
                    mean = np.mean(distances, dtype=np.float32)
                    if len(distances) > 1:
                        sd = np.std(distances, ddof=1, dtype=np.float32)
                    else:
                        sd = 0

                means.append(mean)
                sds.append(sd)

            # Normalize and save result.
            means = cv2.normalize(np.array(means), None, -1, 1, cv2.NORM_MINMAX)
            sds = cv2.normalize(np.array(sds), None, -1, 1, cv2.NORM_MINMAX)
            shape.append([means, sds])

        # Check the medium square error for the means.
        diff_means = 0
        for i in range(360):
            diff_means += abs(shape[0][0][i] - shape[1][0][i])
        mse = (diff_means / 360) ** 2
        self.assertLess(mse, 0.05)

        # Check the medium square error for the standard deviations.
        diff_sds = 0
        for i in range(360):
            diff_sds += abs(shape[0][1][i] - shape[1][1][i])
        mse = (diff_sds / 360) ** 2
        self.assertLess(mse, 0.05)

if __name__ == '__main__':
    unittest.main()
