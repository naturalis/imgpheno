#!/usr/bin/env python

import os
import timeit
import math
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np

import features as ft

contour = None

def grabcut_with_margin(img, iters=5, margin=5):
    """Segment image into foreground and background pixels.

    Runs the GrabCut algorithm for segmentation. Returns an 8-bit
    single-channel mask. Its elements may have one of following values:
        * ``cv2.GC_BGD`` defines an obvious background pixel.
        * ``cv2.GC_FGD`` defines an obvious foreground pixel.
        * ``cv2.GC_PR_BGD`` defines a possible background pixel.
        * ``cv2.GC_PR_FGD`` defines a possible foreground pixel.

    The GrabCut algorithm is executed with `iters` iterations. The ROI is set
    to the entire image, with a margin of `margin` pixels from the edges.
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdmodel = np.zeros((1,65), np.float64)
    fgdmodel = np.zeros((1,65), np.float64)
    rect = (margin, margin, img.shape[1]-margin*2, img.shape[0]-margin*2)
    cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, iters, cv2.GC_INIT_WITH_RECT)
    return mask

def shape_360_v1(contour, rotation=0, step=1, t=8):
    """Returns a shape feature from a contour.

    The rotation in degrees of the contour can be set with `rotation`, which
    must be a value between 0 and 179 inclusive. A rotation above 90 degrees
    is interpreted as a rotation to the left (e.g. a rotation of 120 is
    interpreted as 60 degrees to the left). The returned shape is shifted by
    the rotation. The step size for the angle can be set with `step`. Argument
    `t` is passed to :meth:`weighted_points_nearest`. The returned shape is
    rotation invariant provided that the rotation argument is set properly and
    the rotation is no more than 90 degrees left or right.

    Shape is returned as a tuple ``(intersects, center)``, where ``intersects``
    is a dict where the keys are 0 based angles and the values are lists
    of intersecting points. If `step` is set to 1, then the dict will contain
    the intersections for 360 degrees. ``center`` specifies the center of the
    contour and can be used to calculate distances from center to contour.
    """
    if len(contour) < 6:
        raise ValueError("Contour must have at least 6 points, found %d" % len(contour))
    if not 0 <= rotation <= 179:
        raise ValueError("The rotation must be between 0 and 179 inclusive, found %d" % rotation)

    # Get the center.
    center, _ = cv2.minEnclosingCircle(contour)
    center = np.int32(center)

    # If the rotation is more than 90 degrees, assume the object is rotated to
    # the left.
    if rotation <= 90:
        a = 0
        b = 180
    else:
        a = 180
        b = 0

    # Get the intersecting points with the contour for every degree
    # from the symmetry axis.
    intersects = {}
    for angle in range(0, 180, step):
        # Get the slope for the linear function of this angle and account for
        # the rotation of the object.
        slope = ft.slope_from_angle(angle + rotation, inverse=True)

        # Since pixel points can only be defined with natural numbers,
        # resulting in gaps in between two points, means the maximum
        # gap is equal to the slope of the linear function.
        gap_max = math.ceil(abs(slope))

        # Find al contour points that closely fit the angle's linear function.
        # Only save points for which the distance to the expected point is
        # no more than the maximum gap. Save each point with a weight value,
        # which is used for clustering.
        weighted_points = []
        for p in contour:
            p = p[0]
            x = p[0] - center[0]
            y = p[1] - center[1]
            if math.isinf(slope):
                if x == 0:
                    # Save points that are on the vertical axis.
                    weighted_points.append( (1, tuple(p)) )
            else:
                y_exp = slope * x
                d = abs(y - y_exp)
                if d <= gap_max:
                    w = 1 / (d+1)
                    weighted_points.append( (w, tuple(p)) )

        assert len(weighted_points) != 0, "No intersections found"

        # Cluster the points.
        weighted_points = ft.weighted_points_nearest(weighted_points, t)
        _, points = zip(*weighted_points)

        # Figure out on which side of the symmetry the points lie.
        # Create a line that separates points on the left from points on
        # the right.
        if (angle + rotation) != 0:
            division_line = ft.angled_line(center, angle + rotation + 90, 100)
        else:
            # Points cannot be separated when the division line is horizontal.
            # So make a division line that is rotated 45 degrees to the left
            # instead, so that the points are properly separated.
            division_line = ft.angled_line(center, angle + rotation - 45, 100)

        intersects[angle] = []
        intersects[angle+180] = []
        for p in points:
            side = ft.side_of_line(division_line, p)
            if side < 0:
                intersects[angle+a].append(p)
            elif side > 0:
                intersects[angle+b].append(p)
            else:
                assert side != 0, "A point cannot be on the division line"

    return (intersects, center)

def shape_360_v2(contour, rotation=0, step=1, t=8):
    """Returns a shape feature from a contour.

    The rotation in degrees of the contour can be set with `rotation`, which
    must be a value between 0 and 179 inclusive. A rotation above 90 degrees
    is interpreted as a rotation to the left (e.g. a rotation of 120 is
    interpreted as 60 degrees to the left). The returned shape is shifted by
    the rotation. The step size for the angle can be set with `step`. Argument
    `t` is passed to :meth:`weighted_points_nearest`. The returned shape is
    rotation invariant provided that the rotation argument is set properly and
    the rotation is no more than 90 degrees left or right.

    Shape is returned as a tuple ``(intersects, center)``, where ``intersects``
    is a dict where the keys are 0 based angles and the values are lists
    of intersecting points. If `step` is set to 1, then the dict will contain
    the intersections for 360 degrees. ``center`` specifies the center of the
    contour and can be used to calculate distances from center to contour.
    """
    if len(contour) < 6:
        raise ValueError("Contour must have at least 6 points, found %d" % len(contour))
    if not 0 <= rotation <= 179:
        raise ValueError("The rotation must be between 0 and 179 inclusive, found %d" % rotation)

    # Get the center.
    props = ft.contour_properties([contour], 'Centroid')
    center = props[0]['Centroid']

    # If the rotation is more than 90 degrees, assume the object is rotated to
    # the left.
    if rotation <= 90:
        a = 0
        b = 180
    else:
        a = 180
        b = 0

    # Define the slope for each point and group points by slope.
    slopes = {}
    for p in contour:
        p = tuple(p[0])
        x = p[0] - center[0]
        y = p[1] - center[1]

        if x == 0:
            s = float('inf')
        else:
            s = float(y) / x
            s = round(s, 4)

        if s in slopes:
            slopes[s].append(p)
        else:
            slopes[s] = [p]

    # Get the intersecting points with the contour for every degree from the
    # symmetry axis.
    intersects = {}
    for angle in range(0, 180, step):
        # Get the slope for the linear function of this angle and account for
        # the rotation of the object.
        slope = ft.slope_from_angle(angle + rotation, inverse=True)

        # Since pixel points can only be defined with natural numbers,
        # resulting in gaps in between two points, means the maximum
        # gap is equal to the slope of the linear function.
        gap_max = math.ceil(abs(slope))

        # Dmax set empirically.
        dmax = gap_max * 0.20

        # Make a selection of the contour points which somewhat fit the
        # slope for this angle.
        candidates = []
        for s in slopes:
            if math.isinf(slope):
                if math.isinf(s):
                    d = 0
                else:
                    continue
            else:
                d = abs(slope - s)

            if d == 0 or d <= dmax:
                for p in slopes[s]:
                    candidates.append(p)

        # Find the contour points from the list of candidate points that
        # closely fit the angle's linear function.
        # Only save points for which the distance to the expected point is
        # no more than the maximum gap. Save each point with a weight value,
        # which is used for clustering.
        weighted_points = []
        for p in candidates:
            x = p[0] - center[0]
            y = p[1] - center[1]
            if math.isinf(slope):
                if x == 0:
                    # Save points that are on the vertical axis.
                    weighted_points.append( (1, tuple(p)) )
            else:
                y_exp = slope * x
                d = abs(y - y_exp)
                if d <= gap_max:
                    w = 1 / (d+1)
                    weighted_points.append( (w, tuple(p)) )

        assert len(weighted_points) > 0, "No intersections found for angle %d" % angle

        # Cluster the points.
        weighted_points = ft.weighted_points_nearest(weighted_points, t)
        _, points = zip(*weighted_points)

        # Figure out on which side of the symmetry the points lie.
        # Create a line that separates points on the left from points on
        # the right.
        if (angle + rotation) != 0:
            division_line = ft.angled_line(center, angle + rotation + 90, 100)
        else:
            # Points cannot be separated when the division line is horizontal.
            # So make a division line that is rotated 45 degrees to the left
            # instead, so that the points are properly separated.
            division_line = ft.angled_line(center, angle + rotation - 45, 100)

        intersects[angle] = []
        intersects[angle+180] = []
        for p in points:
            side = ft.side_of_line(division_line, p)
            if side < 0:
                intersects[angle+a].append(p)
            elif side > 0:
                intersects[angle+b].append(p)
            else:
                assert side != 0, "A point cannot be on the division line"

    return (intersects, center)

def test1(contour):
    """Get pre-calculated spot distances from the local database."""
    intersects, center = shape_360_v1(contour, 0)

def test2(contour):
    """Calculate spot distances on run time."""
    intersects, center = shape_360_v2(contour, 0)

if __name__ == "__main__":
    path = "../examples/images/erycina/1.jpg"
    maxdim = 500
    runs = 2

    img = cv2.imread(path)
    if img == None or img.size == 0:
        sys.stderr.write("Cannot open %s (no such file)\n" % path)
        exit()

    max_px = max(img.shape[:2])
    if max_px > maxdim:
        rf = float(maxdim) / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)

    mask = grabcut_with_margin(img, 5, 1)
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    t = timeit.Timer("test1(contour)", "from __main__ import test1, contour")
    print "shape_360_v1: %f seconds" % (t.timeit(runs) / runs)

    t = timeit.Timer("test2(contour)", "from __main__ import test2, contour")
    print "shape_360_v2: %f seconds" % (t.timeit(runs) / runs)
