#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for feature extraction from digital images."""

import collections
import itertools
import logging
import math

import numpy as np
import cv2

# Color spaces
CS_BGR = 101
CS_HSV = 102
CS_LUV = 103

# Color space ranges used in OpenCV. Each upper boundary is exclusive.
CS_RANGE = {
    CS_BGR: ([0, 256], [0, 256], [0, 256]),
    CS_HSV: ([0, 180], [0, 256], [0, 256]),
    CS_LUV: ([0, 101], [-134, 221], [-140, 123])
}

def segment(img, iters=5, margin=5):
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

def split_by_mask(img, mask):
    """Split an image into rectangular segments by binary mask.

    Returns an iterator object which returns each segment.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("All the input arrays must have same shape")
    if len(mask.shape) != 2:
        raise ValueError("Mask must be binary")

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for pointset in contours:
        x,y,w,h = cv2.boundingRect(pointset)
        yield img[y:y+h, x:x+w]

def color_histograms(img, histsize=None, mask=None, colorspace=CS_BGR):
    """Convenience wrapper for :cv2:`calcHist`.

    Returns the histogram for each channel in `img`.

    The input image colorspace must be set with `colorspace`. Value can be
    one of the following:
        * ``CS_BGR`` for the BGR color space.
        * ``CS_HSV`` for the HSV color space.
        * ``CS_LUV`` for the CIE 1976 (L\*, u\*, v\*) color space.

    Default value is ``CS_BGR``.

    .. note::

        OpenCV uses the following color space ranges:
            * BGR: B=0-255, G=0-255, R=0-255
            * HSV: H=0-179, S=0-255, V=0-255
            * LUV: L=0-100, U=-134-220, V=-140-122

        These ranges are also set in ``CS_RANGE``.
    """
    if histsize and len(histsize) != img.ndim:
        raise ValueError("Argument `histsize` must have `img.ndim` elements.")
    if colorspace not in CS_RANGE:
        raise ValueError("Unknown colorspace.")

    hists = []
    for ch in range(img.ndim):
        if histsize == None:
            bins = abs(CS_RANGE[colorspace][ch][1] - CS_RANGE[colorspace][ch][0] - 1)
        else:
            bins = histsize[ch]

        ranges = CS_RANGE[colorspace][ch]
        hist = cv2.calcHist([img], [ch], mask, [bins], ranges)
        hists.append(hist)

    return hists

def unrotate(src, dst, bin_mask):
    """Unrotate an image using the binary mask.
    """
    # Get the center.
    m = cv2.moments(bin_mask, True)
    center = ( int(m['m10']/m['m00']) , int(m['m01']/m['m00']) )

    # Calculate the overall shift.
    outlines = simple_outline(bin_mask)
    shift = 0
    for left,right in outlines[1]:
        shift += right - left

    return shift

    # TODO: Get the rotation in degrees.
    # Compare west side and east side areas?
    angle = None

    # TODO: Rotate a contour to get the optimal angle.
    #   For each point(x,y), multiply mat([x\\y\\1]) by rotation matrix and
    #   convert the result back to point coordinates.

    # TODO: Get rotation matrix.
    affine_transform = cv2.getRotationMatrix2D(center, angle, scale)

    # TODO: Rotate the image.
    rotated = cv2.warpAffine(img, affine_transform, dsize)

def get_largest_countour(img, mode, method):
    """Get the largest contour from a binary image.

    It is a simple wrapper for ref:`cv2.findContours`.
    """
    if len(img.shape) != 2:
        raise ValueError("Input image must be binary")

    contours, hierarchy = cv2.findContours(img, mode, method)
    if len(contours) == 1:
        return contours[0]

    largest = None
    area_max = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour, oriented=False)
        if area > area_max:
            area_max = area
            largest = contour
    return largest

def shape_outline(img, resolution=10):
    """Returns an outline feature from a binary image.

    Returns a single array where the first `resolution` elements represent
    the outline along the horizontal axis, and the last `resolution` elements
    represent the outline along the vertical axis.
    """
    if len(img.shape) != 2:
        raise ValueError("Input image must be binary")

    # Obtain contours (all points) from the mask.
    contour = get_largest_countour(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contour == None:
        return

    # Get bounding rectange of the largest contour.
    im_x, im_y, im_w, im_h = cv2.boundingRect(contour)

    # Get the pointset from the contour.
    pointset = contour[:,0]

    # Traverse contour left to right.
    outline_hor = []
    for i in range(im_w):
        if i % im_w/resolution:
            continue

        # Get all the points where X or Y is equal to `x`
        x = i + im_x
        matches = np.where(pointset == x)

        # Save only the Y values for points where X equals `x`
        yindex = matches[0][np.where(matches[1] == 0)]
        values = pointset[:,1][yindex]

        # Save the extremes, which describe the outer shape.
        outline_hor.append( (min(values), max(values)) )

    assert len(outline_hor) == resolution, "Number of shape elements must be equal to the resolution"

    # Traverse contour top to bottom.
    outline_ver = []
    for i in range(im_h):
        if i % im_h/resolution:
            continue

        # Get all the points where X or Y is equal to `y`
        y = i + im_y
        matches = np.where(pointset == y)

        # Save only the X values for points where Y equals `y`
        xindex = matches[0][np.where(matches[1] == 1)]
        values = pointset[:,0][xindex]

        # Save the extremes, which describe the outer shape.
        outline_ver.append( (min(values), max(values)) )

    assert len(outline_ver) == resolution, "Number of shape elements must be equal to the resolution"

    outline = []
    for n,s in outline_hor:
        v = (s-n)*1.0
        outline.append(v)
    for w,e in outline_ver:
        v = (e-w)*1.0
        outline.append(v)

    return np.array(outline)

def shape_360(contour, rotation=0, step=1, t=8):
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

    # Get distances from center to contour intersections for every degree
    # from the symmetry axis.
    intersects = {}
    for angle in range(0, 180, step):
        # Get the slope for the linear function of this angle and account for
        # the rotation of the object.
        slope = slope_from_angle(angle + rotation, inverse=True)

        # Find al contour points that closely fit the angle's linear function.
        # Only save points for which the distance to the expected point is
        # no more than the maximum gap. Save each point with a weight value,
        # which is used for clustering.
        weighted_points = []
        for p in contour:
            p = np.array(p[0])
            x, y = p - center
            if math.isinf(slope):
                if x == 0:
                    # Save points that are on the vertical axis.
                    weighted_points.append((0.0, tuple(p)))
            else:
                y_exp = slope * x
                d = abs(y - y_exp)

                # Since pixel points can only be defined with natural numbers,
                # resulting in gaps in between two points, means the maximum
                # gap is equal to the slope of the linear function.
                gap_max = math.ceil(abs(slope))
                if d <= gap_max:
                    w = 1 / (d+1)
                    weighted_points.append( (w, tuple(p)) )

        assert len(weighted_points) != 0, "No intersections found"

        # Cluster the points.
        weighted_points = weighted_points_nearest(weighted_points, t)
        _, points = zip(*weighted_points)

        # Figure out on which side of the symmetry the points lie.
        # Create a line that separates points on the left from points on
        # the right.
        if (angle + rotation) != 0:
            division_line = angled_line(center, angle + rotation + 90, 100)
        else:
            # Points cannot be separated when the division line is horizontal.
            # So make a division line that is rotated 45 degrees to the left
            # instead, so that the points are properly separated.
            division_line = angled_line(center, angle + rotation - 45, 100)

        intersects[angle] = []
        intersects[angle+180] = []
        for p in points:
            side = side_of_line(division_line, p)
            if side < 0:
                intersects[angle+a].append(p)
            elif side > 0:
                intersects[angle+b].append(p)
            else:
                assert side != 0, "A point cannot be on the division line"

    return (intersects, center)

def angled_line(center, angle, radius):
    """Returns an angled line.

    The `angle` must be in degrees. The line's center is set at `center` and
    the line length is twice the `radius`. The line's angle is based on the
    vertical axis.
    """
    if not isinstance(center, np.ndarray):
        center = np.array(center)

    if angle > 90:
        angle = 180 - angle
    else:
        angle *= -1
    angle = math.radians(angle)

    x = int(math.sin(angle) * radius)
    y = int(math.cos(angle) * radius)
    end = np.array((x, y))

    return (tuple(center - end), tuple(center + end))

def point_dist(p1, p2):
    """Return the distance between two points.

    Each point is a ``(x,y)`` tuple.
    """
    if not (len(p1) == 2 and len(p2) == 2):
        raise ValueError("Points must be tuples or equivalent")
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return math.hypot(dx, dy)

def side_of_line(l, p):
    """Returns on which side of line `l` point `p` lies.

    Line `l` must be a tuple of two tuples, which are the start and end
    point of the line. Point `p` is a single tuple.

    Returned value is negative, 0, or positive when the point is right,
    collinear, or left from the line, respectively. If the line is horizontal,
    then the returned value is positive.

    Source: http://stackoverflow.com/a/3461533/466781
    """
    a,b = l
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

def slope_from_angle(angle, inverse=False):
    """Returns the function slope for a given angle in degrees.

    Returns ``float("inf")`` if the slope is vertical. When the origin (0,0)
    of the matrix is in the upper left corner (as opposed to bottom left), set
    `inverse` to True.
    """
    if angle % 180 == 0:
        a = float("inf")
    else:
        a = 1 / math.tan( math.radians(angle) )
    if inverse:
        a *= -1
    return a

def shortest_distance_to_contour_point(point, contour):
    """Returns the point from `contour` that is closest to `point`.

    Result is returned as a tuple (point, distance).
    """
    mind = float("inf")
    minp = None
    for p in contour:
        p = p[0]
        d = point_dist(point, p)
        if d < mind:
            mind = d
            minp = p
    return (minp, mind)

def moments_get_center(m):
    """Returns the center from moments."""
    return np.array( (int(m['m10']/m['m00']), int(m['m01']/m['m00'])) )

def moments_get_skew(m):
    """Returns the skew from moments."""
    return m['mu11']/m['mu02']

def moments_get_orientation(m):
    """Returns the orientation in degrees from moments.

    Source: http://stackoverflow.com/a/14720823/466781
    """
    theta = 0.5 * math.atan( (2 * m['mu11']) / (m['mu20'] - m['mu02']) )
    return math.degrees(theta)

def deskew(img, dsize, mask=None):
    """Moment-based image deskew.

    Returns deskewed copy of source image `img`. If binary mask `mask` is
    provided, the skew is derived from the mask, otherwise the source image
    `img` is used, which in that case must be single-channel, 8-bit or a
    floating-point 2D array. Size of output image is set with (x,y) tuple
    `dsize`.

    Source: OpenCV examples
    """
    if mask != None:
        m = cv2.moments(mask, binaryImage=True)
    else:
        m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = moments_get_skew(m)
    affine_matrix = np.float32([[1, skew, -0.5*dsize[0]*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, affine_matrix, dsize, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def extreme_points(points):
    """Returns the two most extreme points from a point set."""
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if len(points.shape) == 2 and points.shape[1] == 2:
        shape = 1
    elif len(points.shape) == 3 and points.shape[1:2] == (1,2):
        shape = 2
    else:
        raise ValueError("Unknown shape for point set")

    maxd = 0
    extremes = None
    for p1, p2 in itertools.combinations(points, 2):
        if shape == 2:
            p1 = p1[1]
            p2 = p2[1]
        d = point_dist(p1, p2)
        if d > maxd:
            maxd = d
            extremes = (p1,p2)
    return extremes

def weighted_points_nearest(points, t=5):
    """Cluster weighted points.

    Each sample in `points` is a tuple of the format (weight, (x,y)).

    For each point combination for which the point distance is no more than
    `t` are reduced to a single point, keeping the point with the highest
    weight value.
    """
    dels = []
    for p1, p2 in itertools.combinations(points, 2):
        if p1 in dels or p2 in dels:
            continue
        d = point_dist(p1[1], p2[1])
        if d <= t:
            if p2[0] > p1[0]:
                points.remove(p1)
                dels.append(p1)
            else:
                points.remove(p2)
                dels.append(p2)
    return points

def get_major_defects(contour):
    """Returns the convexity defects of a contour sorted by severity."""
    # Get convex hull and defects.
    hull = cv2.convexHull(contour, returnPoints = False)
    defects = cv2.convexityDefects(contour, hull)

    # Get the defects and sort them decreasingly.
    major_defects = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        distance = d/256.0
        farthest_point = tuple(contour[f][0])
        major_defects.append( (distance, farthest_point) )
    return sorted(major_defects, reverse=True)

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
