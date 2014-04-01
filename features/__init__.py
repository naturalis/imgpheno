#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for feature extraction from digital images."""

import collections
import math

import numpy as np
import cv2

# Color spaces
CS_BGR = 101
CS_HSV = 102
CS_LCH = 103

# Color space ranges used in OpenCV. Each upper boundary is exclusive.
CS_RANGE = {
    CS_BGR: ([0, 256], [0, 256], [0, 256]),
    CS_HSV: ([0, 180], [0, 256], [0, 256])
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

def hists(img, histsize=None, mask=None, colorspace=CS_BGR):
    """Convenience wrapper for :cv2:`calcHist`.

    Returns the histogram for each channel in `img`.

    The input image colorspace must be set with `colorspace`. Value can be
    one of the following:
        * ``CS_BGR`` for the BGR color space.
        * ``CS_HSV`` for the HSV color space.

    Default value is ``CS_BGR``.

    .. note::

        OpenCV uses the following color space ranges:
            * BGR: B=0-255, G=0-255, R=0-255
            * HSV: H=0-179, S=0-255, V=0-255

        These ranges are also set in ``CS_RANGE``.
    """
    if histsize and len(histsize) != img.ndim:
        raise ValueError("Argument `histsize` must have `img.ndim` elements.")
    if colorspace not in CS_RANGE:
        raise ValueError("Unknown colorspace.")

    hists = []
    for ch in range(img.ndim):
        if histsize == None:
            bins = CS_RANGE[colorspace][ch][1]
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

def simple_outline(img, resolution=10):
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

def shape360(src, img):
    """Returns a shape feature from a binary image.

    Shape is returned as an array of 360 values. The returned shape is
    rotation invariant.
    """
    if len(img.shape) != 2:
        raise ValueError("Input image must be binary")

    # Obtain contours (all points) from the mask.
    contour = get_largest_countour(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get the center.
    m = cv2.moments(img, binaryImage = True)
    center = ( int(m['m10']/m['m00']) , int(m['m01']/m['m00']) )

    # Get convex hull and defects.
    hull = cv2.convexHull(contour, returnPoints = False)
    defects = cv2.convexityDefects(contour, hull)

    # Get the major defects.
    major_defects = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(contour[f][0])
        major_defects.append( (d/256.0, far) )

    # List farthes points ordered by defect size, decreasing.
    major_defects = sorted(major_defects, reverse=True)

    # Get the major defect closest to and right from the center.
    start = None
    min_dist = None
    for d, point in major_defects:
        if point[0] > center[0]:
            dist = point_dist(point, center)
            if min_dist == None or dist < min_dist:
                min_dist = dist
                start = point

    return (contour, center, major_defects, start)

def point_dist(p1, p2):
    """Return the distance between two points.

    Each point is a ``(x,y)`` tuple.
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return math.hypot(dx, dy)

def get_orientation(img):
    """Returns the orientation from a binary image.

    Source: http://stackoverflow.com/a/14720823/466781
    """
    m = cv2.moments(img, binaryImage = True)
    theta = 0.5 * math.atan( (2 * m['mu11']) / (m['mu20'] - m['mu02']) )
    theta = (theta / math.pi) * 180
    return theta

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
