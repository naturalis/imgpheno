#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import sys

import cv2
import numpy as np

COLOR = {
    'black':    (0,0,0),
    'gray':     (105,105,105),
    'blue':     (255,0,0),
    'cyan':     (255,255,0),
    'green':    (0,255,0),
    'red':      (0,0,255)
}

def grabcut(img, iters=5, roi=None, margin=5):
    """Wrapper for OpenCV's grabCut function.

    Runs the GrabCut algorithm for segmentation. Returns an 8-bit
    single-channel mask. Its elements may have the following values:

    * ``cv2.GC_BGD`` defines an obvious background pixel
    * ``cv2.GC_FGD`` defines an obvious foreground pixel
    * ``cv2.GC_PR_BGD`` defines a possible background pixel
    * ``cv2.GC_PR_FGD`` defines a possible foreground pixel

    The GrabCut algorithm is executed with `iters` iterations. The region
    of interest `roi` can be a 4-tuple ``(x,y,width,height)``. If the ROI
    is not set, the ROI is set to the entire image, with a margin of
    `margin` pixels from the borders.

    This method is indirectly executed by :meth:`make`.
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdmodel = np.zeros((1,65), np.float64)
    fgdmodel = np.zeros((1,65), np.float64)

    # Use the margin to set the ROI if the ROI was not provided.
    if not roi:
        roi = (margin, margin, img.shape[1]-margin*2, img.shape[0]-margin*2)

    cv2.grabCut(img, mask, roi, bgdmodel, fgdmodel, iters, cv2.GC_INIT_WITH_RECT)
    return mask

def simple(img, roi):
    """Performs simple image segmentation.

    :param img: Image object from cv2
    :param roi: ROI 4-tuple ``(x,y,width,height)``
    :return: image mask
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]] = cv2.GC_FGD
    return mask

def scale_max_perimeter(img, m):
    """Return a scaled down image based on a maximum perimeter `m`.

    The original image is returned if `m` is None or if the image is smaller.
    """
    perim = sum(img.shape[:2])
    if m and perim > m:
        rf = float(m) / perim
        img = cv2.resize(img, None, fx=rf, fy=rf)
    return img

class DictObject(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.iteritems():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [DictObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, DictObject(b) if isinstance(b, dict) else b)