"""This is the main code used for the sticky trap project"""
#importing modules that are used. not all of them are used at the moment,
#but it is expected that they will be used in the final version.

#import argparse
import logging
import mimetypes
import os
#import sys

import cv2
import numpy as np

import imgpheno

#import random this is for the drawing of contours with random colours
'''
this code would not cooperate and is not neccesary at the moment

import common
'''

'''
Code adapted from github.com/Naturalis/imgpheno/examples/train.py
This is a placeholder and will likely change when integrated with
web functionality for large scale testing.
'''
image_list = []

def main():
    """start of program,
    creates parser to obtain the path to images to analyse.
    This code is a placehoder as of yet to accelerate the process of testing. this will change in the final version.
    """
# TODO: Change the code that gets the path to the images to either be fully automatic
    path = "images/sticky-traps"
    #destination = r"./without"
    image_files = get_images(path)
    print image_files


    for img in image_files:
        contours, trap = find_insects(img)
        run_analysis(contours, img)

        ellipse_img = trap.copy()
        for i in contours:
            if len(i) >= 5:
                ellipse = cv2.fitEllipse(i)
                cv2.ellipse(ellipse_img, ellipse, (0, 0, 255), 1)
        image_list.append(ellipse_img)


def run_analysis(contours, filename):
    """
    does an analysis on the contours found, and returns relevant data
    """

    # TODO: have this function automaticially make a file ready for further analysis with R.

    properties = imgpheno.contour_properties(contours, ('Area', 'MajorAxisLength',))
    major_axes = [i['MajorAxisLength'] for i in properties]
    smaller_than_4 = [i for i in major_axes if i < 12]
    between_4_and_10 = [i for i in major_axes if i >= 12 and i < 40]
    larger_than_10 = [i for i in major_axes if i >= 40]

    areas = [i['Area'] for i in properties]
    average_area = np.mean(areas)
    number_of_insects = len(contours)
    print """
There are %s insects on the trap in %s.
The average area of the insects in %s is %d mm square.
The number of insects smaller than 4mm is %s
The number of insects between 4 and 10 mm is %s
the number of insects larger than 10mm is %s
""" %(number_of_insects, filename, filename, (average_area/4), len(smaller_than_4),
        len(between_4_and_10), len(larger_than_10))


def find_insects(img_file):
    "calls all functions in order to analyse the image."
    img = read_img(img_file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #converts to HSV colourspace for trap roi selection
    """
    note that when displaying hsv as an image using cv2.imshow
    the colours are distorted since imshow assumes bgr colourspace.
    h value of yellow is 30 here.
    """
    mask = trap_detection(hsv) #calls the function that detects the trap based on the HSV image
    corners = corner_selection(mask) #finds the four corners based on an approximation of the contour of the mask.
    trap = perspective_transform(corners, img) #resizes the image.
    if trap is None: # This code shows the corners returned by corner_selection, in case not exactly 4 were returned.
        show_corners(corners, img, img_file)
    #after this the program needs to find the insects present on the trap.

    """
    trap = cv2.bilateralFilter(trap, 50, 60, 100) #This eliminates fine texture from the image

    the above line of code reduces the amount of false positives in the current test images (29th of March),
    However it also multiplies the computing time of each photo by a factor of at least 10.
    these false positives however seem to arise due to the texture of the folders in which the traps were photographed.
    I will not execute this line in the rest of the creation of the program, but i will test the difference with and without
    when the actual fieldwork has taken place.
    """
    r_channel = trap[:, :, 2] #selects the channel with the highest contrast
    image_list.append(trap) #displays the image at the end
    contours = find_contours(r_channel)

    contour_img = trap.copy()
    cv2.drawContours(contour_img, contours, -1, [0, 0, 255], -1)
    image_list.append(contour_img)

    return contours, trap


def get_images(path):
    """returns a list of all images present in the directory 'path'"""
    if not os.path.exists(path):
        logging.error("Cannot open %s (No such file or directory)", path)
        return 1

    images = []

    for item in os.listdir(path):
        imgpath = os.path.join(path, item)
        if os.path.isfile(imgpath):
            mime = mimetypes.guess_type(imgpath)[0]
            if mime and mime.startswith('image'):
                images.append(imgpath)

    if len(images) == 0:
        logging.error("No images found in %s", path)
        return 1
    return images


def read_img(path):
    """This function reads in the images into an array generated by opencv2.
    the image is also resized if it is to large
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    perim = sum(img.shape[:2])
    if perim > 1000:
        ref = float(1000) / perim
        img = cv2.resize(img, None, fx=ref, fy=ref)
    return img

def trap_detection(img):
    """The corner detection did not work, I switched to a contour
    finding algorithm. this return the outer contour,
    this will be the sticky trap. the next step will be to find the corners
    using the contour"""
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(img, lower_yellow, upper_yellow)
    return mask


def corner_selection(binary):
    """
    this funtion returns the corners as found by the approximate_contour() function.
    These points are sorted according to the distance to the leftmost point.
    """
    contour = imgpheno.get_largest_contour(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx = approximate_contour(contour, 0.05)
    coords = approx[:, 0]
    #coords now is a list of the coordinates of the corners, but these coordinates are not in a standard order.
    sorted_points = sort_clockwise(coords)
    return sorted_points


def perspective_transform(corners, img):
    """
    resizes the image by performing a perspective transform. the destination is
    fixed to ensure constant results.
    the traps used in the fieldwork in april have dimentions of 198 by 147 mm.
    """
    width = 588
    height = 792
    points_new = np.array([[0, 0], [width, 0], [0, height], [width, height]], np.float32)
    corners = np.array(corners, np.float32)
    if len(corners) != len(points_new):
        logging.error("Unable to find precisely 4 corners")
        return None
    matrix = cv2.getPerspectiveTransform(corners, points_new)
    sticky_trap = cv2.warpPerspective(img, matrix, (width, height))
    return sticky_trap



def sort_clockwise(coordinates):
    """
    This function needs to sort 4 coordinates according to their place on a
    rectangle. it takes the right most point, and sorts the others by their
    euclidean distance to this point.
    """
    xsorted = coordinates[np.argsort(coordinates[:, 0]), :]
    rightmost = xsorted[0, :]
    sorting = []
    for i in xsorted:
        sorting.append(imgpheno.point_dist(rightmost, i))
    points_sorted = xsorted[np.argsort(sorting), :]
    return points_sorted

def show_corners(corners, img, img_file):
    "shows the corners found on the image."
    for i in corners:
        cv2.circle(img, tuple(i), 5, [0, 0, 255], -1)
    msg = "corners found in " + str(img_file)
    cv2.imshow(msg, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images():
    "Shows all images made during the running of the program for easy lookback."
    if len(image_list) == 0:
        return
    cv2.namedWindow('image')
    i = 0
    cv2.imshow("image", image_list[i])
    while True:
        k = cv2.waitKey(0) & 0xFF

        if k == ord('n'):
            i += 1
            if i >= len(image_list):
                i = 0
            cv2.imshow("image", image_list[i])
        elif k == ord('p'):
            i -= 1
            if i < 0:
                i = len(image_list) - 1
            cv2.imshow("image", image_list[i])
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()


def write_images(destination, images):
    """
    this function takes two arguments: an already existing path where the images are saved,
    and a list of images in the Mat format.
    the function then saves them, naming them according to their place in the image list.
    """
    for i in range(len(images)):
        name = r"%s/Image_%s.jpg" %(destination, (i+1))
        cv2.imwrite(name, image_list[i])

def approximate_contour(contour, epsilon, closed=True):
    """
    returns an approximated contour based on the contour passed to it.
    """
    epsilon2 = epsilon*cv2.arcLength(contour, closed)
    approx = cv2.approxPolyDP(contour, epsilon2, closed)
    return approx

def find_contours(image):
    """
    This function returns all contours found in an image using find contours following adaptive thresholding
    """
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 10)
    #finds the contours in the mask of the thresholded image.
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours



if __name__ == "__main__":
    main()
    show_images()
