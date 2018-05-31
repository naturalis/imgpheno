"""This is the main code used for the sticky trap project"""
# importing modules that are used. not all of them are used at the moment,
# but it is expected that they will be used in the final version.

# import argparse
import logging
import mimetypes
import os
import sys
import common

import cv2
import numpy as np
import yaml

import imgpheno

import time

'''
Code adapted from github.com/Naturalis/imgpheno/examples/train.py
This is a placeholder and will likely change when integrated with
web functionality for large scale testing.
'''
image_list = []
start = time.time()


def main():
    """start of program,
    creates parser to obtain the path to images to analyse.
    This code is a placehoder as of yet to accelerate the process of testing. this will change in the final version.
    """
    # TODO: Change the code that gets the path to the images to either be fully automatic
    path = "images/sticky-traps"
    # destination = r"./without"

    if yml.result_file == "":
        pass
    else:
        open(yml.result_file, "w")
        resultfile = open(yml.result_file, "a+")
        if yml.detailed_size_classes is True:
            resultfile.write("File \t Total number of insects \t Average area \t Between 0 and 1mm \
\t Between 1 and 4mm \t Between 4 and 7mm \t Between 7 and 12mm \t Larger than 12mm \n")
            resultfile.close()
        else:
            resultfile.write(
                "File \t Total number of insects \t Average area \t Smaller \
than 4mm \t Between 4 and 10mm \t Larger than 10mm \n")
            resultfile.close()

    image_files = get_image_paths(path)
    for img in image_files:
        analyse_photo(img)


def analyse_photo(img):
    contours, trap, message = find_insects(img)
    run_analysis(contours, img, message)
    if trap is None:
        noimg = None
        image_list.append(noimg)
    else:
        ellipse_img = trap.copy()
        for i in contours:
            if len(i) >= 5:
                ellipse = cv2.fitEllipse(i)
                cv2.ellipse(ellipse_img, ellipse, (0, 0, 255), 1)
                cv2.imwrite("ellipse.jpg", ellipse_img)
        image_list.append(ellipse_img)


def run_analysis(contours, filename, message):
    """
    does an analysis on the contours found, and returns relevant data
    """
    OUTPUT_FOTO = "foto_output.txt"  # creating temporary output file.
    Foto_output_file = open(OUTPUT_FOTO, "a+")  # cleaning temporary output file.

    # TODO: have this function automaticially make a file ready for further analysis with R.
    # possibly integrated directly with the webapp.
    filename = filename.replace("images/sticky-traps\\", "").replace("images/sticky-traps/", "")
    if message != "":
        results = message
        if yml.result_file == "":
            pass
        else:
            resultfile = open(yml.result_file, "a+")
            resultfile.write(str(results))
            resultfile.close()
    else:
        properties = imgpheno.contour_properties(contours, ('Area', 'MajorAxisLength',))
        major_axes = [i['MajorAxisLength'] for i in properties]

        if yml.detailed_size_classes is True:
            b_0_1 = [i for i in major_axes if i < 4]
            b_1_4 = [i for i in major_axes if 4 <= i < 15]
            b_4_7 = [i for i in major_axes if 15 <= i < 26]
            b_7_12 = [i for i in major_axes if 26 <= i < 45]
            larger_12 = [i for i in major_axes if i >= 45]

            areas = [i['Area'] for i in properties]
            average_area = np.mean(areas)
            number_of_insects = (len(b_0_1) + len(b_1_4) + len(b_4_7) + len(b_7_12) + len(larger_12))

            print """There are %s insects on the trap in %s.
The average area of the insects in %s is %d mm square.
The number of insects between 0 and 1 mm is %s
The number of insects between 1 and 4 mm is %s
The number of insects between 4 and 7 mm is %s
The number of insects between 7 and 12 mm is %s
The number of insects larger than 12 mm is %s
        """ % (number_of_insects, filename, filename,
               (average_area / 4), len(b_0_1), len(b_1_4), len(b_4_7), len(b_7_12), len(larger_12))

            results = """%s \t %s \t %d \t %s \t %s \t %s \t %s \t %s
        """ % (filename, number_of_insects, (average_area / 4), len(b_0_1),
               len(b_1_4), len(b_4_7), len(b_7_12), len(larger_12))

            if yml.result_file == "":
                pass
            else:
                resultfile = open(yml.result_file, "a+")
                resultfile.write(str(results.replace("    ", "")))
                resultfile.close()

        else:
            smaller_than_4 = [i for i in major_axes if 4 <= i < 15]
            between_4_and_10 = [i for i in major_axes if 15 <= i < 38]
            larger_than_10 = [i for i in major_axes if 38 <= i < 45]
            # larger_than_10 = [i for i in major_axes if i >= 38]

            areas = [i['Area'] for i in properties]
            average_area = np.mean(areas)
            number_of_insects = (len(smaller_than_4) + len(between_4_and_10) + len(larger_than_10))

            print """There are %s insects on the trap in %s.
The average area of the insects in %s is %d mm square.
The number of insects smaller than 4 mm is %s
The number of insects between 4 and 10 mm is %s
The number of insects larger than 10 mm is %s
            """ % (number_of_insects, filename, filename,
                   (average_area / 4), len(smaller_than_4), len(between_4_and_10), len(larger_than_10))

            results = """%s \t %s \t %d \t %s \t %s \t %s
            """ % (filename, number_of_insects, (average_area / 4),
                   len(smaller_than_4),
                   len(between_4_and_10), len(larger_than_10))

            if yml.result_file == "":
                pass
            else:
                resultfile = open(yml.result_file, "a+")
                resultfile.write(str(results.replace("    ", "")))
                resultfile.close()


def find_insects(img_file):
    """Call all functions in order to analyse the image."""
    img = read_img(img_file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # converts to HSV colourspace for trap roi selection
    """
    note that when displaying hsv as an image using cv2.imshow
    the colours are distorted since imshow assumes bgr colourspace.
    h value of yellow is 30 here.
    """
    mask = hsv_threshold(hsv)  # calls the function that detects the trap based on the HSV image
    corners = imgpheno.find_corners(
        mask)  # finds the four corners based on an approximation of the contour of the mask.
    # TODO make a config file to edit these values more easily.
    width = 4 * yml.trap_dimensions.Trap_width
    height = 4 * yml.trap_dimensions.Trap_height
    # this height and width must be easily adjustable.
    try:
        points_new = np.array([[0, 0], [width, 0], [0, height], [width, height]], np.float32)
        trap = imgpheno.perspective_transform(img, corners, points_new)  # resizes the image.
    except:
        message = "Analyse niet mogelijk van file " + img_file.replace("images/", "") + "\n"
        trap = None
    if trap is None:  # This code shows the corners returned by find_corners, in case not exactly 4 were returned.
        show_corners(corners, img, img_file)
        contours = None
        trap = None
        message = "Analyse niet mogelijk van file " + img_file.replace("images/", "") + "\n"
    # after this the program needs to find the insects present on the trap.
    else:
        # trap = cv2.bilateralFilter(trap, 50, 60, 100)  # This eliminates fine texture from the image
        """
        the above line of code reduces the amount of false positives in the current test images (29th of March),
        However it also multiplies the computing time of each photo by a factor of at least 10.
        these false positives however seem to arise due to the texture of the folders in which the traps were photographed.
        I will not execute this line in the rest of the creation of the program, but i will test the difference with and without
        when the actual fieldwork has taken place.
        """
        if yml.edges_to_crop:
            trap = crop_image(trap)

        r_channel = trap[:, :, 2]  # selects the channel with the highest contrast
        image_list.append(trap)  # displays the image at the end
        contours = find_contours(r_channel)

        contour_img = trap.copy()
        cv2.drawContours(contour_img, contours, -1, [0, 0, 255], -1)
        image_list.append(contour_img)
        cv2.imwrite("contours.jpg", contour_img)
        message = ""

    return contours, trap, message


def get_image_paths(path):
    """Return a list of all images present in the directory 'path'."""
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
    """
    Read the images into an array generated by opencv2.
    the image is also resized if it is to large
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    perim = sum(img.shape[:2])
    if perim > 1000:
        ref = float(1000) / perim
        img = cv2.resize(img, None, fx=ref, fy=ref)
    return img


# could expand the code so it is more universal, possibly by having target and allowed deviance arguments.
def hsv_threshold(img):
    """
    The corner detection did not work, I switched to a contour
    finding algorithm. this return the outer contour,
    this will be the sticky trap. the next step will be to find the corners
    using the contour
    """

    """
    Knowing which HSV colour code to use can be calculated below 
    by giving a BGR colour code, this will return a HSV colour code. 
    Specified below is the colour blue. To specify the lower and upper HSV colour
    codes use lower = [-10, 100, 100] and upper = [+10, 255, 255] respectively.
    """
    # blue = np.uint8([[[255, 0, 0]]])
    # hsv_blue= cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    # print(hsv_blue) #This will give [120, 255, 255]

    lower = np.array(yml.trap_colours.trap_lower)
    upper = np.array(yml.trap_colours.trap_upper)
    mask = cv2.inRange(img, lower, upper)
    return mask


# possible candidate, is only a small wrapper though, could almost be accomplished with one line of code.
def find_contours(image):
    """
    This function returns all contours found in an image using find contours following adaptive thresholding
    """
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 22)
    # finds the contours in the mask of the thresholded image.
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("thresh.jpg", thresh)
    return contours


def crop_image(img):
    short_edge = yml.cropping_width.along_short_edges * 4
    long_edge = yml.cropping_width.along_long_edges * 4
    width, height = img.shape[0:2]
    roi = img[short_edge: width - short_edge, long_edge: height - long_edge]
    return roi


# [width, height]


def show_corners(corners, img, img_file):
    "shows the corners found on the image."
    for i in corners:
        cv2.circle(img, tuple(i), 5, [0, 0, 255], -1)
    msg = "corners found in " + str(img_file)


def open_yaml(path):
    if not os.path.isfile(path):
        logging.error("Cannot open %s (no such file)" % path)
        return None

    f = open(path, 'r')
    yml = yaml.load(f)
    yml = common.DictObject(yml)
    f.close()

    return yml


yml = open_yaml(r'./sticky-traps.yml')

if __name__ == "__main__":
    main()
    end = time.time()
    print(end - start)
