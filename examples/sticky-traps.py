"""This is the main code used for the sticky trap project"""
# importing modules that are used. not all of them are used at the moment,
# but it is expected that they will be used in the final version.

#  import argparse
import logging
import mimetypes
import os
import sys
import common

import cv2
import numpy as np
import yaml

import imgpheno

'''
Code adapted from github.com/Naturalis/imgpheno/examples/train.py
This is a placeholder and will likely change when integrated with
web functionality for large scale testing.
'''
image_list = []


def main():
    """start of program,
    this code calls other functions in the program and first obtains the paths to the images,
    before calling the function that coordinates the analysis itself
    """
    path = "2e-ronde"
    # destination = r"./without"
    image_files = get_image_paths(path)
    for img in image_files:
        analyse_photo(img)

"""
OUTPUT_FOTO = ./output.txt
    Foto_output_file = open(OUTPUT_FOTO, "a+")
itempath=os.path.abspath(os.path.join("media/", item.get('foto')))
            insect_informatie = analyse_photo(itempath)
        # print insect_informatie
        # print insect_informatie.get("geschat_aantal_insecten")
            Foto_output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(item.get('Val_nummer'), item.get('veldnr'), item.get('datum'), insect_informatie["total_area"],
                    insect_informatie["number_of_insects"], insect_informatie["smaller_than_4"], insect_informatie["between_4_and_10"],
                    insect_informatie["larger_than_10"]))
"""


def analyse_photo(img):
    """This function is called by the sticky traps website.
    It takes a path to an image file, and uses this to call the other functions to perform the main analysis.
    """
    try:
        img = r"%s"%(str(img)) # this step is required to eliminate some bugs where special characters are present in the image path.
    except:
        return None
    contours, trap = find_insects(img)  # this function takes the path of the image and returns the contours of the insects found.
    if contours == None:
        return None
    output = run_analysis(contours, img)  # this function takes the contours and the image of the trap and then gives the relevant data
    return output



def find_insects(img_file):
        """Call all functions in order to analyse the image."""
        img = read_img(img_file)  # the image that the path is pointing to is read and returned as a numpy array
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # converts to HSV colourspace for trap detection
        """
        note that when displaying hsv as an image using cv2.imshow
        the colours are distorted since imshow assumes bgr colourspace.
        h value of yellow is 30 here.
        """
        mask = hsv_threshold(hsv)  # calls the function that detects the trap based on the HSV image, this is a type of thresholding operation
        corners = imgpheno.find_corners(mask)  # finds the four corners based on an approximation of the contour of the mask.
        yml = open_yaml('sticky_traps.yml')  # the settings file is opened and the relevant information is used to calculate the size of the target image to retain standard dimentions for the rest of the analysis
        width = 4*yml.trap_dimensions.Trap_width
        height = 4*yml.trap_dimensions.Trap_height
        # this height and width must be easily adjustable.
        points_new = np.array([[0, 0], [width, 0], [0, height], [width, height]], np.float32)
        trap = imgpheno.perspective_transform(img, corners, points_new)  # resizes the image based on the previous calculations.
        if trap is None:  # This code shows the corners returned by find_corners, in case not exactly 4 were returned.
            # show_corners(corners, img, img_file)
	    return None, trap
        # after this the program needs to find the insects present on the trap.

        """
        trap = cv2.bilateralFilter(trap, 50, 60, 100) #This eliminates fine texture from the image

        the above line of code reduces the amount of false positives in the current test images (29th of March),
        However it also multiplies the computing time of each photo by a factor of at least 10.
        these false positives however seem to arise due to the texture of the folders in which the traps were photographed.
        I will not execute this line in the rest of the creation of the program, but i will test the difference with and without
        when the actual fieldwork has taken place.
        """
        if yml.edges_to_crop:  # checks if there are any edges that need to be cropped by checking the settings file.
            trap = crop_image(trap)  # function here that actually crops the image.
        r_channel = trap[:, :, 2]  # selects the channel with the highest contrast
        #image_list.append(trap)  # displays the image at the end
        contours = find_contours(r_channel)  # this function finds the insects on the red channel of the resized image.

        """the next three commented lines can be used to visualize the areas marked as insects by the program"""
        #contour_img = trap.copy()
        #cv2.drawContours(contour_img, contours, -1, [0, 0, 255], -1)
        #image_list.append(contour_img)

        return contours, trap  # returning the contours of the areas marked as insects and the image of the resized trap for further analysis.


def run_analysis(contours, filename):
    """
    does an analysis on the contours found, and returns relevant data
    Currently it returns the total area of the trap covered in insects,
    the amount of insects estimated to be smaller than 4 mm, the amount of insects betweeen 4 and 10 mm, the amount of insects larger than both these catigories,
    and the total number of insects estimated to be on the trap.
    """

    properties = imgpheno.contour_properties(contours, ('Area',))  # get the mentioned properties, the properties are a list of dictionaries
    # the following functions calculate the number of insects in each size category, >4, >4 & <10, and >10, all in mm.
    # one mm in the foto however is 4 pixels.
    # since we make the estimation of the size based on the area we also have to square the amount of pixels:
    # for example: an insect 4mm long is 16 pixels long in the image. it is also likely 16 pixels wide if we assume that the wingspan and the body length are roughly equal.
    # the area the insect then occupies is 16x16 pixels for a total of 256 pixels.
    # this function only gives a very rough estimate of size, however accurate estimations are quite complicated.
    areas = [i['Area'] for i in properties]

    # the next lines split the areas into three categories based on the area measured in pixels.
    smaller_than_4 = [i for i in areas if i < 256]
    between_4_and_10 = [i for i in areas if i >= 256 and i < 1600]
    larger_than_10 = [i for i in areas if i >= 1600]

    # print areas
    total_area = np.sum(areas)
    number_of_insects = len(contours)  # this line finds the number of seperate areas, wich is the estimated number of insects.
    # next line provides the output in the form of a dictionary.
    output = {"total_area": total_area/4, "smaller_than_4": len(smaller_than_4), "between_4_and_10": len(between_4_and_10),
              "larger_than_10": len(larger_than_10), "number_of_insects": number_of_insects}
    return output


def get_image_paths(path):
    """Return a list of all images present in the directory 'path'."""
    if not os.path.exists(path):  # checks if the given path actually exists, and gives an error otherwise.
        logging.error("Cannot open %s (No such file or directory)", path)
        return 1

    images = []  # this list will be populated by the relative paths to all images found in the given directory

    for item in os.listdir(path):  # iterates over all items found in the directory 'path'
        imgpath = os.path.join(path, item)  # get the full path to the images and check to see if they are actually files.
        if os.path.isfile(imgpath):
            mime = mimetypes.guess_type(imgpath)[0]
            if mime and mime.startswith('image'):  # checking if the file is an actual image
                images.append(imgpath)  # when the file is determined to be an image, it is added to the list of images that will be returned

    if len(images) == 0:  # giving a message if no images were found in the folder.
        logging.error("No images found in %s", path)
        return 1
    return images


def read_img(path):
    """
    Read the images into an array generated by opencv2.
    the image is also resized if it is to large
    """
    if not os.path.isfile(path):  # double checking that the image file does exist, although previous steps should only provide working filepaths.
        print "the file does not exist"
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # loading the image using the cv2 imread function. the image is now represented as a three dimentional matrix, using the rgb colourspace.
    # for more information see the open cv documentation.
    perim = sum(img.shape[:2])  # setting the maximum size on the image and rescaling to a smaller size when needed.
    # the resulting image has a perimeter ~ 2x the size of the final cropped image, this ensures the results don't suffer from this step.
    if perim > 100000:
        ref = float(100000) / perim
        img = cv2.resize(img, None, fx=ref, fy=ref)
    return img  # the image is now returned to the rest of the program for further analysis.



def hsv_threshold(img):
    """The corner detection did not work, I switched to a contour
    finding algorithm. this return the outer contour,
    this will be the sticky trap. the next step will be to find the corners
    using the contour.
    This code as of now only supports the colour yellow, switching to a different colour might make the program not detect the trap.
    """
    lower_yellow = np.array([15, 100, 100])  # these two lines set the upper and lower boundaries of the pixel values assumed to belong to the trap.
    upper_yellow = np.array([45, 255, 255])  # if all values from a given pixel are within the ranges given here, then it is added to the mask that will be returned by this function
    mask = cv2.inRange(img, lower_yellow, upper_yellow)  # creation the mask itself.
    return mask


# possible candidate, is only a small wrapper though, could almost be accomplished with one line of code.
def find_contours(image):
    """
    This function returns all contours found in an image using find contours following adaptive thresholding
    """
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 10)
    """The above line uses inverted thresholding on the red colourchannel to detect the insects on the trap and to add all pixels belonging to the mask returned by this function.
    It uses an adaptable thresholding operation. "The threshold value is the weighted sum of the neighbourhood values, where the weights are a gaussian window." as given by the CV2 documentation.
    this step simultaneously selects the insects and reduces/removes a lot of noise present in the image, like uneven lighting and artefacts created by transparant folders used to store the traps.
    the gaussian window is quite large, but this is necessary to properly select all insects. if the gaussian window is smaller then the centre of the insects will not be selected properly because
    the surrounding enviroment is the same intensity and the pixels there are then considered to be part of the trap itself.
    """
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  #finds the contours in the binary mask of the thresholded image.
    return contours


def crop_image(img):
    """ Crops the image 'img' according to the specifications found in the settings file"""
    short_edge = yml.cropping_width.along_short_edges*4  # calculates the amount of pixels that need to be cropped from the image on the short end of the trap
    long_edge = yml.cropping_width.along_long_edges*4  # same as above, but now with the long edge of the image
    width, height = img.shape[0:2]  # gives the height and width of the trap in order to propely calculate the exact pixels to crop
    roi = img[short_edge: width-short_edge, long_edge: height-long_edge]  # selects the portion of the image that is going to be analysed in later steps.
    return roi
# [width, height]

def show_corners(corners, img, img_file):
    "shows the corners found on the image."
    for i in corners:
        cv2.circle(img, tuple(i), 5, [0, 0, 255], -1)  # this draws the corners in the image
    msg = "corners found in " + str(img_file)
    cv2.imshow(msg, img)  # these lines show the image and wait until a key is pressed before the window is closed again.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images():
    """Shows all images made during the running of the program in order to easily check the results of the program.
    Part of the code copied from OpenCv2 tutorial.
    """
    if len(image_list) == 0:  #if there are no images to display, there is no need to run the function.
        return
    cv2.namedWindow('image')  # create the window in wich the images are displayed
    i = 0  # set the iterator value to zero, keeps track of the image beeing displayed.
    cv2.imshow("image", image_list[i])  # show the first image.
    while True:
        k = cv2.waitKey(0) & 0xFF  # have the program wait for instructions indefinitly.

        if k == ord('n'):  # if the n key is pressed, go to the next photo in the list
            i += 1
            if i >= len(image_list):  # if the end of the list is reached, the first photo is displayed instead.
                i = 0
            cv2.imshow("image", image_list[i])  # show the selected image again.
        elif k == ord('p'):  # go to the previous photo when p is pressed. almost identical to code above
            i -= 1  # move the indicator value back one place
            if i < 0:  # if the function scrolled before the first photo, then display the last image in the list
                i = len(image_list) - 1  # set the iterator value
            cv2.imshow("image", image_list[i])  # display the image
        elif k == ord('q'):  # if q is pressed, then close the window and end the function
            break

    cv2.destroyAllWindows()  # close the window created at the start of the function.


def write_images(destination, images):
    """
    this function takes two arguments: an already existing path where the images are saved,
    and a list of images in the Mat format.
    the function then saves them, naming them according to their place in the image list.
    used in testing the program.
    """
    for i in range(len(images)):
        name = r"%s/Image_%s.jpg" %(destination, (i+1))
        cv2.imwrite(name, image_list[i])

def open_yaml(path):  # funtion used to access the settings contained in the YAML file contained in the path.
    if not os.path.isfile(path):  # return an error if the specified file does not exist.
        logging.error("Cannot open %s (no such file)" % path)
        return None

    f = open(path, 'r')  # opening the file as read only to avoid any overwriting of the file.
    yml = yaml.load(f)
    yml = common.DictObject(yml)
    f.close()

    return yml


yml = open_yaml('sticky_traps.yml')



if __name__ == "__main__":
    main()
    show_images()
