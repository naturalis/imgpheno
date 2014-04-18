#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import mimetypes
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np
from pyfann import libfann
import yaml

import common
import features as ft

def main():
    if sys.flags.debug:
        # Print debug messages if the -d flag is set for the Python interpreter.
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        # Otherwise just show log messages of type INFO.
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Generate training data and train artificial neural networks.')

    # Create a sub parser for sub-commands.
    subparsers = parser.add_subparsers(help='Specify which task to start.')

    # Create an argument parser for sub-command 'traindata'.
    help_data = """Create a tab separated file with training data. Preprocessing
    steps and features to extract must be set in a YAML file. See features.yml
    for an example.
    """
    parser_data = subparsers.add_parser('data',
        help=help_data,
        description=help_data)
    parser_data.add_argument("--features", metavar="FILE", required=True, help="Path to a YAML file with feature extraction parameters.")
    parser_data.add_argument("--output", "-o", metavar="FILE", required=True, help="Output file name for training data. Any existing file with same name will be overwritten.")
    parser_data.add_argument("images", metavar="PATH", help="Directory path to load image files from. Images must be in level one sub directories. Sub directory names will be used as class names.")

    # Create an argument parser for sub-command 'trainann'.
    help_ann = """Train an artificial neural network. Optional training
    parameters can be set in a separate YAML file. See train-params.yml
    for an example file.
    """
    parser_ann = subparsers.add_parser('ann',
        help=help_ann,
        description=help_ann)
    parser_ann.add_argument("--test-data", metavar="FILE", help="Path to tab separated file with test data.")
    parser_ann.add_argument("--params", metavar="FILE", help="Path to a YAML file with ANN training parameters.")
    parser_ann.add_argument("--output", "-o", metavar="FILE", required=True, help="Output file name for the artificial neural network. Any existing file with same name will be overwritten.")
    parser_ann.add_argument("data", metavar="FILE", help="Path to tab separated file with training data.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = "Test an artificial neural network"
    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann,
        description=help_test_ann)
    parser_test_ann.add_argument("--ann", metavar="FILE", required=True, help="A trained artificial neural network")
    parser_test_ann.add_argument("--data", metavar="FILE", required=True, help="Tab separated file containing test data")

    # Create an argument parser for sub-command 'classify'.
    help_classify = "Classify an image"
    parser_classify = subparsers.add_parser('classify',
        help=help_classify,
        description=help_classify)
    parser_classify.add_argument("--ann", metavar="FILE", required=True, help="Path to a trained artificial neural network file.")
    parser_classify.add_argument("--features", metavar="FILE", required=True, help="Path to a YAML file with feature extraction parameters.")
    parser_classify.add_argument("--error", metavar="N", type=float, default=0.001, help="Error threshold. Default is 0.001")
    parser_classify.add_argument("image", metavar="FILE", help="Path to image file to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    if sys.argv[1] == 'data':
        train_data(args.images, args.features, args.output)
    elif sys.argv[1] == 'ann':
        train_ann(args.data, args.output, args.test_data, args.params)
    elif sys.argv[1] == 'test-ann':
        test_ann(args.ann, args.data)
    elif sys.argv[1] == 'classify':
        classify(args.image, args.ann, args.features, args.error)

    sys.exit()

def train_data(images_path, features_path, output_path):
    for path in (images_path, features_path):
        if not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    yml_file = open(features_path, 'r')
    yml = yaml.load(yml_file)
    yml = common.DictObject(yml)
    yml_file.close()

    if 'preprocess' in yml and 'segmentation' in yml.preprocess:
        path = getattr(yml.preprocess.segmentation, 'output_folder', None)
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    out_file = open(output_path, 'w')

    # Get list of image files and set the classes.
    images = {}
    classes = []
    for item in os.listdir(images_path):
        path = os.path.join(images_path, item)
        if os.path.isdir(path):
            classes.append(item)
            images[item] = get_image_files(path)

    # Make codeword for each class.
    codewords = get_codewords(classes, -1, 1)

    # Construct the header row.
    header_primer = ["ID"]
    header_data = []
    header_out = []

    if 'color_histograms' in yml.features:
        for colorspace, bins in vars(yml.features.color_histograms).iteritems():
            for ch, n in enumerate([bins] * len(colorspace)):
                for i in range(n):
                    header_data.append("%s.%d" % (colorspace[ch], i+1))

    if 'shape_outline' in yml.features:
        n = 2 * getattr(yml.features.shape_outline, 'resolution', 15)
        for i in range(n):
            header_data.append("OL.%d" % (i+1,))

    if 'shape_360' in yml.features:
        n = 360 / getattr(yml.features.shape_360, 'step', 1)
        for i in range(n):
            header_data.append("MEAN.%d" % i)
            header_data.append("SD.%d" % i)

    for i in range(len(classes)):
        header_out.append("OUT.%d" % (i+1,))

    # Write the header row.
    out_file.write( "%s\n" % "\t".join(header_primer + header_data + header_out) )

    # Set the training data.
    training_data = common.TrainData(len(header_data), len(classes))
    fp = Fingerprint()
    for im_class, files in images.items():
        for im_path in files:
            if fp.open(im_path, yml) == None:
                logging.info("Failed to read %s. Skipping." % im_path)
                continue

            data = fp.make()

            assert len(data) == len(header_data), "Data length mismatch"

            training_data.append(data, codewords[im_class], label=im_path)

    training_data.finalize()

    # Round all values.
    training_data.round_input(4)

    # Write data rows.
    for label, input_data, output_data in training_data:
        row = []
        row.append( label )
        row.extend( input_data.astype(str) )
        row.extend( output_data.astype(str) )
        out_file.write( "%s\n" % "\t".join(row) )

    out_file.close()
    logging.info("Training data written to %s" % output_path)

def get_image_files(path):
    fl = []
    for item in os.listdir(path):
        im_path = os.path.join(path, item)
        if os.path.isdir(im_path):
            fl.extend( get_image_files(im_path) )
        elif os.path.isfile(im_path):
            mime = mimetypes.guess_type(im_path)[0]
            if mime and mime.startswith('image'):
                fl.append(im_path)
    return fl

def get_codewords(classes, neg=-1, pos=1):
    """Returns codewords for a list of classes."""
    n =  len(classes)
    codewords = {}
    for i, cls in enumerate(sorted(classes)):
        cw = [neg] * n
        cw[i] = pos
        codewords[cls] = cw
    return codewords

def train_ann(train_data_path, output_path, test_data_path=None, ann_params_path=None):
    for path in (train_data_path, test_data_path, ann_params_path):
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    ann_trainer = common.TrainANN()
    if ann_params_path:
        yml_file = open(ann_params_path, 'r')
        yml = yaml.load(yml_file)
        yml = common.DictObject(yml)
        yml_file.close()

        if 'ann' in yml:
            ann_trainer.connection_rate = getattr(yml.ann, 'connection_rate', 1)
            ann_trainer.hidden_layers = getattr(yml.ann, 'hidden_layers', 1)
            ann_trainer.hidden_neurons = getattr(yml.ann, 'hidden_neurons', 8)
            ann_trainer.learning_rate = getattr(yml.ann, 'learning_rate', 0.7)
            ann_trainer.epochs = getattr(yml.ann, 'epochs', 500000)
            ann_trainer.desired_error = getattr(yml.ann, 'error', 0.001)

    train_data = common.TrainData()
    train_data.read_from_file(train_data_path)

    ann = ann_trainer.train(train_data)
    ann.save(output_path)
    logging.info("Artificial neural network saved to %s" % output_path)

    logging.info("Testing the neural network...")
    error = ann_trainer.test(train_data)
    logging.info("Mean Square Error on training data: %f" % error)

    if test_data_path:
        test_data = common.TrainData()
        test_data.read_from_file(test_data_path)
        error = ann_trainer.test(test_data)
        logging.info("Mean Square Error on test data: %f" % error)

def test_ann(ann_path, test_data_path):
    for path in (ann_path, test_data_path):
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    ann = libfann.neural_net()
    ann.create_from_file(ann_path)

    test_data = common.TrainData()
    try:
        test_data.read_from_file(test_data_path)
    except ValueError as e:
        sys.stderr.write("Failed to process the test data: %s\n" % e)
        exit(1)

    logging.info("Testing the neural network...")
    fann_test_data = libfann.training_data()
    fann_test_data.set_train_data(test_data.get_input(), test_data.get_output())

    ann.test_data(fann_test_data)

    error = ann.get_MSE()
    logging.info("Mean Square Error on test data: %f" % error)

def classify(image_path, ann_path, features_path, error):
    for path in (image_path, ann_path, features_path):
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    yml_file = open(features_path, 'r')
    yml = yaml.load(yml_file)
    yml = common.DictObject(yml)
    yml_file.close()

    if 'classes' not in yml:
        sys.stderr.write("Classes are not set in the YAML file. Missing object 'classes'.")
        return 1

    # Load the ANN.
    ann = libfann.neural_net()
    ann.create_from_file(ann_path)

    # Get features from image.
    fp = Fingerprint()
    if fp.open(image_path, yml) == None:
        sys.stderr.write("Failed to read %s\n" % image_path)
        return 1
    features = fp.make()

    # Classify the image.
    codeword = ann.run(features)

    # Get codeword for each class.
    codewords = get_codewords(yml.classes)

    # Get the classification.
    classification = get_classification(codewords, codeword, error)

    logging.info("Codeword: %s" % codeword)
    logging.info("Classification: %s" % ", ".join(classification))

def get_classification(codewords, codeword, error=0.001):
    classes = []
    for cls, cw in codewords.items():
        for i, code in enumerate(cw):
            if code == 1.0 and (code - codeword[i])**2 < error:
                classes.append((codeword[i], cls))
    classes = [x[1] for x in sorted(classes, reverse=True)]
    return classes

class Fingerprint(object):

    def __init__(self):
        self.path = None
        self.params = None
        self.img = None
        self.mask = None
        self.bin_mask = None

    def open(self, path, params):
        self.img = cv2.imread(path)
        self.params = params
        if self.img == None or self.img.size == 0:
            return None
        self.path = path
        self.mask = None
        self.bin_mask = None
        return self.img

    def _preprocess(self):
        if self.img == None:
            raise ValueError("No image loaded")

        if 'preprocess' not in self.params:
            return

        # Resize the image if it is larger then the threshold.
        max_px = max(self.img.shape[:2])
        maxdim = getattr(self.params.preprocess, 'maximum_dimension', None)
        if maxdim and max_px > maxdim:
            logging.info("Scaling down...")
            rf = float(maxdim) / max_px
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Perform segmentation.
        segmentation = getattr(self.params.preprocess, 'segmentation', None)
        if segmentation:
            logging.info("Segmenting...")
            iterations = getattr(segmentation, 'iterations', 5)
            margin = getattr(segmentation, 'margin', 1)
            output_folder = getattr(segmentation, 'output_folder', None)

            self.mask = ft.segment(self.img, iterations, margin)
            self.bin_mask = np.where((self.mask==cv2.GC_FGD) + (self.mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

            if output_folder and os.path.isdir(output_folder):
                # Merge the binary mask with the image.
                img_masked = cv2.bitwise_and(self.img, self.img, mask=self.bin_mask)

                # Save the masked image to the output folder.
                fname_base = os.path.basename(self.path)
                fname_parts = os.path.splitext(fname_base)
                fname = "%s.png" % (fname_parts[0])
                out_path = os.path.join(output_folder, fname)
                cv2.imwrite(out_path, img_masked)

    def make(self):
        if self.img == None:
            raise ValueError("No image loaded")

        logging.info("Processing %s..." % self.path)

        self._preprocess()

        logging.info("Extracting features...")

        data_row = []

        if not 'features' in self.params:
            raise ValueError("Nothing to do. Features to extract not set.")

        if 'color_histograms' in self.params.features:
            logging.info("- Running color:histograms...")
            data = self.get_color_histograms()
            data_row.extend(data)

        if 'shape_outline' in self.params.features:
            logging.info("- Running shape:outline...")
            data = self.get_shape_outline()
            data_row.extend(data)

        if 'shape_360' in self.params.features:
            logging.info("- Running shape:360...")
            data = self.get_shape_360()
            data_row.extend(data)

        return data_row

    def get_color_histograms(self):
        if self.bin_mask == None:
            raise ValueError("Binary mask not set")

        histograms = []
        for colorspace, bins in vars(self.params.features.color_histograms).iteritems():
            bins = [int(bins)] * len(colorspace)

            if colorspace.lower() == "bgr":
                colorspace = ft.CS_BGR
                img = self.img
            elif colorspace.lower() == "hsv":
                colorspace = ft.CS_HSV
                img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            elif colorspace.lower() == "luv":
                colorspace = ft.CS_LUV
                img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LUV)
            else:
                raise ValueError("Unknown colorspace")

            hists = ft.color_histograms(img, bins, self.bin_mask, colorspace)

            for hist in hists:
                hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
                histograms.extend( hist.ravel() )
        return histograms

    def get_shape_outline(self):
        if self.bin_mask == None:
            raise ValueError("Binary mask not set")

        resolution = getattr(self.params.features.shape_outline, 'resolution', 15)

        outline = ft.shape_outline(self.bin_mask, resolution)
        outline = cv2.normalize(outline, None, -1, 1, cv2.NORM_MINMAX)
        return outline

    def get_shape_360(self):
        if self.bin_mask == None:
            raise ValueError("Binary mask not set")

        rotation = getattr(self.params.features.shape_360, 'rotation', None)
        step = getattr(self.params.features.shape_360, 'step', 1)
        t = getattr(self.params.features.shape_360, 't', 8)
        contour = ft.get_largest_countour(self.bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # If the rotation is not set, try to fit an ellipse on the contour to
        # get the rotation angle.
        if rotation == None:
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])

        intersects, center = ft.shape_360(contour, rotation, step, t)

        # For each angle save the minimum distance from center to contour.
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

        # Normalize the data.
        means = np.array(means)
        means = cv2.normalize(means, None, -1, 1, cv2.NORM_MINMAX)
        sds = np.array(sds)
        sds = cv2.normalize(sds, None, -1, 1, cv2.NORM_MINMAX)

        return np.array( zip(means, sds) ).ravel()

if __name__ == "__main__":
    main()
