#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import csv
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

import features as ft

def main():
    if sys.flags.debug:
        # Print debug messages if the -d flag is set for the Python interpreter.
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        # Otherwise just show log messages of type INFO.
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Make training data')

    # Create a sub parser for sub-commands.
    subparsers = parser.add_subparsers(help='Specify which task to start.')

    # Create an argument parser for sub-command 'traindata'.
    help_data = """Create a tab separated file with training data. See the file
    train_data.yml for an example YAML file for creating training data.
    """
    parser_data = subparsers.add_parser('data',
        help=help_data,
        description=help_data)
    parser_data.add_argument('path', metavar='FILE', help="Path to a YAML file with training data parameters.")

    # Create an argument parser for sub-command 'trainann'.
    help_ann = """Train artificial neural network. See the file
    train_ann.yml for an example YAML file for creating training data.
    """
    parser_ann = subparsers.add_parser('ann',
        help=help_ann,
        description=help_ann)
    parser_ann.add_argument('path', metavar='FILE', help="Path to a YAML file with ANN training parameters.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = "Test an artificial neural network"
    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann,
        description=help_test_ann)
    parser_test_ann.add_argument("-a", "--ann", metavar="FILE", help="A trained artificial neural network")
    parser_test_ann.add_argument("-t", "--test-data", metavar="FILE", help="Tab separated file containing test data")

    # Create an argument parser for sub-command 'classify'.
    help_classify = "Classify an image"
    parser_classify = subparsers.add_parser('classify',
        help=help_classify,
        description=help_classify)
    parser_classify.add_argument("--ann", metavar="FILE", help="A trained artificial neural network")
    parser_classify.add_argument("--image", metavar="FILE", help="Image file to be classified")
    parser_classify.add_argument('--maxdim', metavar='N', type=float, default=None, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is no limit.")
    parser_classify.add_argument('--segment', action='store_const', const=True, required=False, help="Segment the image.")
    parser_classify.add_argument('--seg-iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser_classify.add_argument('--seg-margin', metavar='N', type=int, default=5, help="The margin of the foreground rectangle from the edges. Default is 5.")
    parser_classify.add_argument("--error", metavar="N", type=float, default=0.001, help="Error threshold. Default is 0.001")

    # Parse arguments.
    args = parser.parse_args()

    if sys.argv[1] == 'data':
        train_data(args)
    elif sys.argv[1] == 'ann':
        train_ann(args)
    elif sys.argv[1] == 'test-ann':
        test_ann(args)
    elif sys.argv[1] == 'classify':
        classify(args)

    sys.exit()

def train_data(args):
    if not os.path.isfile(args.path):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.path)
        return

    yml_file = open(args.path, 'r')
    yml = yaml.load(yml_file)
    yml = DictObject(yml)
    yml_file.close()

    if 'io' not in yml:
        sys.stderr.write("%s is missing the 'io' object\n" % args.path)
        return
    if 'image_path' not in yml.io:
        sys.stderr.write("%s is missing the 'io.image_path' object\n" % args.path)
        return
    if 'train_data' not in yml.io:
        sys.stderr.write("%s is missing the 'io.train_data' object\n" % args.path)
        return
    if not os.path.isdir(yml.io.image_path):
        sys.stderr.write("Cannot open %s (no such directory)\n" % yml.io.image_path)
        return
    if 'preprocess' in yml and 'segmentation' in yml.preprocess:
        output_folder = getattr(yml.preprocess.segmentation, 'output_folder')
        if not os.path.isdir(output_folder):
            sys.stderr.write("Cannot open %s (no such directory)\n" % output_folder)
            return

    out_file = open(yml.io.train_data, 'w')

    # Get list of image files and set the classes.
    images = {}
    classes = []
    for item in os.listdir(yml.io.image_path):
        path = os.path.join(yml.io.image_path, item)
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
            header_data.append("SHAPE.%d" % i)

    for i in range(len(classes)):
        header_out.append("OUT.%d" % (i+1,))

    # Write the header row.
    out_file.write( "%s\n" % "\t".join(header_primer + header_data + header_out) )

    # Set the training data.
    training_data = TrainData(len(header_data), len(classes))
    fp = Fingerprint()
    for im_class, files in images.items():
        for im_path in files:
            if fp.open(im_path, yml) == None:
                logging.info("Failed to read %s. Skipping." % im_path)
                continue

            fp.preprocess()
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
    logging.info("Training data written to %s" % yml.io.train_data)

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

def train_ann(args):
    if not os.path.isfile(args.path):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.path)
        return

    yml_file = open(args.path, 'r')
    yml = yaml.load(yml_file)
    yml = DictObject(yml)
    yml_file.close()

    if 'io' not in yml:
        sys.stderr.write("%s is missing the 'io' object\n" % args.path)
        return
    if 'train_data' not in yml.io:
        sys.stderr.write("%s is missing the 'io.train_data' object\n" % args.path)
        return
    if 'ann' not in yml.io:
        sys.stderr.write("%s is missing the 'io.ann' object\n" % args.path)
        return
    if not os.path.isfile(yml.io.train_data):
        sys.stderr.write("Cannot open %s (no such file)\n" % yml.io.train_data)
        return
    if 'test_data' in yml.io and not os.path.isfile(yml.io.test_data):
        sys.stderr.write("Cannot open %s (no such file)\n" % yml.io.test_data)
        return

    train_data = TrainData()
    train_data.read_from_file(yml.io.train_data)

    ann_trainer = TrainANN()
    ann_trainer.connection_rate = getattr(yml.ann, 'connection_rate', 1)
    ann_trainer.hidden_layers = getattr(yml.ann, 'hidden_layers', 1)
    ann_trainer.hidden_neurons = getattr(yml.ann, 'hidden_neurons', 8)
    ann_trainer.learning_rate = getattr(yml.ann, 'learning_rate', 0.7)
    ann_trainer.epochs = getattr(yml.ann, 'epochs', 500000)
    ann_trainer.desired_error = getattr(yml.ann, 'error', 0.001)

    ann = ann_trainer.train(train_data)
    ann.save(yml.io.ann)
    logging.info("Artificial neural network saved to %s" % yml.io.ann)

    logging.info("Testing the neural network...")
    error = ann_trainer.test(train_data)
    logging.info("Mean Square Error on training data: %f" % error)

    if 'test_data' in yml.io:
        test_data = TrainData()
        test_data.read_from_file(yml.io.test_data)
        error = ann_trainer.test(test_data)
        logging.info("Mean Square Error on test data: %f" % error)

def test_ann(args):
    if not os.path.isfile(args.test_data):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.test_data)
        return
    if not os.path.isfile(args.ann):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.ann)
        return

    ann = libfann.neural_net()
    ann.create_from_file(args.ann)

    test_data = TrainData()
    test_data.read_from_file(args.test_data)

    sys.stderr.write("Testing the neural network...\n")
    fann_test_data = libfann.training_data()
    fann_test_data.set_train_data(test_data.get_input(), test_data.get_output())

    ann.test_data(fann_test_data)

    error = ann.get_MSE()
    sys.stderr.write("Mean Square Error on test data: %f\n" % error)

def classify(args):
    if not os.path.isfile(args.ann):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.ann)
        return
    if not os.path.isfile(args.image):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.image)
        return

    # Load the ANN.
    ann = libfann.neural_net()
    ann.create_from_file(args.ann)

    # Get features from image.
    features = get_fingerprint(args.image, args, HIST_BINS)

    # Classify the image.
    codeword = ann.run(features)

    # Get codeword for each class.
    classes = ("Brachypetalum","Cochlopetalum","Paphiopedilum","Parvisepalum")
    codewords = get_codewords(classes)

    # Get the classification.
    classification = get_classification(codewords, codeword, args.error)

    sys.stderr.write("Codeword: %s\n" % codeword)
    sys.stderr.write("Classification: %s\n" % classification)

def get_classification(codewords, codeword, error=0.001):
    classes = []
    for cls, cw in codewords.items():
        for i, code in enumerate(cw):
            if code == 1.0 and (code - codeword[i])**2 < error:
                classes.append((codeword[i], cls))
    classes = [x[1] for x in sorted(classes, reverse=True)]
    return classes

class DictObject(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.iteritems():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [DictObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, DictObject(b) if isinstance(b, dict) else b)

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

    def preprocess(self):
        if self.img == None:
            raise ValueError("No image loaded")

        if 'preprocess' not in self.params:
            return

        logging.info("Preprocessing %s..." % self.path)

        # Resize the image if it is larger then the threshold.
        max_px = max(self.img.shape[:2])
        maxdim = getattr(self.params.preprocess, 'maximum_dimension', None)
        if maxdim and max_px > maxdim:
            logging.info("- Scaling down...")
            rf = float(maxdim) / max_px
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Perform segmentation.
        segmentation = getattr(self.params.preprocess, 'segmentation', None)
        if segmentation:
            logging.info("- Segmenting...")
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

        logging.info("Fingerprinting %s..." % self.path)

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
        shape = []
        for angle in range(360):
            mind = float("inf")
            minp = None
            for p in intersects[angle]:
                d = ft.point_dist(center, p)
                if d < mind:
                    mind = d
                    minp = p
            if mind == float("inf"):
                mind = 0
            shape.append(mind)

        # Normalize the shape.
        shape = np.array(shape)
        shape = cv2.normalize(shape, None, -1, 1, cv2.NORM_MINMAX)

        return shape

class TrainData(object):
    """Class for storing training data."""

    def __init__(self, num_input = 0, num_output = 0):
        self.num_input = num_input
        self.num_output = num_output
        self.labels = []
        self.input = []
        self.output = []
        self.counter = 0

    def read_from_file(self, path, ignore=["ID"], output_prefix="OUT"):
        fh = open(path, 'r')
        reader = csv.reader(fh, delimiter="\t")

        # Figure out the format of the data.
        header = reader.next()
        input_start = None
        output_start = None
        for i, field in enumerate(header):
            if field in ignore:
                continue
            if field.startswith(output_prefix):
                if output_start == None:
                    output_start = i
                self.num_output += 1
            else:
                if input_start == None:
                    input_start = i
                self.num_input += 1

        if self.num_input < 1 or self.num_output < 1:
            fh.close()
            raise ValueError("Incorrect format for the input file")

        input_end = input_start + self.num_input
        output_end = output_start + self.num_output

        for row in reader:
            self.labels.append(None)
            self.input.append(row[input_start:input_end])
            self.output.append(row[output_start:output_end])

        self.finalize()
        fh.close()

    def __len__(self):
        return len(self.input)

    def __iter__(self):
        return self

    def next(self):
        if self.counter >= len(self.input):
            self.counter = 0
            raise StopIteration
        else:
            self.counter += 1
            i = self.counter - 1
            return (self.labels[i], self.input[i], self.output[i])

    def append(self, input, output, label=None):
        if isinstance(self.input, np.ndarray):
            raise ValueError("Cannot add data once finalized")
        if len(input) != self.num_input:
            raise ValueError("Incorrect input array length (expected length of %d)" % self.num_input)
        if len(output) != self.num_output:
            raise ValueError("Incorrect output array length (expected length of %d)" % self.num_output)

        self.labels.append(label)
        self.input.append(input)
        self.output.append(output)

    def finalize(self):
        self.input = np.array(self.input).astype(float)
        self.output = np.array(self.output).astype(float)

    def normalize_input_columns(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")

        for col in range(self.num_input):
            tmp = cv2.normalize(self.input[:,col], None, alpha, beta, norm_type)
            self.input[:,col] = tmp[:,0]

    def normalize_input_rows(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")

        for i, row in enumerate(self.input):
            self.input[i] = cv2.normalize(row, None, alpha, beta, norm_type).reshape(-1)

    def round_input(self, decimals=4):
        self.input = np.around(self.input, decimals)

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

class TrainANN(object):
    """Train an artificial neural network."""

    def __init__(self):
        self.ann = None
        self.connection_rate = 1
        self.learning_rate = 0.7
        self.hidden_layers = 1
        self.hidden_neurons = 8
        self.epochs = 500000
        self.iterations_between_reports = 1000
        self.desired_error = 0.001
        self.training_algorithm = libfann.TRAIN_INCREMENTAL
        self.train_data = None
        self.test_data = None

    def set_train_data(self, data):
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        self.train_data = data

    def set_test_data(self, data):
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        if data.num_input != self.train_data.num_input:
            raise ValueError("Number of inputs of test data must be same as train data")
        if data.num_output != self.train_data.num_output:
            raise ValueError("Number of output of test data must be same as train data")
        self.test_data = data

    def train(self, train_data):
        self.set_train_data(train_data)

        hidden_layers = [self.hidden_neurons] * self.hidden_layers
        layers = [self.train_data.num_input]
        layers.extend(hidden_layers)
        layers.append(self.train_data.num_output)

        sys.stderr.write("Network layout:\n")
        sys.stderr.write("* Neuron layers: %s\n" % layers)
        sys.stderr.write("* Connection rate: %s\n" % self.connection_rate)
        if self.training_algorithm not in (libfann.TRAIN_RPROP,):
            sys.stderr.write("* Learning rate: %s\n" % self.learning_rate)

        self.ann = libfann.neural_net()
        self.ann.create_sparse_array(self.connection_rate, layers)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        self.ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        self.ann.set_training_algorithm(self.training_algorithm)

        fann_train_data = libfann.training_data()
        fann_train_data.set_train_data(self.train_data.get_input(), self.train_data.get_output())

        self.ann.train_on_data(fann_train_data, self.epochs, self.iterations_between_reports, self.desired_error)
        return self.ann

    def test(self, test_data):
        self.set_test_data(test_data)

        fann_test_data = libfann.training_data()
        fann_test_data.set_train_data(self.test_data.get_input(), self.test_data.get_output())

        self.ann.reset_MSE()
        self.ann.test_data(fann_test_data)

        return self.ann.get_MSE()

if __name__ == "__main__":
    main()
