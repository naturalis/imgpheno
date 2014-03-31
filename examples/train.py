#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import csv
import mimetypes
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np
from pyfann import libfann

import features as ft

HIST_BINS = (10,10,10)
OUTLINE_RES = 15
NORM_RANGE = (-1, 1)

def main():
    parser = argparse.ArgumentParser(description='Make training data')

    # Create a sub parser for sub-commands.
    subparsers = parser.add_subparsers(help='Specify which task to start.')

    # Create an argument parser for sub-command 'traindata'.
    help_data = "Make training data"
    parser_data = subparsers.add_parser('data',
        help=help_data,
        description=help_data)
    parser_data.add_argument('path', metavar='PATH', help='Top directory for image files. Images must be in level one sub directories.')
    parser_data.add_argument('-o', '--output', metavar='FILE', required=True, help='Path for training data file.')
    parser_data.add_argument('-t', '--test', metavar='FILE', default=None, help='Path for test data file.')
    parser_data.add_argument('--maxdim', metavar='N', type=float, default=None, help="Limit the maximum dimension for an input image. The input image is resized if width or height is larger than N. Default is no limit.")
    parser_data.add_argument('--split', metavar='N', type=int, default=None, help="Split into training and test data, where each Nth row is test data. Use --test to set the output file for test data.")
    parser_data.add_argument('--segment', metavar='PATH', help="Segment the image. Masked images are saved in PATH.")
    parser_data.add_argument('--seg-iters', metavar='N', type=int, default=5, help="The number of segmentation iterations. Default is 5.")
    parser_data.add_argument('--seg-margin', metavar='N', type=int, default=5, help="The margin of the foreground rectangle from the edges. Default is 5.")

    # Create an argument parser for sub-command 'trainann'.
    help_ann = "Train artificial neural network"
    parser_ann = subparsers.add_parser('ann',
        help=help_ann,
        description=help_ann)
    parser_ann.add_argument("--hidden-layers", metavar="N", type=int, default=1, help="Number of hidden neuron layers. Default is 1")
    parser_ann.add_argument("--hidden-neurons", metavar="N", type=int, default=8, help="Number of hidden neurons per hidden layer. Default is 8")
    parser_ann.add_argument("--epochs",metavar= "N", type=int, default=500000, help="Maximum number of epochs. Default is 500000")
    parser_ann.add_argument("--error", metavar="N", type=float, default=0.001, help="Desired error. Default is 0.001")
    parser_ann.add_argument("--learning-rate", metavar="N", type=float, default=0.7, help="Learning rate. Default is 0.7")
    parser_ann.add_argument("--connection-rate", metavar="N", type=float, default=1, help="Connection rate. Default is 1, the network will be fully connected.")
    parser_ann.add_argument('--split', metavar='N', type=int, default=None, help="Split into training and test data, where each Nth row is test data.")
    parser_ann.add_argument("-i", "--input", metavar="FILE", help="Tab separated file containing training data")
    parser_ann.add_argument("-t", "--test", metavar="FILE", default=None, help="Tab separated file containing test data")
    parser_ann.add_argument("-o", "--output", metavar="FILE", help="Name of the output file")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = "Classify an image"
    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann,
        description=help_test_ann)
    parser_test_ann.add_argument("--ann", metavar="FILE", help="A trained artificial neural network")
    parser_test_ann.add_argument("-i", "--input", metavar="FILE", help="Tab separated file containing test data")

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
    if not os.path.isdir(args.path):
        sys.stderr.write("Cannot open %s (no such directory)\n" % args.path)
        return
    if args.segment and not os.path.isdir(args.segment):
        sys.stderr.write("Cannot open %s (no such directory)\n" % args.segment)
        return
    if args.split and not args.test:
        sys.stderr.write("Must specify --test when using --split\n")
        return
    if args.output:
        out_file = open(args.output, 'w')
    if args.test:
        test_file = open(args.test, 'w')

    n_data_cols = (sum(HIST_BINS) * 2) + (OUTLINE_RES * 2)

    # Get list of image files.
    images = {}
    classes = []
    for item in os.listdir(args.path):
        path = os.path.join(args.path, item)
        if os.path.isdir(path):
            classes.append(item)
            images[item] = get_image_files(path)

    # Make codeword for each class.
    codewords = get_codewords(classes, *NORM_RANGE)

    # Write header row.
    header = ["ID"]
    channel_strs = ("BGR", "HSV")
    for channel_str in channel_strs:
        for ch, n in enumerate(HIST_BINS):
            for v in range(n):
                header.append("%s.%d" % (channel_str[ch], v+1))

    for i in range(OUTLINE_RES*2):
        header.append("OL.%d" % (i+1,))

    for i in range(len(classes)):
        header.append("OUT.%d" % (i+1,))

    out_file.write( "%s\n" % "\t".join(header) )
    if args.test:
        test_file.write( "%s\n" % "\t".join(header) )

    # Set the training data.
    training_data = TrainData(n_data_cols, len(classes))
    for im_class, files in images.items():
        for im_path in files:
            data = get_fingerprint( im_path, args, HIST_BINS )
            if not data:
                sys.stderr.write("Failed to read %s. Skipping.\n" % im_path)
                continue
            training_data.append(data, codewords[im_class], im_path)
    training_data.finalize()

    if args.split:
        training_data.split(args.split)

    # Write data rows.
    for label, input_data, output_data, is_test in training_data:
        row = []
        row.append( label )
        row.extend( input_data.astype(str) )
        row.extend( output_data.astype(str) )
        if is_test:
            test_file.write( "%s\n" % "\t".join(row) )
        else:
            out_file.write( "%s\n" % "\t".join(row) )

    if args.output:
        out_file.close()
        sys.stderr.write("Training data written to %s\n" % args.output)
    if args.test:
        test_file.close()
        sys.stderr.write("Test data written to %s\n" % args.test)

    return 0

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

def get_fingerprint(path, args, bins = None):
    img = cv2.imread(path)
    bin_mask = None

    if img == None or img.size == 0:
        return None

    sys.stderr.write("Processing %s...\n" % path)

    # Resize the image if it is larger then the threshold.
    max_px = max(img.shape[:2])
    if args.maxdim and max_px > args.maxdim:
        rf = args.maxdim / max_px
        img = cv2.resize(img, None, fx=rf, fy=rf)

    # Create copy in HSV color space.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Perform segmentation.
    if args.segment:
        mask = ft.segment(img, args.seg_iters, args.seg_margin)

        # Create a binary mask. Foreground is made white, background black.
        bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

        if isinstance(args.segment, str) and os.path.isdir(args.segment):
            # Merge the binary mask with the image.
            img_masked = cv2.bitwise_and(img, img, mask=bin_mask)

            # Save the masked image to the output folder.
            fname_base = os.path.basename(path)
            fname_parts = os.path.splitext(fname_base)
            fname = "%s.png" % (fname_parts[0])
            out_path = os.path.join(args.segment, fname)
            cv2.imwrite(out_path, img_masked)

    # Extract features
    hists = ft.hists(img, bins, bin_mask, ft.CS_BGR)
    hists_hsv = ft.hists(img_hsv, bins, bin_mask, ft.CS_HSV)
    outline = ft.simple_outline(bin_mask, OUTLINE_RES)

    # Construct data row
    row = []
    for hist in hists:
        hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
        row.extend( hist.reshape(-1) )

    for hist in hists_hsv:
        hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
        row.extend( hist.reshape(-1) )

    outline = cv2.normalize(outline, None, -1, 1, cv2.NORM_MINMAX)
    row.extend(outline)

    return row

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
    if not os.path.isfile(args.input):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.input)
        return
    if args.test and not os.path.isfile(args.test):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.test)
        return

    train_data = TrainData()
    train_data.read_from_file(args.input)

    if args.split:
        train_data.split(args.split)

    ann_trainer = TrainANN()
    ann_trainer.connection_rate = args.connection_rate
    ann_trainer.hidden_layers = args.hidden_layers
    ann_trainer.hidden_neurons = args.hidden_neurons
    ann_trainer.learning_rate = args.learning_rate
    ann_trainer.epochs = args.epochs
    ann_trainer.desired_error = args.error

    ann = ann_trainer.train(train_data)
    ann.save(args.output)
    sys.stderr.write("Artificial neural network saved to %s\n" % args.output)

    sys.stderr.write("Testing the neural network...\n")
    error = ann_trainer.test(train_data)
    sys.stderr.write("Mean Square Error on test data: %f\n" % error)

def test_ann(args):
    if not os.path.isfile(args.input):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.input)
        return
    if not os.path.isfile(args.ann):
        sys.stderr.write("Cannot open %s (no such file)\n" % args.ann)
        return

    ann = libfann.neural_net()
    ann.create_from_file(args.ann)

    test_data = TrainData()
    test_data.read_from_file(args.input)

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

class TrainData(object):
    """Class for storing training data."""

    def __init__(self, num_input = 0, num_output = 0):
        self.num_input = num_input
        self.num_output = num_output
        self.labels = []
        self.input = []
        self.output = []
        self.test_set = []
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
            is_test = self.counter in self.test_set
            return (self.labels[i], self.input[i], self.output[i], is_test)

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

    def split(self, every=2):
        if every < 2:
            raise ValueError("Every must be at least 2")
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this function")
        self.test_set = []
        for i in range(len(self.input)):
            if i % every == 0:
                self.test_set.append(i)

    def has_test_set(self):
        return len(self.test_set) > 0

    def get_input(self, test=False):
        if self.has_test_set():
            tmp = []
            for i in range(len(self.input)):
                if test and i in self.test_set:
                    tmp.append(self.input[i])
                elif not test and i not in self.test_set:
                    tmp.append(self.input[i])
            return tmp
        else:
            return self.input

    def get_output(self, test=False):
        if self.has_test_set():
            tmp = []
            for i in range(len(self.output)):
                if test and i in self.test_set:
                    tmp.append(self.output[i])
                elif not test and i not in self.test_set:
                    tmp.append(self.output[i])
            return tmp
        else:
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
        fann_test_data.set_train_data(self.test_data.get_input(test=True), self.test_data.get_output(test=True))

        self.ann.reset_MSE()
        self.ann.test_data(fann_test_data)

        return self.ann.get_MSE()

if __name__ == "__main__":
    main()
