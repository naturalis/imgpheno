#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import sys

import cv2
import numpy as np
from pyfann import libfann

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

class TrainData(object):
    """Class for storing training data."""

    def __init__(self, num_input = 0, num_output = 0):
        self.num_input = num_input
        self.num_output = num_output
        self.labels = []
        self.input = []
        self.output = []
        self.counter = 0

    def read_from_file(self, path, dependent_prefix="OUT:"):
        """Reads training data from file.

        Data is loaded from TSV file `path`. File must have a header row,
        and columns with a name starting with `dependent_prefix` are used as
        classification columns. Optionally, sample labels can be stored in
        a column with name "ID". All remaining columns are used as predictors.
        """
        fh = open(path, 'r')
        reader = csv.reader(fh, delimiter="\t")

        # Figure out the format of the data.
        header = reader.next()
        input_start = None
        output_start = None
        label_idx = None
        for i, field in enumerate(header):
            if field == "ID":
                label_idx = i
            elif field.startswith(dependent_prefix):
                if output_start == None:
                    output_start = i
                self.num_output += 1
            else:
                if input_start == None:
                    input_start = i
                self.num_input += 1

        if self.num_input == 0:
            fh.close()
            raise ValueError("No input columns found in training data.")
        if self.num_output  == 0:
            fh.close()
            raise ValueError("No output columns found in training data.")

        input_end = input_start + self.num_input
        output_end = output_start + self.num_output

        for row in reader:
            if label_idx != None:
                self.labels.append(row[label_idx])
            else:
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
        self.iterations_between_reports = self.epochs / 100
        self.desired_error = 0.0001
        self.training_algorithm = 'TRAIN_RPROP'
        self.activation_function_hidden = 'SIGMOID_STEPWISE'
        self.activation_function_output = 'SIGMOID_STEPWISE'
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
        if self.training_algorithm not in ('TRAIN_RPROP',):
            sys.stderr.write("* Learning rate: %s\n" % self.learning_rate)
        sys.stderr.write("* Activation function for the hidden layers: %s\n" % self.activation_function_hidden)
        sys.stderr.write("* Activation function for the output layer: %s\n" % self.activation_function_output)
        sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)

        self.ann = libfann.neural_net()
        self.ann.create_sparse_array(self.connection_rate, layers)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_hidden(getattr(libfann, self.activation_function_hidden))
        self.ann.set_activation_function_output(getattr(libfann, self.activation_function_output))
        self.ann.set_training_algorithm(getattr(libfann, self.training_algorithm))

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
