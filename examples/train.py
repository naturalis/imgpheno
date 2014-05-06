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
    steps and features to extract must be set in a YAML file. See orchids.yml
    for an example.
    """
    parser_data = subparsers.add_parser('data',
        help=help_data,
        description=help_data)
    parser_data.add_argument("--conf", metavar="FILE", required=True, help="Path to a YAML file with feature extraction parameters.")
    parser_data.add_argument("--output", "-o", metavar="FILE", required=True, help="Output file name for training data. Any existing file with same name will be overwritten.")
    parser_data.add_argument("images", metavar="PATH", help="Directory path to load image files from. Images must be in level one sub directories. Sub directory names will be used as class names.")

    # Create an argument parser for sub-command 'trainann'.
    help_ann = """Train an artificial neural network. Optional training
    parameters can be set in a separate YAML file. See orchids.yml
    for an example file.
    """
    parser_ann = subparsers.add_parser('ann',
        help=help_ann,
        description=help_ann)
    parser_ann.add_argument("--test-data", metavar="FILE", help="Path to tab separated file with test data.")
    parser_ann.add_argument("--conf", metavar="FILE", help="Path to a YAML file with ANN training parameters.")
    parser_ann.add_argument("--output", "-o", metavar="FILE", required=True, help="Output file name for the artificial neural network. Any existing file with same name will be overwritten.")
    parser_ann.add_argument("data", metavar="TRAIN_DATA", help="Path to tab separated file with training data.")

    # Create an argument parser for sub-command 'test-ann'.
    help_test_ann = """Test an artificial neural network. If `--output` is
    set, then `--conf` must also be set. See orchids.yml for an example YAML
    file with class names."""
    parser_test_ann = subparsers.add_parser('test-ann',
        help=help_test_ann,
        description=help_test_ann)
    parser_test_ann.add_argument("--ann", metavar="FILE", required=True, help="A trained artificial neural network.")
    parser_test_ann.add_argument("--output", "-o", metavar="FILE", help="Output file name for the test results. Specifying this option will output a table with the classification result for each sample.")
    parser_test_ann.add_argument("--conf", metavar="FILE", help="Path to a YAML file with class names.")
    parser_test_ann.add_argument("--error", metavar="N", type=float, default=0.01, help="The maximum error for classification. Default is 0.01")
    parser_test_ann.add_argument("data", metavar="TEST_DATA", help="Path to tab separated file containing test data.")

    # Create an argument parser for sub-command 'classify'.
    help_classify = """Classify an image. See orchids.yml for an example YAML
    file with class names."""
    parser_classify = subparsers.add_parser('classify',
        help=help_classify,
        description=help_classify)
    parser_classify.add_argument("--ann", metavar="FILE", required=True, help="Path to a trained artificial neural network file.")
    parser_classify.add_argument("--conf", metavar="FILE", required=True, help="Path to a YAML file with class names.")
    parser_classify.add_argument("--error", metavar="N", type=float, default=0.01, help="The maximum error for classification. Default is 0.01")
    parser_classify.add_argument("image", metavar="IMAGE", help="Path to image file to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    if sys.argv[1] == 'data':
        train_data(args.images, args.conf, args.output)
    elif sys.argv[1] == 'ann':
        train_ann(args.data, args.output, args.test_data, args.conf)
    elif sys.argv[1] == 'test-ann':
        if args.output and not args.conf:
            sys.stderr.write("Option '--conf' must be set when '--output' is set.\n")
            sys.exit()
        test_ann(args.ann, args.data, args.output, args.conf, args.error)
    elif sys.argv[1] == 'classify':
        classify(args.image, args.ann, args.conf, args.error)

    sys.exit()

def train_data(images_path, conf_path, output_path):
    """Generate training data."""
    for path in (images_path, conf_path):
        if not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    yml = open_yaml(conf_path)
    if not yml:
        return 1

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

    for feature, args in vars(yml.features).iteritems():
        if feature == 'color_histograms':
            for colorspace, bins in vars(args).iteritems():
                for ch, n in enumerate(bins):
                    for i in range(1, n+1):
                        header_data.append("%s:%d" % (colorspace[ch], i))

        if feature == 'shape_outline':
            n = 2 * getattr(args, 'resolution', 15)
            for i in range(1, n+1):
                header_data.append("OUTLINE.%d" % i)

        if feature == 'shape_360':
            step = getattr(args, 'step', 1)
            output_functions = getattr(args, 'output_functions', {'mean_sd': 1})
            for f_name, f_args in vars(output_functions).iteritems():
                if f_name == 'mean_sd':
                    for i in range(0, 360, step):
                        header_data.append("360:%d.MN" % i)
                        header_data.append("360:%d.SD" % i)

                if f_name == 'color_histograms':
                    for i in range(0, 360, step):
                        for cs, bins in vars(f_args).iteritems():
                            for j, color in enumerate(cs):
                                for k in range(1, bins[j]+1):
                                    header_data.append("360:%d.%s:%d" % (i,color,k))

    for i in range(len(classes)):
        header_out.append("OUT:%d" % i)

    # Write the header row.
    out_file.write( "%s\n" % "\t".join(header_primer + header_data + header_out) )

    # Set the training data.
    training_data = common.TrainData(len(header_data), len(classes))
    fp = Fingerprint()
    failed = []
    for im_class, files in images.items():
        for im_path in files:
            if fp.open(im_path, yml) == None:
                logging.warning("Failed to read %s. Skipping." % im_path)
                failed.append(im_path)
                continue

            try:
                data = fp.make()
            except ValueError as e:
                logging.error("Fingerprint failed: %s" % e)
                logging.warning("Skipping.")
                failed.append(im_path)
                continue

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

    # Print list of failed objects.
    if len(failed) > 0:
        logging.warning("Some files could not be processed:")
        for path in failed:
            logging.warning("- %s" % path)

def train_ann(train_data_path, output_path, test_data_path=None, conf_path=None):
    """Train an artificial neural network."""
    for path in (train_data_path, test_data_path, conf_path):
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    ann_trainer = common.TrainANN()
    if conf_path:
        yml = open_yaml(conf_path)
        if not yml:
            return 1

        if 'ann' in yml:
            ann_trainer.connection_rate = getattr(yml.ann, 'connection_rate', 1)
            ann_trainer.hidden_layers = getattr(yml.ann, 'hidden_layers', 1)
            ann_trainer.hidden_neurons = getattr(yml.ann, 'hidden_neurons', 8)
            ann_trainer.learning_rate = getattr(yml.ann, 'learning_rate', 0.7)
            ann_trainer.epochs = getattr(yml.ann, 'epochs', 500000)
            ann_trainer.desired_error = getattr(yml.ann, 'error', 0.0001)

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

def test_ann(ann_path, test_data_path, output_path=None, conf_path=None, error=0.01):
    """Test an artificial neural network."""
    for path in (ann_path, test_data_path, conf_path):
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    if output_path and not conf_path:
        raise ValueError("Argument `conf_path` must be set when `output_path` is set")

    if conf_path:
        yml = open_yaml(conf_path)
        if not yml:
            return 1
        if 'classes' not in yml:
            sys.stderr.write("Classes are not set in the YAML file. Missing object 'classes'.\n")
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

    mse = ann.get_MSE()
    logging.info("Mean Square Error on test data: %f" % mse)

    if not output_path:
        return

    out_file = open(output_path, 'w')
    out_file.write( "%s\n" % "\t".join(['ID','Class','Classification','Match']) )

    # Get codeword for each class.
    codewords = get_codewords(yml.classes)

    total = 0
    correct = 0
    for label, input, output in test_data:
        total += 1
        row = []

        if label:
            row.append(label)
        else:
            row.append("")

        if len(codewords) != len(output):
            sys.stderr.write("Codeword length (%d) does not match the number of classes. "
                "Please make sure the correct classes are set in %s\n" % (len(output), conf_path))
            exit(1)

        class_e = get_classification(codewords, output, error)
        assert len(class_e) == 1, "The codeword for a class can only have one positive value"
        row.append(class_e[0])

        codeword = ann.run(input)
        class_f = get_classification(codewords, codeword, error)
        row.append(", ".join(class_f))

        # Check if the first items of the classifications match.
        if len(class_f) > 0 and class_f[0] == class_e[0]:
            row.append("+")
            correct += 1
        else:
            row.append("-")

        out_file.write( "%s\n" % "\t".join(row) )

    fraction = float(correct) / total
    out_file.write( "%s\n" % "\t".join(['','','',"%.3f" % fraction]) )
    out_file.close()

    logging.info("Correctly classified: %.1f%%" % (fraction*100))
    logging.info("Testing results written to %s" % output_path)

def classify(image_path, ann_path, conf_path, error):
    """Classify an image with a trained artificial neural network."""
    for path in (image_path, ann_path, conf_path):
        if path and not os.path.exists(path):
            sys.stderr.write("Cannot open %s (no such file or directory)\n" % path)
            return 1

    yml = open_yaml(conf_path)
    if not yml:
        return 1
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


def get_image_files(path):
    """Recursively obtain a list of image files from a path."""
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

def get_classification(codewords, codeword, error=0.01):
    if len(codewords) != len(codeword):
        raise ValueError("Lenth of `codewords` must be equal to `codeword`")
    classes = []
    for cls, cw in codewords.items():
        for i, code in enumerate(cw):
            if code == 1.0 and (code - codeword[i])**2 < error:
                classes.append((codeword[i], cls))
    classes = [x[1] for x in sorted(classes, reverse=True)]
    return classes

def open_yaml(path):
    if not os.path.isfile(path):
        sys.stderr.write("Cannot open %s (no such file)\n" % path)
        return None

    f = open(path, 'r')
    yml = yaml.load(f)
    yml = common.DictObject(yml)
    f.close()

    return yml

class Fingerprint(object):
    """Generate numerical features from an image."""

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
                fname = os.path.basename(self.path)
                out_path = os.path.join(output_folder, fname)
                cv2.imwrite(out_path, img_masked)

    def make(self):
        if self.img == None:
            raise ValueError("No image loaded")

        logging.info("Processing %s ..." % self.path)

        self._preprocess()

        logging.info("Extracting features...")

        data_row = []

        if not 'features' in self.params:
            raise ValueError("Features to extract not set. Nothing to do.")

        for feature, args in vars(self.params.features).iteritems():
            if feature == 'color_histograms':
                logging.info("- Running color:histograms...")
                data = self.get_color_histograms(self.img, args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'shape_outline':
                logging.info("- Running shape:outline...")
                data = self.get_shape_outline(args, self.bin_mask)
                data_row.extend(data)

            elif feature == 'shape_360':
                logging.info("- Running shape:360...")
                data = self.get_shape_360(args, self.bin_mask)
                data_row.extend(data)

        return data_row

    def get_color_histograms(self, src, args, bin_mask=None):
        histograms = []
        for colorspace, bins in vars(args).iteritems():
            if colorspace.lower() == "bgr":
                colorspace = ft.CS_BGR
                img = src
            elif colorspace.lower() == "hsv":
                colorspace = ft.CS_HSV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            elif colorspace.lower() == "luv":
                colorspace = ft.CS_LUV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2LUV)
            else:
                raise ValueError("Unknown colorspace '%s'" % colorspace)

            hists = ft.color_histograms(img, bins, bin_mask, colorspace)

            for hist in hists:
                hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
                histograms.extend( hist.ravel() )
        return histograms

    def get_shape_outline(self, args, bin_mask):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        resolution = getattr(args, 'resolution', 15)

        outline = ft.shape_outline(bin_mask, resolution)
        outline = cv2.normalize(outline, None, -1, 1, cv2.NORM_MINMAX)
        return outline

    def get_shape_360(self, args, bin_mask):
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        rotation = getattr(args, 'rotation', 0)
        step = getattr(args, 'step', 1)
        t = getattr(args, 't', 8)
        output_functions = getattr(args, 'output_functions', {'mean_sd': True})

        # Get the largest contour from the binary mask.
        contour = ft.get_largest_countour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Set the rotation.
        if rotation == 'FIT_ELLIPSE':
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])
        if not 0 <= rotation <= 180:
            raise ValueError("Rotation must be an integer between 0 and 180, found %s" % rotation)

        # Extract shape feature.
        intersects, center = ft.shape_360(contour, rotation, step, t)

        # Create a masked image.
        if 'color_histograms' in output_functions:
            img_masked = cv2.bitwise_and(self.img, self.img, mask=bin_mask)

        # Run the output function for each angle.
        means = []
        sds = []
        histograms = []
        for angle in range(0, 360, step):
            for f_name, f_args in vars(output_functions).iteritems():
                # Mean distance + standard deviation.
                if f_name == 'mean_sd':
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

                # Color histograms.
                if f_name == 'color_histograms':
                    # Get a line from the center to the outer intersection point.
                    line = None
                    if len(intersects[angle]) > 0:
                        line = ft.extreme_points([center] + intersects[angle])

                    # Create a mask for the line, where the line is foreground.
                    line_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                    if line != None:
                        cv2.line(line_mask, tuple(line[0]), tuple(line[1]), 255, 1)

                    # Create histogram from masked + line masked image.
                    hists = self.get_color_histograms(img_masked, f_args, line_mask)
                    histograms.append(hists)

        # Normalize results.
        if 'mean_sd' in output_functions:
            means = cv2.normalize(np.array(means), None, -1, 1, cv2.NORM_MINMAX)
            sds = cv2.normalize(np.array(sds), None, -1, 1, cv2.NORM_MINMAX)

        # Group the means+sds together.
        means_sds = np.array(zip(means, sds)).flatten()

        return np.append(means_sds, histograms)

if __name__ == "__main__":
    main()
