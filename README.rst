========
ImgPheno
========

ImgPheno is a Python packages for extracting useful features from digital
images.

See the nbclassify_ package for example usage of imgpheno.

.. image:: https://travis-ci.org/naturalis/feature-extraction.svg?branch=master
   :target: https://travis-ci.org/naturalis/feature-extraction

Requirements
============

This Python package has the following dependencies:

* NumPy_

* OpenCV_ (2.4.x)

  * Python bindings

* Python_ (2.7.x)

For some of the example scripts you need additional dependencies:

* PyYAML_

On Debian (based) systems, most dependencies can be installed from the
software repository::

    apt-get install opencv python python-opencv python-numpy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index::

    pip install numpy pyyaml

.. _nbclassify: https://github.com/naturalis/img-classify
.. _NumPy: http://www.numpy.org/
.. _OpenCV: http://opencv.org/
.. _Python: https://www.python.org/
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
