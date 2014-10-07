========
ImgPheno
========

ImgPheno is a Python packages for extracting useful features from digital
images.

See the nbclassify_ package for example usage of imgpheno.

.. image:: https://travis-ci.org/naturalis/imgpheno.svg?branch=master
   :target: https://travis-ci.org/naturalis/imgpheno

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

    apt-get install python-opencv python-numpy python-yaml

More recent versions of some Python packages can be obtained via the Python
Package Index::

    pip install -r requirements.txt


Installation
============

The ImgPheno_ package can be installed from the GitHub repository::

    git clone https://github.com/naturalis/imgpheno.git
    cd imgpheno/
    python setup.py install

Or if you have a source archive file::

    pip install imgpheno-0.1.0.tar.gz


.. _nbclassify: https://github.com/naturalis/nbclassify
.. _NumPy: http://www.numpy.org/
.. _OpenCV: http://opencv.org/
.. _Python: https://www.python.org/
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
