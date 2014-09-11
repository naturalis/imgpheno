from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='imgpheno',
    version='0.1.0',
    description='Extract useful features from digital images',
    long_description=long_description,
    url='https://github.com/naturalis/feature-extraction',
    author='Naturalis Biodiversity Center',
    author_email='serrano.pereira@naturalis.nl',
    #license='',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Recognition',
        #'License ::',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='opencv numpy image recognition computer vision features',
    packages=['imgpheno'],
    install_requires=['numpy'],
)
