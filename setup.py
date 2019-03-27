# -*- coding: utf-8 -*-

"""
empi4py Setup
=============
Contains the setup script required for installing the *empi4py* package.
This can be ran directly by using::

    pip install .

or anything equivalent.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, with_statement

# Built-in imports
from codecs import open

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list by reading the file and splitting it up
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Get the version from the __version__.py file
version = None
with open('empi4py/__version__.py', 'r') as f:
    exec(f.read())

# Setup function declaration
setup(name='empi4py',
      version=version,
      author="Ellert van der Velden",
      author_email="ellert_vandervelden@outlook.com",
      maintainer="1313e",
      description=("empi4py: mpi4py with more utilities"),
      long_description=long_description,
      url="https://github.com/1313e/empi4py",
      license='BSD-3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development'
          ],
      keywords=('empi4py'),
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
      packages=find_packages(),
      package_dir={'empi4py': "empi4py"},
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      )
