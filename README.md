# ml-tools

Ben Kite

This directory is my collection of tools for machine learning models.
Everything written here is meant for use with sklearn models. However,
the ultimate goal with this repo is to eventually provide scripts that
work with a variety of machine learning libraries.  Everything was
written in Python 3.5.

The following packages are required:
- sklearn
- pandas
- numpy
- datetime

cross_validate.py: 

Contains functions for cross validation analysis conducted by randomly
splitting data into training and test sets. This script is still in
development, but it works in its current state. Eventually I want the
cross validation process to work across a variety of machine learning
libraries to compare the test performance of models.

