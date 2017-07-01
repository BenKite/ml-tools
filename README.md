# ml-tools

Ben Kite

This directory is my collection of useful tools for evaluating and
comparing machine learning models. Everything written here is meant
for use with sklearn models. Everything was written in Python 3.5. 

The following packages are required:
- sklearn
- pandas
- numpy
- datetime

classcompare.py:
Two functions that are used together to validate classifiers. These
functions allow the user to specify multiple sklearn classifiers and
evaluate them on the the same randomly sampled training and test
data. The classCheck function does most of the work, but the simmer
function is what runs the check repeatedly. This allows the user to
determine how log loss varies due to sampling variability. This
provides an idea of how classifiers should be expected to
compare in the future.

regcompare.py:
Functions used to compare competing regression models. Like the
aforementioned classcompare script, repeated validations are done on
training and test data. Currently the only metric for fit is
R-squared. This script also contains a function for iterating across
different parameter values for a model and determining how test
accuracy changes as a results.
