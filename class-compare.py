## Ben Kite

## This script provides an automated method for comparing classifiers.

import numpy, pandas
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

## The classCheck function is internal to the simmer function.
def classCheck(data, propTrain, classifier, features, outcome, probout = True):
    ind = data.index.values
    size = int(numpy.round(len(ind)*propTrain))
    use = numpy.random.choice(ind, size, replace = False)
    train = data.loc[use]
    test = data.loc[set(ind) - set(use)]
    logloss = []
    names = []
    for c in classifier:
        names.append(str(c).split("(")[0])
        trained = c.fit(train[features], train[outcome])
        if probout:
            test["prediction"] = trained.predict_proba(test[features])[:,1]
        else:
            test["prediction"] = trained.predict(test[features])
        ll = log_loss(test[outcome], test["prediction"])
        logloss.append(ll)
    logloss = pandas.DataFrame(logloss)
    logloss = logloss.transpose()
    logloss.columns = names
    return(logloss)

## The data argument is a data frame with the features and outcome
## nsamples is the number of replications of spliting the data into training and test
## propTrain is the proportion of cases assigned to the training set
## classifier is a list of sklearn classifiers (even a single classifier needs to be in a list)
## features is a list of predictor variables
## outcome is the binary outcome variable of interest
## probout is logical and indicates if the probability of a 1 should be used as the prediction
## This returns a data frame summarizing how the classifiers performed.
## The values returned are the log loss values for each classifier across the nsamples replications.
def simmer(data, nsamples, propTrain, classifier, features, outcome, probout = True):
    sd = dict()
    for i in range(0, nsamples):
        sd[i] = classCheck(dat, propTrain, classifier, features, outcome, probout)
    output = pandas.concat(sd)
    output = output.reset_index(drop = True)
    return(output)


## This function provides a quick glance at how the difference classifier compare
def simmer_plot(x):
    classes = x.columns
    for c in classes:
        plt.hist(x[c], alpha = .5)
    plt.legend(classes)
    plt.xlabel("log loss")
    plt.show()
    
## Small example
x = numpy.random.normal(size = 100)
y = numpy.random.binomial(1, .5, 100)

dat = pandas.DataFrame({"x" : x, "y" : y})

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

classifier = [RandomForestClassifier(), BaggingClassifier(), LogisticRegression()]
    
x = simmer(dat, 100, .6, classifier, ["x"], "y", probout = True)
x

simmer_plot(x)