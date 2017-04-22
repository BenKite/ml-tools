## Ben Kite

## This script provides an automated method for comparing classifiers.

import numpy, pandas
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime

## The classCheck function is internal to the simmer function.
def classCheck(data, propTrain, classifier, features, outcome, classNames = None, probout = True, useauc = True):
    ind = data.index.values
    size = int(numpy.round(len(ind)*propTrain))
    use = numpy.random.choice(ind, size, replace = False)
    train = data.loc[use]
    test = data.loc[set(ind) - set(use)]
    classmeas = []
    if classNames == None:
        names = []
    for c in classifier:
        if classNames == None:
            names.append(str(c).split("(")[0])
        trained = c.fit(train[features], train[outcome])
        if probout:
            test["prediction"] = trained.predict_proba(test[features])[:,1]
        else:
            test["prediction"] = trained.predict(test[features])
        if useauc:
            out = roc_auc_score(test[outcome], test["prediction"])
        else:
            out = log_loss(test[outcome], test["prediction"])
        classmeas.append(out)
    classmeas = pandas.DataFrame(classmeas)
    classmeas = classmeas.transpose()
    if classNames == None:
        classmeas.columns = names
    else:
        classmeas.columns = classNames
    return(classmeas)

## The data argument is a data frame with the features and outcome
## nsamples is the number of replications of spliting the data into training and test
## propTrain is the proportion of cases assigned to the training set
## classifier is a list of sklearn classifiers (even a single classifier needs to be in a list)
## features is a list of predictor variables
## outcome is the binary outcome variable of interest
## classNames allows the user to specific names for the classifiers to display in the output. This was included incase the same classifier (with varying options) is used multiple times. Defaults to None, which uses the names of the classifiers in sklearn.
## maxTime is the maximum number of minutes the function should be allowed to run
## probout is logical and indicates if the probability of a 1 should be used as the prediction
## This returns a data frame summarizing how the classifiers performed.
## The values returned are the log loss values for each classifier across the nsamples replications.
def simmer(data, classifier, features, outcome, nsamples = 100, propTrain = .8, classNames = None, maxTime = 1440, probout = True, useauc = True):
    tstart = datetime.now()
    sd = dict()
    for i in range(0, nsamples):
        sd[i] = classCheck(data, propTrain, classifier, features, outcome, classNames, probout, useauc)
        if (datetime.now() - tstart).seconds/60 > maxTime:
            print("Stopped at " + str(i + 1) + " replications to keep things under " + str(maxTime) + " minutes")
            break
    output = pandas.concat(sd)
    output = output.reset_index(drop = True)
    return(output)


## This function provides a quick glance at how the difference classifier compare
def simmer_plot(x):
    classes = x.columns
    for c in classes:
        plt.hist(x[c], alpha = .5)
    plt.legend(classes)
    plt.xlabel("classification measure")
    plt.show()
    
## Small example
#x = numpy.random.normal(size = 100)
#y = numpy.random.binomial(1, .5, 100)

#dat = pandas.DataFrame({"x" : x, "y" : y})

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.linear_model import LogisticRegression

#classifiers = [RandomForestClassifier(), BaggingClassifier(), LogisticRegression()]
    
## Run until it hits 6 seconds (.1 minutes)
#x = simmer(dat, classifiers, ["x"], "y", maxTime = .1)

#simmer_plot(x)

## Now with special nicknames for the classifiers
#x = simmer(dat, classifiers, ["x"], "y", classNames = ["Fangorn Forest", "Frodo Baggins", "Legolas Regression(?)"])
#simmer_plot(x)