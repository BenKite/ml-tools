## Ben Kite

## This script provides an automated method for comparing regression models.

import numpy, pandas
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime

## This computes the root mean square logarithmic error, I will incorporate this later
def rmsle (actual, pred):
    n = len(actual)
    l1 = numpy.log(actual + 1)
    l2 = numpy.log(pred + 1)
    t = (l1 - l2) * (l1 - l2)
    tsum = numpy.sum(t)
    return(numpy.sqrt(tsum*(1/n)))


## The regCheck function is internal to the simmer function.
def regCheck(data, propTrain, models, features, outcome, regNames = None):
    ind = data.index.values
    size = int(numpy.round(len(ind)*propTrain))
    use = numpy.random.choice(ind, size, replace = False)
    train = data.loc[use]
    test = data.loc[set(ind) - set(use)]
    regmeas = []
    if regNames == None:
        names = []
    for m in models:
        if regNames == None:
            names.append(str(m).split("(")[0])
        trained = m.fit(train[features], train[outcome])
        test["prediction"] = trained.predict(test[features])
        out = r2_score(test[outcome], test["prediction"])
        regmeas.append(out)
    regmeas = pandas.DataFrame(regmeas)
    regmeas = regmeas.transpose()
    if regNames == None:
        regmeas.columns = names
    else:
        regmeas.columns = regNames
    return(regmeas)

## The data argument is a data frame with the features and outcome
## nsamples is the number of replications of spliting the data into training and test
## propTrain is the proportion of cases assigned to the training set
## models is a list of sklearn regressors (even a single classifier needs to be in a list)
## features is a list of predictor variables
## outcome is the continuous outcome of interest
## regNames allows the user to specific names for the models to display in the output.
## maxTime is the maximum number of minutes the function should be allowed to run
## This returns a data frame summarizing how the models performed.
def simmer(data, models, features, outcome, nsamples = 100, propTrain = .8, regNames = None, maxTime = 1440):
    tstart = datetime.now()
    sd = dict()
    for i in range(0, nsamples):
        sd[i] = regCheck(data, propTrain, models, features, outcome, regNames)
        if (datetime.now() - tstart).seconds/60 > maxTime:
            print("Stopped at " + str(i + 1) + " replications to keep things under " + str(maxTime) + " minutes")
            break
    output = pandas.concat(sd)
    output = output.reset_index(drop = True)
    return(output)


## This function provides a quick glance at how the difference classifier compare
def simmer_plot(x):
    dat = x
    models = dat.columns
    for m in models:
        plt.hist(dat[m], alpha = .5)
    plt.legend(models)
    plt.show()
    
