## Ben Kite

## This script provides automated methods for comparing regression models.

## These functions are used to compare different models for
## regression, as well as to determine how model parameter changes
## influence test performance.

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


## The regCheck function is internal to the simmer function below.
## This splits the data into training and cross-validation test sets
## randomly, and then reports the R-squared values for each model
## fitted to the data.
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


## simmer

## The simmer function is a wrapper which applies the repeated random
## splits and returns a data frame summarizing the performances of the
## models.

## The data argument is a data frame with the features and outcome.
## nsamples is the number of replications of spliting the data into training and test.
## propTrain is the proportion of cases assigned to the training set.
## models is a list of sklearn regressors (even a single classifier needs to be in a list).
## features is a list of predictor variables.
## outcome is the continuous outcome of interest.
## regNames allows the user to specific names for the models to display in the output.
## maxTime is the maximum number of minutes the function should be allowed to run.
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



## simmer_plot

## Provides a quick glance at how the difference classifier compare with a histogram.

## The x argument is the return of the simmer function.
def simmer_plot(x):
    dat = x
    models = dat.columns
    for m in models:
        plt.hist(dat[m], alpha = .5)
    plt.legend(models)
    plt.show()


    
## paramTester
    
## This is used to determine how changing a parameter of the model
## influences cross validation accuracy.

## param indicates what parameter should be varied (e.g., alpha for
## Lasso).

## model is a string indicating what model should be fitted (e.g.,
## "Lasso" or "MLPRegressor").

## values is a list which indicates what levels of the parameter
## indicated by param should be used.

## the remaining arguments are passed to the simmer function defined
## above. Note that this function has different defaults.

def paramTester(param, model, values, features, outcome, data, nsamples = 100, propTrain = .5, maxTime = 10):
    models = []
    names = []
    for v in values:
        models.append(eval(model + "(" + param + "=" + str(v) + ")"))
        names.append(param + " = " + str(v))
    out = simmer(data, models, features, outcome, nsamples, propTrain, regNames = names, maxTime = maxTime)
    return(out)
  
## This is an example of its use.
paramTester("alpha", "Lasso", [.5, .1, .01]) 
