## Ben Kite

## This script provides automated methods for comparing prediction models.

## These functions are used to compare different models for
## regression or classification, as well as to determine how model parameter changes
## influence test performance.

import numpy, pandas
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingRegressor
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


## The repCheck function is internal to the simmer function below.
## This splits the data into training and cross-validation test sets
## randomly, and then reports the R-squared  or log loss values for each model
## fitted to the data.
def repCheck(data, propTrain, models, features, outcome, meas, mNames = None):
    ind = data.index.values
    size = int(numpy.round(len(ind)*propTrain))
    use = numpy.random.choice(ind, size, replace = False)
    train = data.loc[use]
    test = data.loc[set(ind) - set(use)]
    fitmeas = []
    if mNames == None:
        names = []
    for m in models:
        if mNames == None:
            names.append(str(m).split("(")[0])
        trained = m.fit(train[features], train[outcome])
        test["prediction"] = trained.predict(test[features])
        if meas == "r2":
            out = r2_score(test[outcome], test["prediction"])
            fitmeas.append(out)
        if meas == "logloss":
            out = log_loss(test[outcome], test["prediction"])
            fitmeas.append(out)
        if meas not in ["logloss", "r2"]:
            return("Please specify \"logloss\" or \"r2\" for meas argument")
    fitmeas = pandas.DataFrame(fitmeas)
    fitmeas = fitmeas.transpose()
    if mNames == None:
        fitmeas.columns = names
    else:
        fitmeas.columns = mNames
    return(fitmeas)


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
def simmer(data, models, features, outcome, meas, nsamples = 100, propTrain = .8, mNames = None, maxTime = 1440):
    tstart = datetime.now()
    sd = dict()
    for i in range(0, nsamples):
        sd[i] = repCheck(data, propTrain, models, features, outcome, meas, mNames)
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

def tuner(param, model, values, features, outcome, data, meas, nsamples = 100, propTrain = .5, maxTime = 10):
    models = []
    names = []
    length = len(param)
    for v in values:
        tmp = param[0] + " = " +  str(v[0])
        for i in range(1, length):
            tmp = tmp + ", " + (param[i] + " = " +  str(v[i]))
        models.append(eval(model + "(" + tmp + ")"))
        names.append(str(param) + " = " + str(v))
    out = simmer(data, models, features, outcome, meas, nsamples, propTrain, mNames = names, maxTime = maxTime)
    return(out)
    
def modelSelect(output, criterion = "mean"):
    out = output.mean()
    out = pandas.DataFrame(out)
    out.columns = [criterion + " fit"]
    out.sort_values(criterion + " fit", ascending = False)
    return(out)
  
## Here is a short working example of a regression classifier
means = [0, 0, 0]
cov = numpy.matrix(([1, .7, .7], [.7, 1, .7], [.7, .3, .7]))
N = 1000 

train = numpy.random.multivariate_normal(means, cov, N)
train = pandas.DataFrame(train)
train.columns = ["x1", "x2", "y"]
preds = ["x1", "x2"]

ds = [2, 3]
es = [50, 70]

values = []
for d in ds:
    for e in es:
        values.append([d, e])

model = "GradientBoostingRegressor"
param = ["max_depth", "n_estimators"]

output = tuner(param, model, values, preds, "y", train, "r2", propTrain = .90, maxTime = .5)
output

modelSelect(output)
