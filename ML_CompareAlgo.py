# Classification Project: Sonar rocks or mines

# Load libraries
from matplotlib import pyplot 
from pandas import read_csv
from pandas import set_option
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier 
# Load dataset 

url = 'test3.csv'
dataset = read_csv(url)
targetcolumn= 'Dx:Cancer'

 
# Summarize Data

# Descriptive statistics
# shape
print(dataset.shape)
# types
print(dataset.dtypes)
# head 
print(dataset.head(20))
# descriptions, change precision to 3 places
set_option('precision', 3)
print(dataset.describe())
# class distribution
print(dataset.groupby(targetcolumn).size())


# Data visualizations

# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()

# Prepare Data

# Split-out validation dataset
n_col=36
X = dataset.drop([targetcolumn],axis=1) 
Y=dataset[targetcolumn]
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)

validation_size = 0.20 #80 percent is for trained. 20% is for testing
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

print(" X Train shape : {} ".format(X_train.shape))
print(" Y Train shape : {} ".format(Y_train.shape))
print(" X Validation shape : {} ".format(X_validation.shape))
print(" X Validation shape : {} ".format(Y_validation.shape))

#=================================================================================
#=========================Start Define ML Function & Tuning=======================
#=================================================================================
#=================================================================================

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


#Tune KNN
def TuneKNN(X_T,Y_T):
    neighbors = [1,3,5,7,9,11,13,15,17,19,21]
    param_grid = dict(n_neighbors=neighbors)

    model = KNeighborsClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_T, Y_T) 
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
    return (grid_result.best_params_)


# Tune SVM
def TuneSVM(X_T,Y_T):
    c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = dict(C=c_values, kernel=kernel_values)

    model = SVC(gamma='auto')
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_T, Y_T)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
#    for mean, stdev, param in zip(means, stds, params):
#        print("%f (%f) with: %r" % (mean, stdev, param))
    return(grid_result.best_params_)

#fit and predict using all of the algorithm
def LoopAll(Arr):   
    results = []
    names = []
    print("==================================================")
    print("Algorithm\t|\tCV Results\t\t|  Accuracy\t|\tConfusion")
    print("         \t|   Mean   \t|Std Dev  \t| \t\t\t\t")
    for name, model in Arr:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        model.fit(X_train, Y_train) 
        predictions = model.predict(X_validation)
        acc=accuracy_score(Y_validation, predictions)
        cm=confusion_matrix(Y_validation, predictions) 
        print(" {:6}  \t| {:.4f}   \t| {:.4f}  \t| {:.4f}   \t| {} {}".format(name,cv_results.mean(), cv_results.std(),acc,cm[0],cm[1]))
       

    # Compare Algorithms
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()
    
#=================================================================================
#=========================Finish Define Function==================================
#=========================Start Running ML prediction=============================
#=================================================================================
knnbest=TuneKNN(X,Y)
svcbest=TuneSVM(X,Y) 

# Normal Algorithms
models = []
models.append(('H-SVM', SVC(gamma='auto',C=svcbest['C'],kernel=svcbest['kernel'])))
models.append(('SVM', SVC(gamma='auto')))
models.append(('H-KNN', KNeighborsClassifier(n_neighbors=knnbest['n_neighbors'])))
models.append(('KNN', KNeighborsClassifier(n_neighbors=4)))
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier())) 
LoopAll(models)

# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(n_estimators=10)))
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10)))
ensembles.append(('VC', VotingClassifier(estimators=models)))
LoopAll(ensembles)
