'''
Run a range of ML models on power data from Crown
Tests a range of hyperparameters using nested cross validation
Data extracted from JSON files (via crown_training_processing.py)
'''

import numpy as np
import matplotlib.pyplot as plt
import crown_training_processing
from collections import Counter
from abc import ABC, abstractmethod
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin

class Processing:
    '''
    Contains basic data processing functionality
    '''
    @staticmethod
    def extract_data():
        '''
        Extract data from JSON and process through crown_training_processing.py
        :return:
            X - data of shape (20_
            y - labels
            class_weighting - split between classes (returned as a list with length = num of classes)
        '''

        L1, L2, W = crown_training_processing.main()                # pull logvar data from processing file

        # x = features, y = classification labels (0 or 1)
        X = np.concatenate((L1, L2), axis=0)                                    # training data
        y = np.concatenate((np.zeros(L1.shape[0]), np.ones(L2.shape[0])))              # create labels

        print(f'Training data shape: {X.shape}')

        return X, y, W

    @staticmethod
    def frange(start, stop, step):
        '''
        Custom formula for range with non-integer steps
        '''
        while start < stop:
            yield start
            start += step

    @staticmethod
    def mrange(start, stop, factor):
        '''
        Custom generator for range with multiplicative steps
        '''

        while start <= stop:
            yield start
            start *= factor

class Analysis:
    '''
    Gives results of the model training
    '''
    @staticmethod
    def confusion(cm):
        '''
        Print confusion matrix statistics
        '''
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        print('True Positives(TP) = ', TP)
        print('True Negatives(TN) = ', TN)
        print('False Positives(FP) = ', FP)
        print('False Negatives(FN) = ', FN)

        confusion_stats = {
            'model_accuracy': (TP + TN) / float(TP + TN + FP + FN),
            'model_error': (FP + FN) / float(TP + TN + FP + FN),
            'precision': TP / float(TP + FP),
            'recall': TP / float(TP + FN),
            'true_positive_rate': TP / float(TP + FN),
            'false_positive_rate': FP / float(FP + TN),
            'specificity': TN / (TN + FP),
            }
        print(confusion_stats)

    @staticmethod
    def roc(y_test, y_pred):
        '''
        Plot ROC curve
        '''

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        ROC_AUC = roc_auc_score(y_test, y_pred)
        print('ROC AUC : {:.4f}'.format(ROC_AUC))

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.show()

class Training:
    '''
    Performs the model training
    '''

    @staticmethod
    def nested_cross_validation(X, y, model_type):
        '''
        Performs KFold cross validation (CV) training of the selected model
        Consists of an inner CV loop (cross validation for hyperparameter tuning)
        and an outer loop CV (holding back 1 fold for testing of final model with optimised hyperparameters)

        :param X: data
        :param y: labels
        :param model_type: selected model keyword
        :return:
            X_train - final training data
            y_train - final training labels
            X_test - final test data
            y_test - final test labels
            model - model object
        '''

        # create final model split (test data to remain for the end)
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # define outer CV split
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
        best_hypers = {}

        for train_ix, test_ix in cv_outer.split(X_train):
            # create outer kfold splits for the data
            X_train_val, X_test_val = X_train[train_ix], X_train[test_ix]
            y_train_val, y_test_val = y_train[train_ix], y_train[test_ix]

            # run inner kfold cross validation to get the best hyperparameters
            params = Training.hyperparameters(model=model_type, X_train=X_train_val, y_train=y_train_val)

            # add the best hyperparams to a dictionary
            for key, value in params.items():
                if key not in best_hypers:
                    best_hypers[key] = []
                best_hypers[key].append(value)

        # aggregate the best params across all folds
        final_hypers = {}
        for key, value in best_hypers.items():
            # for float params - take the average
            if all(isinstance(x, (float, int)) for x in value):
                final_hypers[key] = np.mean(value)
            # for integer or string - take the mode
            if all(isinstance(x, (int, str)) for x in value):
                counter = Counter(value)
                mode,_ = counter.most_common(1)[0]
                final_hypers[key] = mode

        print(f"FINAL HYPERPARAMETERS: {final_hypers}")

        # pass through relevant model to get final scores
        if model_type == 'svm':
            model = SVC(**final_hypers)
        if model_type == 'random_forest':
            model = RandomForestClassifier(**final_hypers)
        if model_type == 'knn':
            model = KNeighborsClassifier(**final_hypers)
        if model_type == 'lda':
            model = LinearDiscriminantAnalysis(**final_hypers)
        if model_type == 'gaussian_nb':
            model = GaussianNB(**final_hypers)
        if model_type == 'gaussian_pc':
            params = {}
            params['kernel'] = C(final_hypers['c']) * RBF(length_scale=final_hypers['l']) + WhiteKernel(noise_level=final_hypers['n'])
            params['max_iter_predict'] = final_hypers['max_iter_predict']
            model = GaussianProcessClassifier(**params)

        Training.run_model(model, X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test, model

    @staticmethod
    def hyperparameters(model, X_train, y_train):
        '''s
        Run cross-validation grid search for each of the models to find the best hyperparameters
        :param model: name of the model
        :param X_train: array of training data
        :param y_train: array of training classification labels
        :return grid_search.best_params_: the parameters that optimise the accuracy score
        '''

        # Support Vector Machine
        if model == 'svm':

            param_grid = {
                'C': list(Processing.frange(0,1.0,0.1)),                # regularisation parameter
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],                         # type of kernel used
                'class_weight': ['balanced'],                                           # make up for any deficits in one class
                'degree': list(range(1,6,1)),                                           # only applicable for poly
                'gamma': list(Processing.frange(0.02,0.05,0.001)),
            }
            # run cross validation
            grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2, n_jobs=-1)

        # Random Forest
        if model == 'random_forest':
            param_grid = {
                'n_estimators': list(range(50, 100, 10)),         # number of trees in the forest
                'max_features': ['sqrt', 'log2'],                 # number of features to consider when splitting
                'max_depth': [None],                              # max depth of tree (None = splits until nodes are pure)
                'min_samples_split': list(range(2,8,1)),          # min samples needed to split a node
                'min_samples_leaf': list(range(1,5,1)),           # min samples required to be at a leaf (endpoint)
                'bootstrap': [False],                             # use random samples with replacement or not
                # 'class_weight': ['balanced']
            }
            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=2, n_jobs=-1)

        # K Nearest Neighbours
        if model == 'knn':
            param_grid = {
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': list(range(10, 50, 10)),                                   # for ball tree and kd tree
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                'n_neighbors': list(range(1, 10, 1)),
                'p': [1, 2],                                                            # for minkowski. p=1 = manhattan dist, p=2 = euclidian
                'weights': ['uniform', 'distance']
            }
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=2, n_jobs=-1, scoring='accuracy', return_train_score=True)

        # Linear Discriminant Analysis
        if model == 'lda':

            # different methods for calculating covariance matrix
            covariance_estimators = [
                None,
                LedoitWolf(),
                OAS(),
                ShrunkCovariance(shrinkage=0.1),
                ShrunkCovariance(shrinkage=0.5),
                ShrunkCovariance(shrinkage=0.9)
            ]

            param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'covariance_estimator': covariance_estimators,
            }

            grid_search = GridSearchCV(LinearDiscriminantAnalysis(), param_grid, cv=5, verbose=2, n_jobs=-1,
                                       scoring='accuracy', return_train_score=True)

        # Gaussian Naive Bayes
        if model == 'gaussian_nb':
            param_grid = {
                'var_smoothing': list(Processing.mrange(1e-5, 1, 10)),      # amount of variance smoothing
            }
            grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, verbose=2, n_jobs=-1, scoring='accuracy',
                                       return_train_score=True)

        # Gaussian Process Classification
        if model == 'gaussian_pc':
            param_grid = {
                'c': list(np.logspace(-2, 2, 5)),           # sigma (vertical scale)
                'l': list(np.logspace(-2, 2, 5)),           # length scale
                'n': list(np.logspace(-2, 0, 3)),           # noise level
                'max_iter_predict': list(range(100, 500, 100))
            }

            # to optimise GPC, need to optimise c,l,n parameters individually before combining into kernel
            # create custom class to do this
            class CustomGPC(BaseEstimator, ClassifierMixin):
                # class that follows standardised format for ClassifierMixin
                def __init__(self, c=1.0, l=1.0, n=1.0, max_iter_predict=100):
                    # initialise the variables and model
                    self.c = c
                    self.l = l
                    self.n = n
                    self.max_iter_predict = max_iter_predict
                    # formula for the gaussian process kernel
                    kernel = C(self.c) * RBF(length_scale=self.l) + WhiteKernel(noise_level=self.n)
                    self.model = GaussianProcessClassifier(kernel=kernel, max_iter_predict=self.max_iter_predict)

                def fit(self, X, y):
                    # fit the model (cross-validation)
                    self.model.fit(X, y)
                    return self

                def predict(self, X):
                    return self.model.predict(X)

            grid_search = GridSearchCV(CustomGPC(), param_grid, cv=5, verbose=2, n_jobs=-1)

        # run the gridsearch
        grid_search.fit(X_train, y_train)
        results = grid_search.cv_results_

        # analyse performance of each fold
        for mean_score, params in zip(results['mean_test_score'], results['params']):
            print(f"Mean CV log loss: {mean_score:.3f} with params: {params}")

        # print outcomes
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation log loss: {grid_search.best_score_:.3f}")

        return grid_search.best_params_

    @staticmethod
    def run_model(model, X_train, y_train, X_test, y_test):
        '''
        Run the selected model
        :param model: name of the model
        :param X_train: array of training data
        :param y_train: array of training classification labels
        :param X_test: array of testing data
        :param y_test: array of testing classification labels
        :return accuracy_score(y_test, y_pred): model accuracy score
        '''

        # train model
        model.fit(X_train, y_train)
        # run inference on test data
        y_pred = model.predict(X_test)

        # print performance
        print("Model performance:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        Analysis.confusion(cm)
        Analysis.roc(y_test, y_pred)

        return accuracy_score(y_test, y_pred)

### define base class for various models
class BaseModel(ABC):
    # template for model classes
    def __init__(self):
        self.X, self.y, self.W = Processing.extract_data()
        self.model_type = None      # specify which model will be run

    def run(self):
        if self.model_type is None:
            raise NotImplementedError("Must define model_type.")
        print(f'Running {self.model_type}...')
        # intiate nested cross-validation
        X_train, y_train, X_test, y_test, model = Training.nested_cross_validation(self.X, self.y, self.model_type)
        self.plot(X_train, y_train, model)

        return model, self.W

    @abstractmethod
    def plot(self, X_train, y_train, model):
        # plotting scatter mesh plot of training data with probability contours
        pass

class SupportVectorMachine(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'svm'

    def plot(self, X_train, y_train, model):
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)

        # refit model with new params
        model.fit(X_train_2d, y_train)

        h = 0.02  # mesh grid parameter: lower value = more defined boundary

        # Create a mesh to plot the decision boundary
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot decision boundary
        plt.subplot(1, 1, 1)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot training points
        plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title('SVM Decision Boundary (PCA Reduced)')
        plt.show()

class RandomForest(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'random_forest'

    def plot(self, X_train, y_train, model):
        pass

class KNearestNeighbors(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'knn'

    def plot(self, X_train, y_train, model):
        pass

class LinearDiscriminant(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'lda'

    def plot(self, X_train, y_train, model):
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)

        h = 0.02  # step size in the mesh
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # predict the probability ranges for each class
        Z = model.predict_proba(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
        Z = Z.reshape(xx.shape)

        coolwarm_r = plt.cm.coolwarm.reversed()
        contour = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.8)  # plot probability gradient
        plt.colorbar(contour, label='Class 2 Probability')

        plt.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c='red', edgecolor='k', label='Right')
        plt.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c='blue', edgecolor='k', label='Left')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()

class GaussianNaiveBayes(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'gaussian_nb'

    def plot(self, X_train, y_train, model):
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)

        h = 0.02  # step size in the mesh
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict_proba(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
        Z = Z.reshape(xx.shape)

        contour = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.colorbar(contour, label='Class 2 Probability')

        plt.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c='blue', edgecolor='k', label='Class 1')
        plt.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c='red', edgecolor='k', label='Class 2')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.show()

class GaussianProcessC(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = 'gaussian_pc'

    def plot(self, X_train, y_train, model):
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)

        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        Z = model.predict_proba(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        plt.colorbar(contour, label='Class 2 Probability')
        # plot 0.5 contour
        contour_lines = plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--')
        plt.clabel(contour_lines, fmt={0.5: '0.5'}, inline=True, fontsize=10)

        plt.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c='blue', edgecolor='k', label='0')
        plt.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c='red', edgecolor='k', label='1')
        plt.title('Predicted Class 2 Probabilities')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()


#########

def main():
    '''
    Main function to pick which model to run
    '''
    model_type = input("Model options: svm, random forest, knn, lda, gaussian nb, gaussian pc\n"
                       "Type model type here: ").strip().lower()

    model_functions = {
        'random forest': RandomForest,
        'svm': SupportVectorMachine,
        'gaussian pc': GaussianProcessC,
        'knn': KNearestNeighbors,
        'gaussian nb': GaussianNaiveBayes,
        'lda': LinearDiscriminant
    }

    # Look up and call the corresponding function
    if model_type in model_functions:
        model = model_functions[model_type]()
        final_model, W = model.run()
    else:
        print(f"Model type '{model_type}' is not supported.")

    # save down model & spatial filters W
    folder = 'models'
    os.makedirs(folder, exist_ok=True)

    joblib.dump({'spatial_filters': W, 'model': final_model}, f'{folder}/{model_type}_{datetime.now()}.joblib')
    print("Model saved successfully.")

    return

if __name__ == "__main__":
    main()
