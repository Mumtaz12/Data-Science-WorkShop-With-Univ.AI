
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class LinearReg:
    """
    This is a basic linear regression class.
    
    create the following methods (instructions are in the method docstrings)
    
    methods to complete:
    ______________
    
    _center_data: a function which centers the data. Should only be fed X_train

        arguments
        ---------
        the following arguments should come from the training set:
        X: 2-dimensional nd.array
        the design or feature matrix, one column per feature and one row per datapoint
        y: 1-dimensional nd.array
        a 1-d numpy array representing the response variable
        ---------

        returns 
        _______
        centered X and y nd.arrays
    
    fit:
        a function which fits the data.
        
        arguments
        ---------
        same as _center_data
        ---------

        returns
        -------
        None
        
    predict: this method will make a prediction given X_test.
        
        Arguments
        ---------
        same as _center_data, but feed only the test set.
        ---------

        Returns
        -------
        y_pred_test: nd.array
            your prediction of y for the test set

    
    ______________
        
    Make sure the following attributes get assigned:
    self._x_means
    self._y_mean
    self.coef_
    self.intercept_

    """
    def __init__(self) -> None:
        print("building linear regression instance")
    
    def _center_data(self, X, y):
        """
        INSTRUCTIONS:
        1. assign `_x_means` to self, along the axis such that 
           the numbers of means matches the number of features (2)
        2. assign `_y_mean` to self (y.mean())
        3. subtract _x_means from X and assign it to X_centered
        4. subtract _y_mean from y and assign it to y_centered
        """
        self._x_means = X.mean(axis=0)
        self._y_mean = y.mean(axis = 0)

        X_centered = X - self._x_means
        y_centered = y - self._y_mean

        return X_centered, y_centered

        
    def fit(self, X, y) -> None:
        """
        INSTRUCTIONS:
        1. center the data by calling _center_data
        2. Use the centered data to impliment the matrix formula for linear regression. 
        3. Use the matrix formula shown in class and assign the result to self.coef_
        4. Calculate the intercept by _y_mean - _x_means @ coef_ and save it as attribute intercept_
        """
        #assert isinstance(X, np.ndarray), "X must be an np.array"
        #assert isinstance(y, np.ndarray), "y must be an np.array"
        assert len(X) == len(y), "X and y must have the same number of datapoints"
        try:
            X = X.values
            y = y.values
        except:
            pass
        
        #your code starts here:
        X_centered, y_centered = self._center_data(X,y)
        self.coef_ = np.linalg.pinv((X_centered.T @ X))@(X_centered.T @ y_centered)
        try:
            self.intercept_ = self._y_mean - self._x_means @ self.coef_
        except:
            print(self._x_means.shape, self._y_mean.shape)
    
    def predict(self, X) -> np.ndarray:
        """
        INSTRUCTIONS:
        1. Take the dot product of X and coef_ and add the intercept_.
        2. return the result
        """
        
        #your code starts here:
        yhat = self.intercept_ + X @ self.coef_
        return yhat