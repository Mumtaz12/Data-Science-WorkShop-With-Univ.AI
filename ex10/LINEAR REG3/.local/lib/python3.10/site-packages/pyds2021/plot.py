from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .LinReg import *

def build_results_df(x_test, y_test, y_pred_test):
    results_df = np.hstack((x_test,  y_test, y_pred_test))
    results_df = pd.DataFrame(results_df)
    results_df.columns = list(x_test.columns) + ["test_set", "prediction"] #
    results_df = results_df.sort_values(x_test.columns[0])
    results_df["residuals"] = (results_df["test_set"] - results_df["prediction"])**2
    return results_df

def scatter3d(df, ax):
    #df = pd.read_csv('2016.csv')
    sns.set(style = "darkgrid")

    x = df['TV']
    y = df['Radio']
    z_hat = df['prediction']
    z = df["test_set"]

    ax.set_xlabel("TV")
    ax.set_ylabel("Radio")
    ax.set_zlabel("Sales")

    ax.scatter(x, y, z_hat, s = 50, label = "ground_truth")

    ax.scatter(x, y, z, s = 50, label = "prediction")
    ax.set_title("prediction vs test set")
    ax.legend()

def build_results_df(x_test, y_test, y_pred_test):
    results_df = np.hstack((x_test,  y_test, y_pred_test))
    results_df = pd.DataFrame(results_df)
    results_df.columns = list(x_test.columns) + ["test_set", "prediction"] #
    results_df = results_df.sort_values(x_test.columns[0])
    results_df["residuals"] = np.abs(results_df["test_set"] - results_df["prediction"])
    return results_df

def plot3d(x_test, y_test, y_pred_test):
    df = build_results_df(x_test, y_test, y_pred_test)
    fig, ax = plt.subplots(1,2, figsize = (10,10), subplot_kw={'projection':'3d'})#,  projection = '3d')
    scatter3d(df, ax[0])
    residuals_plot(df, fig, ax[1])
    fig.tight_layout(pad = 5)

    plt.show()

def residuals_plot(df, fig, ax):
    # fig = plt.figure(figsize = (9,9))
    # ax = fig.add_subplot(111, projection = '3d')
    x = df['TV']
    y = df['Radio']
    z_hat = df['prediction']
    z = df["test_set"]
    residuals = np.abs(z -z_hat)
    cmap = 'viridis'
    p = ax.scatter(x, y, residuals, cmap='viridis', 
        c = residuals, linewidth=1, s = 50, alpha = 1, label = "residuals")
    
    ax.set_xlabel("TV")
    ax.set_ylabel("Radio")
    ax.set_zlabel("Sales")

    #cbar = plt.colorbar(residuals.values)
    fig.colorbar(p,  fraction=0.05, ax = ax,  pad=0.2)
    
    ax.set_title("residuals")
    plt.legend()
    #p.tight_layout()

def fit_and_plot_linreg(X_train, X_test, y_train, y_test):
    mylreg = LinearReg()
    mylreg.fit(X_train.values, y_train.values)
    y_pred_test = mylreg.predict(X_test.values)
    plot3d(X_test, y_test, y_pred_test)
