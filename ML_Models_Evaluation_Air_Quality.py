# -*- coding: utf-8 -*-
"""
Code to compare the metrics of Deep Learning model with the initial data cleaning
1. Deep Learning
2. Random Forest Regressor
3. Stochastic Gradient Boosting Regressor
4. XGBoost Regressor
5. SVM Regressor

For multivariate remove '['AH']' from history, rmse and r2 score except for SVR, SGB
Dataset: 

    AIR QUALITY 
The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical 
sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a 
significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to 
February 2005 (one year)representing the longest freely available recordings of on field deployed air 
quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic 
Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located 
reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are 
present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually 
affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.

Attribute Information:
Date (DD/MM/YYYY)
Time (HH.MM.SS)
CO_GT - True hourly averaged concentration CO in mg/m^3
PT08_S1_CO - PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
C6H6_GT- True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
PT08_S2_NMHC- PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
Nox_GT- True hourly averaged Nitric oxide concentration in ppb (reference analyzer)
PT08_S3_Nox- PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
NO2_GT- True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
PT08_S4_NO2- PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
PT08_S5_O3 - PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
T - Temperature in Â°C
RH- Relative Humidity (%)
AH - Absolute Humidity


Created on Aug 4th 2022
Modified on 

@author: Bhanu Chander V 
"""

import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow import nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error as mse, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
#matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from functools import partial

import scipy.io as scio
import numpy as np
from tabulate import tabulate

mpl.rcParams['figure.dpi'] = 600

#-----Import .MAT file into a dataframe--------
data = pd.read_excel(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\AirQualityUCI.xlsx')

data.columns

DATA1 = data.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'],axis = 1)

DATA1.shape
DATA1.info()
DATA1.columns

DATA1['NMHC(GT)'].value_counts()
DATA1.replace(-200, np.nan, inplace = True)

#Checking if any rows have null cells
DATA1.isnull().sum()
DATA1.isna().sum()

# 8443 values out of 9357 are -200. This coulmn is not needed thus dropping this coulumn .
DATA1.drop('NMHC(GT)', axis=1, inplace=True)

# Use fillna function to fill the missing values with an estimate value
for i in DATA1.columns:
    DATA1[i] = DATA1[i].fillna(DATA1[i].mean())
    
# Finding co-relation between different gases

DATA1.mean().plot.pie(ylabel='',radius = 2,autopct = "%.2f%%");
    
'''.........Feature Selection................'''
#tart with correlation matrix and heatmap 
corr_matrix = DATA1.corr()
corr_matrix.to_excel(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\datasets\Air Quality\output\corr_matrix.xlsx')

#plt.rcParams["figure.figsize"] = [18, 18]
sns.set(font_scale=1.6)
sns.heatmap(DATA1.corr(), cmap="BrBG", annot = True, annot_kws={"size":16}, square = True)

'''.........Pair plot................'''
sns.pairplot(DATA1, palette = 'rainbow')


'''.......Univariate Feature Selection......'''

y1 = DATA1['RH']
y2 = DATA1['AH']
X = DATA1.drop(['RH', 'AH'], axis = 1)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

bestfeatures = SelectKBest(score_func = f_regression, k ='all')

fit = bestfeatures.fit(X,y1)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'score']
featureScores.to_excel(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\datasets\Air Quality\output\feature_scores_for_RH_skbest.xlsx')

fit = bestfeatures.fit(X,y2)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'score']
featureScores.to_excel(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\datasets\Air Quality\output\feature_scores_for_AH_skbest.xlsx')

'''
#Concentration of air pollutants
for i in DATA1.columns:
  fig =px.scatter(DATA1,x=data["datetime"],y=i,color =i)
  fig.show()
'''

''' DATA Normalizing '''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range =(0,1))
scaled = scaler.fit_transform(DATA1)
df = pd.DataFrame(scaled)

df = df.rename(columns={1: 'CO(GT)', 
                        2: 'PT08.S1(CO)', 
                        3: 'C6H6(GT)', 
                        4: 'PT08.S2(NMHC)', 
                        5: 'NOx(GT)', 
                        6: 'PT08.S3(NOx)', 
                        7: 'NO2(GT)', 
                        8: 'PT08.S4(NO2)', 
                        9: 'PT08.S5(O3)', 
                        10: 'T', 
                        11: 'RH', 
                        12: 'AH'})
                        #23: ,"diesel_actual_torque", 
                        #24: "instant_fuel_consumption",
                        #25: "trip_fuel"})

df_infer = DATA1.drop(['RH', 'AH'], axis =1)
Target_infer1 = DATA1[['RH','AH']]

from sklearn.model_selection import train_test_split
#X = DATA1.drop(['price'],axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df_infer,
                                                    Target_infer1,
                                                    random_state=0,
                                                    test_size=0.33)
n_cols = df_infer.shape[1]

Target_infer1 = x_train_1
df_infer = y_train_1

'''......Deep Learning Model Training ...........'''

DL_model = Sequential([
    #BatchNormalization(momentum=0.99, epsilon=0.001),
    #Dense(100, activation=partial(tf.nn.leaky_relu, alpha=0.01), input_shape=(n_cols, )),
    Dense(100, activation='relu', input_shape=(n_cols, )),
    #BatchNormalization(),
    #Dense(100, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    #Dense(50, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    Dense(100, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    Dense(50, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1),
    #Dropout(rate=0.3)
])

#DF_model.summary()

optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

DL_model.compile(optimizer=optimiser, loss='mean_squared_error')

history_DL = DL_model.fit(x_train_1,
                    y_train_1['AH'],
                    validation_split=0.3,
                    shuffle=True,
                    batch_size=32,
                    epochs=200,
                    verbose=2)

#########   Plots for model 1 inferencing 1 of position #############



'''..........................................................
..................Deep Learning Model Prediction.............
.............................................................'''

# test data
df_infer_1 = x_test_1

#del x_test_1

prediction_DL = DL_model.predict(df_infer_1)

rmse_DL_model = mse(y_test_1['AH'], prediction_DL, squared=True)
r2score_DL_model = r2_score(y_test_1['AH'], prediction_DL)

 
'''....................................................
..................Linear Regression ...................
.......................................................'''

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
history_LR = lin_reg.fit(x_train_1, y_train_1['AH'])
prediction_LR = history_LR.predict(x_test_1)

rmse_LR_model = mse(y_test_1['AH'], prediction_LR, squared=True)
r2score_LR_model = r2_score(y_test_1['AH'], prediction_LR)


'''....................................................
............Random Forest Model Prediction ............
.......................................................'''

from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(
    n_estimators = 200, criterion = "squared_error", max_depth=None,
    min_samples_split=2, min_samples_leaf=1, max_features = "auto",
    bootstrap=True ,random_state=None, verbose=2)

history_RF = RF_model.fit(x_train_1, y_train_1['AH'])

prediction_RF = RF_model.predict(x_test_1)
#Target_infer_1 = Target_infer_1.to_numpy()

# Evaluating Metrics for DL model
rmse_RF_model = mse(y_test_1['AH'], prediction_RF, squared=True)
r2score_RF_model = r2_score(y_test_1['AH'], prediction_RF)

'''....................................................
...............Stochastic Gradient Boosting.. .........
.......................................................'''

from sklearn.ensemble import GradientBoostingRegressor

GBR_model = GradientBoostingRegressor(
    learning_rate = 0.1, n_estimators = 120, loss = "squared_error", max_depth=None,
    criterion = "squared_error", min_samples_split=2, min_samples_leaf=1, 
    alpha = 0.9, random_state=None, warm_start = False, #max_features = "auto", 
    validation_fraction= 0.1, tol = 0.0001, verbose=2)

# As gradient boosting is one of the boosting algorithms it is used to minimize bias error of the model.
# It gives a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees.
# it usually outperforms random forest.
# Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# SubSample - The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias. Values must be in the range (0.0, 1.0]
# min_samples_split - The minimum number of samples required to split an internal node, for int values must be between [2, inf)
# min_samples_leaf - The minimum number of samples required to be at a leaf node.
# history_RF = RF_model.fit(x_train_1, y_train_1), for int values must be between [1, inf)
# max_depth -int, default=3, Tune this parameter for best performance; the best value depends on the interaction of the input variables. Values must be in the range [1, inf)
# Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
# warm_start, bool, default=False, When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.
# tol - default=1e-4, Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops. Values must be in the range (0.0, inf)

history_GBR = GBR_model.fit(x_train_1, y_train_1['AH'])

prediction_GBR = GBR_model.predict(x_test_1)
#Target_infer_1 = Target_infer_1.to_numpy()

rmse_GBR_model = mse(y_test_1['AH'], prediction_GBR, squared=True)
r2score_GBR_model = r2_score(y_test_1['AH'], prediction_GBR)


'''....................................................
.......................XG Boost........................
.......................................................'''

from xgboost import XGBRegressor

XGB_model = XGBRegressor(booster = 'gbtree', eta = 0.1, max_depth = 12, 
                         gamma = 0, max_delta_step =0,subsample =1, 
                         colsample_bytree =1, alpha=0
                         )#max_leaf_nodes =, scale_pos_weight =, lambda = 0.1, )

# booster - gbtree: for tree-based models, gblinear: for linear models
# tree based boosters always outperforms the linear booster and thus the later is rarely used.
# 'eta' is analogous to learning rate in GB 
# gamma specifies the minimum loss reducion required to make a split
# In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
# subsample of GBM. (0.5,1) Denotes the fraction of observations to be randomly samples for each tree. subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
# olsample_bytree - Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
# lambda = L2 regularization term on weights, default 1, (analogous to Ridge regression), This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
# alpha = L1 regularization term on weight (analogous to Lasso regression). Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
# scale_pos_weight - A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.


history_XGB = XGB_model.fit(x_train_1, y_train_1['AH'])

prediction_XGB = XGB_model.predict(x_test_1)
#Target_infer_1 = Target_infer_1.to_numpy()

rmse_XGB_model = mse(y_test_1['AH'], prediction_XGB, squared=True)
r2score_XGB_model = r2_score(y_test_1['AH'], prediction_XGB)


'''....................................................
.......................SVM Regressor........................
.......................................................'''

from sklearn.svm import SVR

SVR_model = SVR(kernel= 'linear', gamma = 'auto', tol = 1e-3, C = 1, epsilon=0.1, 
                verbose = True, max_iter = 500000)

SVR_model = SVR(kernel= 'rbf',  gamma = 'auto', tol = 1e-3, C = 1, epsilon=0.1, 
                shrinking= True, verbose = True, max_iter = 30000)

SVR_model = SVR(kernel= 'poly', degree=5, gamma = 'auto', tol = 1e-3, C = 1, epsilon=0.1, 
                shrinking= True, verbose = True, max_iter = 200000)
'''
# Linear Kernel
SVR_model = SVR(kernel= 'linear', gamma = 'auto', tol = 1e-3, C = 1, epsilon=0.1, 
                verbose = True, max_iter = 200)
'''
#Arguments
# degree = 3 (default): only for Poly kernel
# gamma = {'auto', 'scale'}, Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
# tol: Tolerance for stopping criterion.
# C: float, default=1.0, Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
# epsilon: float, default=0.1, Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
# shrinking: bool, default=True, Whether to use the shrinking heuristic. If the number of iterations is large, then shrinking can shorten the training time. However, if we loosely solve the optimization problem (e.g., by using a large stopping tolerance), the code without using shrinking may be much faster
# max_iter: int, default=-1, Hard limit on iterations within solver, or -1 for no limit.

# When training an SVM with the Radial Basis Function (RBF) kernel, two parameters must be considered: C and gamma. 
# The parameter C, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. 
# A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. 
# gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
# Proper choice of C and gamma is critical to the SVM’s performance. One is advised to use GridSearchCV with C and gamma spaced exponentially far apart to choose good values.

history_SVR = SVR_model.fit(x_train_1, y_train_1['AH'])

prediction_SVR = SVR_model.predict(x_test_1)
#Target_infer_1 = Target_infer_1.to_numpy()

rmse_SVR_model = mse(y_test_1['AH'], prediction_SVR, squared=True)
r2score_SVR_model = r2_score(y_test_1['AH'], prediction_SVR)
print(r2score_SVR_model)

'''....................................................
..........Light Gradient Boosting Machine (LGBM)........
.......................................................'''

from lightgbm import LGBMRegressor

LGBMR_model = LGBMRegressor(boosting_type ='dart', 
                            learning_rate = 0.1, n_estimators = 120)

#Arguments (to be filled)
# boosting_type: 'gdbt', 'dart', 'goss', 'rf'
# dart - dropouts meet multiple additive regression trees

# When training an SVM with the Radial Basis Function (RBF) kernel, two parameters must be considered: C and gamma. 
# The parameter C, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. 
# A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. 
# gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
# Proper choice of C and gamma is critical to the SVM’s performance. One is advised to use GridSearchCV with C and gamma spaced exponentially far apart to choose good values.

history_LGBMR = LGBMR_model.fit(x_train_1, y_train_1['AH'])

prediction_LGBMR = LGBMR_model.predict(x_test_1)
#Target_infer_1 = Target_infer_1.to_numpy()

rmse_LGBMR_model = mse(y_test_1['AH'], prediction_LGBMR, squared=True)
r2score_LGBMR_model = r2_score(y_test_1['AH'], prediction_LGBMR)




'''...............................................
...............Tabulate and save Metrics .........
..................................................
'''

data = [
        ['Linear Regression', round(rmse_LR_model,3), round(r2score_LR_model,3)],
        ['Deep Learning', round(rmse_DL_model,3), round(r2score_DL_model,3)],
        ['Random Forest', round(rmse_RF_model,3), round(r2score_RF_model,3)],
        ['Stochastic Grad Boosting', round(rmse_GBR_model,3), round(r2score_GBR_model,3)],
        ['Light GBM', round(rmse_LGBMR_model,3), round(r2score_LGBMR_model,3)],
        ['XGBoost', round(rmse_XGB_model,3), round(r2score_XGB_model,3)]
        #['Support Vector Regressor', round(rmse_SVR_model,3), round(r2score_SVR_model,3)]
        ]

head = ["Model", 'RMSE', 'r2 Score']

metric_table = tabulate(data, headers=head, tablefmt = "grid")

print(metric_table)

'''
metric_xl = pd.DataFrame(
    {'Deep Learning': [round(rmse_DL_model,3),round(r2score_DL_model,3), 
                             round(max_DL_model,3),round(min_DL_model,3),round(perc_DL_model,3)],
     'Random Forest': [round(rmse_RF_model,3),round(r2score_RF_model,3), 
                              round(max_RF_model,3),round(min_RF_model,3),round(perc_RF_model,3)],
     'Stochastic Gradient Boosting': [round(rmse_GBR_model,3),round(r2score_GBR_model,3), 
                              round(max_GBR_model,3),round(min_GBR_model,3),round(perc_GBR_model,3)],
     'XGBoost': [round(rmse_XGB_model,3),round(r2score_XGB_model,3), 
                              round(max_XGB_model,3),round(min_XGB_model,3),round(perc_XGB_model,3)],
     'Support Vector Regressor': [round(rmse_SVR_model,3),round(r2score_SVR_model,3), 
                              round(max_SVR_model,3),round(min_SVR_model,3),round(perc_SVR_model,3)]
     },
     index =['RMSE', 'r2 Score', 'Max error', 'Min error', 'Max % error']
     )

metric_xl.to_excel(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Advanced Technology\Close Circuit pump\codes\Vehicle 1\validation\outputs\errors\error_metrics.xlsx')
'''


'''...............................................
.........Error Visualization, Bar charts .........
..................................................
'''

mpl.rcParams['figure.dpi'] = 300
#%matplotlib inline
plt.plot(y_test_1['AH'], color = 'k', label = 'Actual')
plt.plot(pd.DataFrame(prediction_DL) - y_test_1['AH'].reset_index(), color = 'b', label = 'DL')
plt.plot(prediction_LR- y_test_1['AH'], color = 'r', label = 'LR')
plt.plot(prediction_RF- y_test_1['AH'], color = 'g', label = 'RF')
plt.plot(prediction_GBR- y_test_1['AH'], color = 'c', label = 'GBR')
plt.plot(prediction_XGB- y_test_1['AH'], color = 'm', label = 'XGB')
plt.plot(prediction_LGBMR- y_test_1['AH'], color = 'purple', label = 'LGBM')
plt.plot(prediction_SVR- y_test_1['AH'], color = 'brown', label = 'LGBM')
plt.title('Plots of Prediction vs Grud Truth')

#plt.suptitle('Plots of Prediction vs Grud Truth')
#plt.title('DL = Deep Learning, RF = Random Forest, GBR = Gradient Boosting, XGB = XGBoost, SVR = Support Vector', fontsize = 6)
#plt.set_xticklabels(labels)
#plt.xticks(labels)
#plt.yticks([])
#plt.text(legend_des, loc = 'center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)
plt.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0, ncol=3)
plt.show()



labels = ['LR', 'DL', 'RF', 'GBR', 'LGB', 'XGB', 'SVR']

legend_des = ['LR = Linear Regression', 'DL = Deep Learning', 'RF = Random Forest', 
              'GBR = Gradient Boosting', 'LGB = Light Grd Boosting', 
              'XGB = XGBoost', 'SVR = Support Vector']

RMSE_vector = [rmse_LR_model, rmse_DL_model, rmse_RF_model, rmse_GBR_model, 
               rmse_LGBMR_model, rmse_XGB_model, rmse_SVR_model]

R2_vector = [r2score_LR_model, r2score_DL_model, r2score_RF_model, r2score_GBR_model, 
               r2score_LGBMR_model, r2score_XGB_model, r2score_SVR_model]


x = np.arange(len(labels)) # the label locations
width = 0.5  # width of the bars


mpl.rcParams['figure.dpi'] = 600
#%matplotlib inline
plt.bar(labels, RMSE_vector, width, label = 'RMSE')
plt.suptitle('Bar chart showing RMSE of selected models')
plt.title('DL = Deep Learning, RF = Random Forest, GBR = Gradient Boosting, XGB = XGBoost, SVR = Support Vector', fontsize = 6)
#plt.set_xticklabels(labels)
plt.xticks(labels)
#plt.yticks([])
#plt.text(legend_des, loc = 'center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)
#plt.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0, ncol=3)
plt.show()

mpl.rcParams['figure.dpi'] = 600
#%matplotlib inline
plt.bar(labels, R2_vector, width, label = 'r2_score')
plt.suptitle('Bar chart showing r2score of selected models')
plt.title('DL = Deep Learning, RF = Random Forest, GBR = Gradient Boosting, XGB = XGBoost, SVR = Support Vector', fontsize = 6)
#plt.set_xticklabels(labels)
plt.xticks(labels)
#plt.yticks([])
#plt.text(legend_des, loc = 'center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)
#plt.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0, ncol=3)
plt.show()

# Two bar graphs in one plot


fig, ax = plt.subplots()
rect1 = ax.bar(x- width/2, RMSE_vector, width, label = 'RMSE')
rect2 = ax.bar(x- width/2, R2_vector, width, label = 'r2_score')

ax.set_title('RMSE and r2 Scores for all models')
ax.set_ylabel('Values')
ax.set_xticklabels(labels)
ax.legend()

'''........................................................
----------------SOME LEARNINGS-----------------------------

In Decision Trees, Consider a single training dataset that we randomly split into two 
parts. Now, let’s use each part to train a decision tree in order to 
obtain two models. When we fit both these models, they would yield different
 results. Decision trees are said to be associated with high variance due 
 to this behavior.
 
 In boosting, the trees are built sequentially such that each subsequent tree 
 aims to reduce the errors of the previous tree. Each tree learns from its predecessors and updates the residual errors. Hence, the tree that grows next in the sequence will learn from an updated version of the residuals.

The base learners in boosting are weak learners in which the bias is high, 
and the predictive power is just a tad better than random guessing. Each of 
these weak learners contributes some vital information for prediction, 
enabling the boosting technique to produce a strong learner by effectively
combining these weak learners. The final strong learner brings down both 
the bias and the variance.

In contrast to bagging techniques like Random Forest, in which trees are grown 
to their maximum extent, boosting makes use of trees with fewer splits. 
Such small trees, which are not very deep, are highly interpretable. 
Parameters like the number of trees or iterations, the rate at which the 
gradient boosting learns, and the depth of the tree, could be optimally 
selected through validation techniques like k-fold cross validation. 
Having a large number of trees might lead to overfitting. So, it is necessary 
to carefully choose the stopping criteria for boosting.


The boosting ensemble technique consists of three simple steps:

1. An initial model F0 is defined to predict the target variable y. This model will be associated with a residual (y – F0)
2. A new model h1 is fit to the residuals from the previous step
3. Now, F0 and h1 are combined to give F1, the boosted version of F0. The mean squared error from F1 will be lower than that from F0:

    F1(x) = F0(x) + h1(x)

To improve the performance of F1, we could model after the residuals of F1 and create a new model F2:
    F2(x) = F1(x) + h2(x)
This can be done for ‘m’ iterations, until residuals have been minimized as much as possible:    
    Fm(x) = Fm-1(x) + hm(x)
    
    
    
    
    
    