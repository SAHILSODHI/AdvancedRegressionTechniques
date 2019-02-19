#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 15:56:30 2018

@author: sahilsodhi
"""
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df = train_df.append(test_df, sort = False).reset_index(drop=True)
test_df_id = test_df.pop('Id')

train_df.info()
        
#categ_columns = train_df.select_dtypes(include=['object']).copy().columns.values
#categ_columns_int = train_df.select_dtypes(include=['int']).copy().columns.values


# Nominal Classification
train_df['MSSubClass']= train_df['MSSubClass'].astype('object')


#converting neighborhood to 5 nominal classification based on SalePrice
neighborhood_map = {
        "MeadowV" : 0, "IDOTRR" : 0,"BrDale" : 0,"BrkSide" : 0, "Edwards" : 0,"OldTown" : 0,
        "Sawyer" : 0,"Blueste" : 0,"SWISU" : 0, "NPkVill": 0,"NAmes" : 0,
        "Mitchel": 1,"SawyerW" : 1,"Gilbert" : 1,"NWAmes" : 1,
        "Blmngtn" : 2,"CollgCr" : 2,"Crawfor" : 2,"Veenker" : 2,"ClearCr" : 2,"Somerst" : 2,
        "Timber" : 3,
        "StoneBr" : 4,"NridgHt" : 4,"NoRidge" : 4
        }

house_dict = {"1Story": 0,"1.5Unf":1,"1.5Fin":2, "SFoyer":3 ,"SLvl":3,"2Story": 4,"2.5Unf":5,"2.5Fin":6}

roofstyle_map = {"Flat":0, "Gable":1, "Hip":1, "Shed":1, "Gambrel":2, "Mansard":2}

garage_dict = {None: 0, "Unf": 0, "RFn": 1, "Fin": 2}

rating_map = {None: 0, "No":0, "Po": 1, "Mn": 2, "Fa": 2, "Av": 3, "TA": 3, "Gd": 4, "Ex": 5}

train_df["RoofStyle"] = train_df["RoofStyle"].map(roofstyle_map).astype(int)
train_df["HouseStyle"] = train_df["HouseStyle"].map(house_dict).astype(int)
train_df["Neighborhood"] = train_df["Neighborhood"].map(neighborhood_map).astype(int)
train_df["ExterCond"] = train_df["ExterCond"].map(rating_map).astype(int)
train_df["BsmtQual"] = train_df["BsmtQual"].map(rating_map).astype(int)
train_df["BsmtCond"] = train_df["BsmtCond"].map(rating_map).astype(int)
train_df["KitchenQual"] = train_df["KitchenQual"].map(rating_map).astype(int)
train_df["FireplaceQu"] = train_df["FireplaceQu"].map(rating_map).astype(int)
train_df["GarageQual"] = train_df["GarageQual"].map(rating_map).astype(int)
train_df["GarageCond"] = train_df["GarageCond"].map(rating_map).astype(int)
train_df["GarageFinish"] = train_df["GarageFinish"].map(garage_dict).astype(int)
train_df["ExterQual"] = train_df["ExterQual"].map(rating_map).astype(int)
train_df["HeatingQC"] = train_df["HeatingQC"].map(rating_map).astype(int)
train_df["BsmtExposure"] = train_df["BsmtExposure"].map(rating_map).astype(int)
train_df['TotalFloorArea'] = train_df['1stFlrSF']+train_df['2ndFlrSF']
train_df['TotalPorchArea']=train_df['OpenPorchSF']+train_df['EnclosedPorch']+train_df['3SsnPorch']+train_df['ScreenPorch']
train_df['TotalBathrooms']=train_df['FullBath']+train_df['BsmtHalfBath']+train_df['BsmtFullBath']+train_df['HalfBath']
train_df['HouseAge'] = train_df['YearBuilt'].max() - train_df['YearBuilt']
train_df['YearsSinceRemodelled'] = train_df['YearBuilt'].max() - train_df['YearRemodAdd']
train_df['YearsSinceSold'] = train_df['YearBuilt'].max() - train_df['YrSold']
train_df['GarageAge'] = train_df['YearBuilt'].max() - train_df['GarageYrBlt']

#find correlation with the target variable, remove the features with low correlation
train_df[train_df.columns[1:]].corr(method = 'pearson')['SalePrice'][:].sort_values(ascending = True)

train_df = train_df.drop(['Id','PoolQC','GarageYrBlt','MiscFeature','Fence','Alley','YearBuilt','YrSold','YearRemodAdd','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
                         'FullBath','BsmtHalfBath','BsmtFullBath','HalfBath','1stFlrSF','2ndFlrSF'],1)

categories = len(train_df.select_dtypes(include=['object']).columns)
numerical_features = len(train_df.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', categories, 'categorical and ',
      numerical_features, 'numerical features')

# One hot encoding for categorical features in the dataset.
categorical_features = train_df.select_dtypes(include=['object']).copy().columns
for column in categorical_features:
        onehot_train_df = pd.get_dummies(train_df[column], prefix=column)
        train_df = train_df.drop(column, 1)
        train_df = train_df.join(onehot_train_df)

# Dividing the train_df to train_df and test_df.
test_df = train_df
train_df = train_df[train_df.SalePrice > 0]
test_df['SalePrice']=test_df['SalePrice'].fillna(value = -1,axis=0)
test_df = test_df[test_df.SalePrice == -1]
test_df = test_df.drop(['SalePrice'],1)
        
# Handling null values for training and test datasets.
for col in train_df.columns:
    train_df[col]=train_df[col].fillna(value = train_df[col].median(),axis=0)

for col in test_df.columns:
    test_df[col]=test_df[col].fillna(value = test_df[col].median(),axis=0)

#find correlation of independent variables with the target variable.
correlation = train_df[train_df.columns[1:]].corr()['SalePrice'][:].sort_values(ascending = True)

# Outlier detection using boxplots and jointplot.

# Overall Quality vs Sale Price
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# TotalFloorArea vs Sale Price
sns.jointplot(x=train_df['TotalFloorArea'], y=train_df['SalePrice'], kind='reg')

# Living Area vs Sale Price
sns.jointplot(x=train_df['GrLivArea'], y=train_df['SalePrice'], kind='reg')
# Removing outliers and influential points manually 
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000)].index).reset_index(drop=True)

# Neighborhood vs Sale Price
var = 'Neighborhood'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# HouseAge vs Sale Price
sns.jointplot(x=train_df['HouseAge'], y=train_df['SalePrice'], kind='reg')

# YearsSinceRemodelled vs Sale Price
sns.jointplot(x=train_df['YearsSinceRemodelled'], y=train_df['SalePrice'], kind='reg')

# GarageAge vs Sale Price
sns.jointplot(x=train_df['GarageAge'], y=train_df['SalePrice'], kind='reg')
#corr = train_df.corr()

# Test for normal distribution using histogram and probability plot
# Plot Histogram
sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())

# To resolve skewness, bring the value close to 0. Can further apply box-cox transformation for dependent features.
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
print("Skewness: %f" % train_df['SalePrice'].skew())

train_df.columns.get_loc("SalePrice")
df1 = train_df.pop('SalePrice')
train_df['SalePrice'] = df1

X=train_df.drop('SalePrice',axis=1)
y = train_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_df = sc.transform(test_df)

#feature extraction using PCA
# number of principle components which accounts for more than 85 percent variance.
from sklearn.decomposition import PCA
pca = PCA(n_components = 22)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_test_df = pca.transform(X_test_df)
explained_variance = pca.explained_variance_ratio_


#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(normalize = False)
lin_reg.fit(X_train, y_train)

lin_pred=lin_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test,lin_pred))

#Plot to visualize prediction errors.
#https://scikit-learn.org/0.16/auto_examples/plot_cv_predict.html

fig, ax = plt.subplots()
ax.scatter(y_test, lin_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Linear Regression')
plt.show()

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lin_reg, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Lasso Regression
from sklearn.linear_model import Lasso
lass_reg = Lasso(normalize=True,alpha=0.0001,max_iter=50)
lass_reg=lass_reg.fit(X_train,y_train)
#Predicting test dataset for lasso regression model
lasso_pred=lass_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test,lasso_pred))

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha': [0.00001,0.0001,0.001,0.01], 'max_iter': [50,100,200]}]
grid_search = GridSearchCV(estimator = lass_reg,
                           param_grid = parameters,
                           scoring = 'explained_variance',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lass_reg, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, lasso_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Lasso Regression')
plt.show()

#Ridge Regression
from sklearn.linear_model import Ridge
ridg_reg = Ridge(alpha=0.8,max_iter=10)
ridg_reg = ridg_reg.fit(X_train, y_train)
# Predicting test dataset for ridge regression model
ridg_pred = ridg_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test, ridg_pred))



from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = ridg_reg, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, ridg_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Ridge Regression')
plt.show()

#RandomForest Regression
from sklearn.ensemble import RandomForestRegressor
rforest=RandomForestRegressor(n_estimators=68, max_depth=11, bootstrap = True,random_state=0)
rforest=rforest.fit(X_train,y_train)
#Predicting test dataset
forest_pred=rforest.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test, forest_pred))



#Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, forest_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Random Forest')
plt.show()

#min_samples_split: Minimum number of observation which is required in a node to be considered for splitting. 
#It is used to control overfitting.
#Gradient Boosting Regressor
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 590, max_depth = 2, min_samples_split = 2, 
                                         learning_rate = 0.06, loss = 'ls')
gboost=clf.fit(X_train,y_train)
#Predicting test dataset
gb_pred=gboost.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test, gb_pred))



#Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, gb_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Gradient Boosting Regressor')
plt.show()


#XGBoost
import xgboost as xgb
regr = xgb.XGBRegressor(
                        colsample_bytree=0.2,
                        learning_rate=0.07,
                        max_depth=4,
                        min_child_weight=2,
                        n_estimators=4000,
                        reg_alpha=0.9,
                        reg_lambda=0.6,
                        subsample=0.2,
                        silent=True)

regr.fit(X_train, y_train)
xg_pred = regr.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test, xg_pred))
#0.011492987568884776
print ("Root Mean Square Logarithmic Error (Blending Models)=",RMLSE)



#Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, xg_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Extreme Gradient Boosting(XG Boost)')
plt.show()


#AdaBoost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
regr_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15),n_estimators=600,random_state=0,
                             learning_rate=2,loss="exponential")
regr_ada.fit(X_train, y_train)
ada_pred=regr_ada.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(y_test, ada_pred))
print ("Root Mean Square Logarithmic Error (Blending Models)=",RMLSE)


#Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, ada_pred, edgecolors=(0, 0, 0), alpha=1, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=3, color = 'black')
ax.set(xlabel='Measured', ylabel='Predicted')
plt.title('Adaptive Boosting')
plt.show()

y_ada = regr_ada.predict(X_test_df)  
y_xgb = regr.predict(X_test_df)
y_gboost = gboost.predict(X_test_df)
y_rforest = rforest.predict(X_test_df)
y_pred = (y_ada+y_xgb+y_gboost+y_rforest)/4

y_pred2 = np.expm1(y_pred)
y_pred4 = np.e**(y_pred) - 1

#Blending Models
blend_pred=(ada_pred+gb_pred+xg_pred+forest_pred)/4
RMLSE = np.sqrt(mean_squared_log_error(y_test, blend_pred))
print ("Root Mean Square Logarithmic Error (Blending Models)=",RMLSE)

submission = pd.DataFrame({
        "Id": test_df_id,
        "SalePrice": y_pred2
    })
    
submission.to_csv('submission.csv', index=False)