# --------------
## Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Lasso, Ridge

## Split the data and preprocess
df = pd.read_csv(path)
train = df[df['source'] == 'train']
test = df[df['source'] == 'test']
## Baseline regression model
train_baseline = train[['Item_Weight', 'Item_MRP', 'Item_Visibility']]
y_baseline = train['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(train_baseline, y_baseline, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)
## Effect on R-square if you increase the number of predictors
X_2 = train.drop(columns = ['Item_Outlet_Sales','Item_Identifier', 'source'])
X2_train, X2_test, y2_train, y2_test = train_test_split(X_2, train.Item_Outlet_Sales, test_size=0.33, random_state=42)
lr_2 = LinearRegression(normalize=True)
lr_2.fit(X2_train, y2_train)
y2_pred = lr_2.predict(X2_test)
r2_2 = r2_score(y2_test, y2_pred)
print(r2_2)
## Effect of decreasing feature from the previous model
X_3 = X_2.drop(['Item_Visibility', 'Outlet_Years'], axis=1)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_3, train.Item_Outlet_Sales, test_size=0.33, random_state=42)
lr_3 = LinearRegression(normalize=True)
lr_3.fit(X3_train, y3_train)
y3_pred = lr_3.predict(X3_test)
r2_3 = r2_score(y3_test, y3_pred)
print(r2_3)
## Detecting hetroskedacity
residuals = y3_pred - y3_test
#plt.scatter(y3_test, residuals, c='b')
#plt.title('Residual Plot vs Y_true')
## Model coefficients
predictors = X3_train.columns
coefficients = lr_3.coef_
coef = pd.Series(coefficients, predictors).sort_values()
#coef.plot(kind='bar', title='Model Cofficients')
## Ridge regression
alpha_values = [0.001,.01,0.1,5,10]
def adj_r2(Model,y_true,y_hat):
    N = len(y_hat)
    P = len(Model.coef_)
    r2 = r2_score(y_true, y_hat)
    adj = 1 - (1-r2)*(N-1)/(N-P-1)
    return adj
def Ridge_Reg(alpha) :
    R1 = Ridge(alpha=alpha, normalize=True)
    R1.fit(X3_train, y3_train)
    pred = R1.predict(X3_test)
    return R1, pred

for i in alpha_values :
    Model_Ridge, Predictions = Ridge_Reg(i)
    predictors_R = X3_train.columns
    coefficients_R = Model_Ridge.coef_
    coef_R = pd.Series(coefficients_R, predictors_R).sort_values()
    #plt.figure(figsize=(10,10))
    #coef_R.plot(kind='bar', title='Ridge Model - Alpha value is {} and Adj_R value is {}'.format(i,adj_r2(Model_Ridge,y3_test,Predictions)))

## Lasso regression
def Lasso_Reg(alpha) :
    L1 = Lasso(alpha=alpha, normalize=True)
    L1.fit(X3_train, y3_train)
    pred = L1.predict(X3_test)
    return L1, pred

for i in alpha_values :
    Model_Lasso, Predictions = Lasso_Reg(i)
    predictors_L = X3_train.columns
    coefficients_L = Model_Lasso.coef_
    coef_L = pd.Series(coefficients_L, predictors_L).sort_values()
    #plt.figure(figsize=(10,10))
    #coef_L.plot(kind='bar', title='Lasso Model - Alpha value is {} and Adj_R value is {}'.format(i,adj_r2(Model_Lasso,y3_test,Predictions)))

## Cross vallidation
train.drop(columns=['source', 'Item_Identifier'], inplace=True)
test.drop(columns=['source', 'Item_Identifier'], inplace=True)

X_train = train.drop(columns=['Item_Outlet_Sales'])
y_train = train['Item_Outlet_Sales']

scorer = make_scorer(mean_squared_error, greater_is_better = False)
Ridge_kfold = Ridge(alpha=0.1, normalize=True)
errors = -(cross_val_score(Ridge_kfold, X_train, y_train, scoring=scorer, cv=10))
print(errors, errors.mean(), errors.std())


