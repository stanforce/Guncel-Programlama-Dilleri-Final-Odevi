import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn import model_selection

df_2 = pd.read_excel('SATILIK_EV1.xlsx')
df = df_2.copy()

df.columns

df = df[['Fiyat', 'Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]

df

X = df.drop(["Fiyat"], axis = 1)
y = df["Fiyat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 144)

# GridSearchCV 

params = {"colsample_bytree":[0.4,0.5,0.6],
         "learning_rate":[0.01,0.02,0.09],
         "max_depth":[2,3,4,5,6],
         "n_estimators":[100,200,500,2000]}

xgb = XGBRegressor()

grid = GridSearchCV(xgb, params, cv = 10, n_jobs = -1, verbose = 2)

grid.fit(X_train, y_train)

grid.best_params_

xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.09, max_depth = 4, n_estimators = 2000)

model_xgb = xgb1.fit(X_train, y_train)

model_xgb.predict(X_test)[15:20]

y_test[15:20]

model_xgb.score(X_test, y_test)

model_xgb.score(X_train, y_train)

np.sqrt(-1*(cross_val_score(model_xgb, X_test, y_test, cv=10, scoring='neg_mean_squared_error'))).mean()

importance = pd.DataFrame({"Importance": model_xgb.feature_importances_},
                         index=X_train.columns)

importance

odasayısı = float(input("Enter the number of rooms: "))
net_m2 = float(input("Enter the net square meter: "))
kat = float(input("Enter the floor number: "))
yas = float(input("Enter the age of building: "))

new_input = pd.DataFrame({'Oda_Sayısı':[odasayısı], 'Net_m2':[net_m2], 'Katı':[kat], 'Yaşı':[yas]})

predict = model_xgb.predict(new_input)

print("The predicted price of the house is: ₺{:.2f}".format(predict[0]))
