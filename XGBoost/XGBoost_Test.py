#**Load Data**


import pandas as pd

df = pd.read_csv('train.csv')

df.head()

df.set_index('id',inplace=True)

y = df['click']

x = df.loc[:,df.columns!='click']

x.info()

#**Transformation**

from sklearn.preprocessing import LabelEncoder
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = x.select_dtypes(include=numerics)
non_numeric_columns = list(set(x.columns) - set(numeric_columns)) + list(
    set(numeric_columns) - set(x.columns))
le = LabelEncoder()
x[non_numeric_columns] = x[non_numeric_columns].apply(le.fit_transform)

#**Model Training**

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=3)

xgb.fit(X_train,y_train)

y_predict = xgb.predict(X_test)

#**Evaluation**

from sklearn.metrics import accuracy_score, precision_score
print(accuracy_score(y_pred=y_predict,y_true=y_test))
print(precision_score(y_pred=y_predict,y_true=y_test))

#**Apply model on test data**

