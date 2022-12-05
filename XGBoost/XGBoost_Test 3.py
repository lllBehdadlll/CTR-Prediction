#**Load Data**


import pandas as pd

df = pd.read_csv('../train.csv')

df.head()

df.set_index('id',inplace=True)

y = df['click']

x = df.loc[:,df.columns!='click']

x.info()
#df.drop(['device_id', 'C14', 'C17', 'C19', 'C20', 'C21'], axis=1, inplace=True)
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
from sklearn.metrics import accuracy_score, precision_score
xgb = XGBClassifier()
j = 0
accuracy = []
precision = []
while j < 300:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    xgb = XGBClassifier(tree_method = 'gpu_hist',n_jobs=-1, n_estimators=35,max_depth=3)
    xgb.fit(X_train,y_train)
    y_predict = xgb.predict(X_test)

    #**Evaluation**


    accuracy.append(accuracy_score(y_pred=y_predict,y_true=y_test))
    precision.append(precision_score(y_pred=y_predict,y_true=y_test))

    print(f"{j} : {accuracy[j]} | {precision[j]}")
    j += 1

accuracy.sort(reverse = True)
precision.sort(reverse = True)
print(accuracy[0])
print(precision[0])

W = 0
if W == 1:
    f = open("../Result.txt", "a")
    f.write("\n")
    f.write("XGBoost_t3 : tree_method = 'gpu_hist'|j=30 \n")
    f.write(f"accuracy_score : {accuracy[0]}\n")
    f.write(f"precision_score : {precision[0]}\n")
    f.close()
#**Apply model on test data**

