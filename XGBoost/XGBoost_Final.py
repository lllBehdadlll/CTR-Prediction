import pandas as pd

df = pd.read_csv('../train.csv')

df.set_index('id',inplace=True)

y = df['click']

x = df.loc[:,df.columns!='click']



from sklearn.preprocessing import LabelEncoder
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = x.select_dtypes(include=numerics)
non_numeric_columns = list(set(x.columns) - set(numeric_columns)) + list(
    set(numeric_columns) - set(x.columns))
le = LabelEncoder()
x[non_numeric_columns] = x[non_numeric_columns].apply(le.fit_transform)

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

xgb = XGBClassifier(tree_method = 'gpu_hist',n_jobs=-1, n_estimators=35,max_depth=3)
xgb.fit(x,y)

df_test = pd.read_csv("../test.csv")
df_test.set_index('id',inplace=True)

from sklearn.preprocessing import LabelEncoder
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = df_test.select_dtypes(include=numerics)
non_numeric_columns = list(set(df_test.columns) - set(numeric_columns)) + list(
    set(numeric_columns) - set(df_test.columns))
df_test[non_numeric_columns] = df_test[non_numeric_columns].apply(le.fit_transform)

y_predict = xgb.predict(df_test)

df_final = pd.DataFrame()
df_final['id'] = df_test.index
df_final['click'] = y_predict
df_final.to_csv('submission.csv')

