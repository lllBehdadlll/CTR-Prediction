
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier



data_df = pd.read_csv('../train.csv')


train_count = int(len(data_df) * 0.9)

X_train = data_df.iloc[:train_count,2:]
y_train = data_df.iloc[:train_count,1]


X_val = data_df.iloc[train_count:,2:]
y_val = data_df.iloc[train_count:,1]



X_val.head()


cat_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.5,
    task_type='GPU',
    loss_function='Logloss',
#     gpu_ram_part=0.9,
#     boosting_type='Plain',
#     max_ctr_complexity=2,
#     depth=6,
#     gpu_cat_features_storage='CpuPinnedMemory',
)


model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    verbose=10,
)



model.get_feature_importance(prettified=True)




test_df = pd.read_csv('../test.csv')



X_test = test_df.iloc[:, 1:]
X_test.head()



y_test = model.predict(X_test,
                        prediction_type='Probability',
                        ntree_start=0, ntree_end=model.get_best_iteration(),
                        thread_count=-1, verbose=None)



id_test = test_df.iloc[:, 0:1]
id_test



id_test.join(pd.DataFrame(y_test))




'''
submission_df = pd.read_csv("/kaggle/input/avazu-ctr-prediction/sampleSubmission.gz")

submission_df["click"] = y_test[:, 1]
submission_df.to_csv("submission.csv", index=False)
submission_df.head()
'''