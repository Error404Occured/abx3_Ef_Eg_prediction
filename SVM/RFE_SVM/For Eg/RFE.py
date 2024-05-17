
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# %matplotlib inline 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import re
import random
import os
from sklearn.model_selection import train_test_split

X = pd.read_csv(r"C:\Users\hp\Downloads\rfe\SVM\For Eg\my_X_eg.csv")
y = pd.read_csv(r"C:\Users\hp\Downloads\rfe\SVM\For Eg\y_eg.csv")

def run_cv(X, y, pos,n_cv = 4,normalize=False):
    """
    Function to run Cross-validation
    """
    kf = KFold(n_splits=n_cv)
    errors = []
    for idx, (train, val) in enumerate(kf.split(X)):
        print('Current rfe_run_idx:',pos,' Current index:',idx)
        if normalize:
            _X_cv_train = X.values[train]
            _X_cv_val = X.values[val]
            scaler = StandardScaler()
            X_cv_train = scaler.fit_transform(_X_cv_train)
            X_cv_val = scaler.transform(_X_cv_val)

        else:
            X_cv_train = X.values[train]
            X_cv_val = X.values[val]

        y_cv_train = y.values[train]
        y_cv_val = y.values[val]

        # Model fit and prediction
        model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
        model.fit(_X_cv_train, y_cv_train)
        y_pred_val = model.predict(X_cv_val)

#         epochs = 50
#         batch_size = 32

#         checkpoint_dir = './checkpoints/'
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         # Define the file path
#         filepath = os.path.join(checkpoint_dir, 'RFE_model_ef.weights.h5')
#         checkpoint = ModelCheckpoint(filepath,
#                                      monitor='val_loss',
#                                      verbose=1,
#                                      save_best_only=True,
#                                      save_weights_only=True,
#                                      mode='min')

# #         model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# #         model.fit(X_cv_train, y_cv_train, validation_data=(X_cv_val, y_cv_val), batch_size=batch_size, epochs=epochs,callbacks=[checkpoint])

#         model.load_weights(filepath)
#         y_pred_val = model.predict(X_cv_val)
        rmse_val = np.sqrt(mean_squared_error(y_cv_val, y_pred_val))
        errors.append(rmse_val)
    return np.mean(np.array(errors))

current_Xcols = X.columns
rfe_results = {"rmse_cv":[], "sel_cols":[]}
while(True):
    rfe_run_idx = 0
    _rmse_cvs = []
    for rm_idx, rm_col in enumerate(current_Xcols):
        _Xcols = current_Xcols.drop(rm_col)

        # Get CV error for this set
        _rmse_cv = run_cv(X[_Xcols], y,rfe_run_idx, normalize=True)
        _rmse_cvs.append(_rmse_cv)

    _rmse_cvs = np.array(_rmse_cvs)

    worst_col = current_Xcols[np.argmin(_rmse_cvs)]

    print("Worst column %s" %worst_col)
    print("RFE RMSE CV %.2f" %np.min(_rmse_cvs))
    with open('output_eg.txt', 'a') as file:
        file.write("Worst column %s\n" % worst_col)
        file.write("RFE RMSE CV %.2f\n" % np.min(_rmse_cvs))
    current_Xcols = current_Xcols.drop(worst_col)
    if((len(rfe_results["rmse_cv"])>0) and (np.min(_rmse_cvs)>rfe_results["rmse_cv"][-1])):
        break
    rfe_results["rmse_cv"].append(np.min(_rmse_cvs))
    rfe_results["sel_cols"].append(current_Xcols)
    rfe_run_idx+=1

n_features= len(X.columns) - np.arange(len(rfe_results["rmse_cv"])) - 1
min_idx = np.argmin(rfe_results["rmse_cv"])
rmse = rfe_results["rmse_cv"][min_idx]
sel_features = rfe_results["sel_cols"][min_idx]
print(sel_features)
