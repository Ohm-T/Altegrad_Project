from sklearn.metrics import accuracy_score, log_loss
import pandas as pd
import numpy as np
import os
os.chdir("E:/TSP/M2_DS_22_23/ALTEGRAD/altegrad_challenge_2022")

def logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()



data_gt = pd.read_csv('sample_submission_test.csv').iloc[: , 1:]
data_seq_bl = pd.read_csv('sample_submission_str.csv').iloc[: , 1:]
print('Len data ground truth : ', len(data_gt))
print('Len data sequence baseline test : ', len(data_seq_bl))

# Evaluation with metrics

#print('Accuracy Score : ', accuracy_score(, y_pred_proba))

print('Log_Loss : ', logloss(data_gt.to_numpy(), data_seq_bl.to_numpy()))