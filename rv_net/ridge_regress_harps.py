from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.linear_model import Ridge
from rv_net import load_dataset_ridge
import tensorflow as tf

def ridge_regress_harps(TRAIN_FILE_NAME_LIST, VAL_FILE_NAME_LIST, alpha, verbose):
  rms_avg_list = []
  weight_decay_list_t = []
  gaussian_noise_list_t = []
  rms_x_list = []

  all_bjds_val = []
  all_pred_val = []
  all_labels_val = []
  all_mean_val_preds = []
  all_mean_val_bjds = []
  all_mean_val_labels = []
  all_mean_val_bjds = []
  avg_list = []

  for index in range(0, len(VAL_FILE_NAME_LIST)):
    TRAIN_FILE_NAME = TRAIN_FILE_NAME_LIST[index]
    VAL_FILE_NAME = VAL_FILE_NAME_LIST[index]
    train_X, train_Y, train_bjd  = load_dataset_ridge.load_dataset_ridge(TRAIN_FILE_NAME)
    val_X, val_Y, val_bjd = load_dataset_ridge.load_dataset_ridge(VAL_FILE_NAME)

    pred_run_val = []
    labels_run_val = []
    bjd_run_val = []
    for k in range(0,10):
      model = Ridge(alpha=alpha).fit(train_X, train_Y)
      val_pred_Y = model.predict(val_X)
      pred_run_val.append(val_pred_Y)
      labels_run_val.append(val_Y)
      bjd_run_val.append(val_bjd)
      rms_avg = np.sqrt(np.mean(np.square(val_Y -val_pred_Y)))
      rms_avg_list.append(rms_avg)
      if verbose == True:
        print(model)
        print("________________________")
        print("Cross-val number: "+str(index+1)+", Run number: "+str(k+1))
        print("rms: "+str(rms_avg))
      else:
        continue
    mean_val_preds = np.mean(pred_run_val, axis=0)
    mean_val_labels = np.mean(labels_run_val, axis=0)
    mean_val_bjds = np.mean(bjd_run_val, axis=0)
    all_mean_val_preds.append(mean_val_preds.tolist())
    all_mean_val_labels.append(mean_val_labels.tolist())
    all_mean_val_bjds.append(mean_val_bjds.tolist())
  avg = np.mean(rms_avg_list)
  avg_list.append(avg)
  print("________________________")
  print("average rms = "+str(avg)+" m/s")

  #flatten the lists
  all_mean_val_preds = [item for sublist in all_mean_val_preds for item in sublist]
  all_mean_val_labels = [item for sublist in all_mean_val_labels for item in sublist]
  all_mean_val_bjds = [item for sublist in all_mean_val_bjds for item in sublist]
  
  return all_mean_val_preds, all_mean_val_labels, all_mean_val_bjds, avg_list, alpha
