
from tf_util import config_util
from tf_util import configdict
from rv_net import data_HARPS_N
import tensorflow as tf
import numpy as np


def load_dataset_ridge(filename):
  dataset_hparams = configdict.ConfigDict(dict(
    ccf_feature_name="Rescaled CCF_residuals_cutoff", #CCF_residuals
    label_feature_name= "activity signal",#"RV",
    label_feature_name2= "BJD",
    batch_size=300,
    label_rescale_factor=1000,
  ))
  dataset = data_HARPS_N.DatasetBuilder(filename, dataset_hparams, tf.estimator.ModeKeys.EVAL)()
  batches = list(dataset)
  ccf_data, labels, bjds = zip(*[(batch["ccf_data"], batch["label"], batch["bjd"]) for batch in batches])
  ccf_data = np.concatenate(ccf_data)
  labels = np.concatenate(labels)
  bjds = np.concatenate(bjds)
  assert len(ccf_data.shape) == 2
  assert len(labels.shape) == 1
  assert len(bjds.shape) == 1
  assert ccf_data.shape[0] == labels.shape[0]
  print("Read dataset with {} examples".format(labels.shape[0]))
  return ccf_data, labels, bjds
