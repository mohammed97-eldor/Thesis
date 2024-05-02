import numpy as np
from tqdm import tqdm

def evaluate_instance(gt_lst, outputs):

  instances = len(gt_lst)
  predictions = outputs.shape[0]
  eval = np.ones((instances, predictions)) * -1

  for i in range(instances):
    for j in range(predictions):
      intersection = np.logical_and(gt_lst[i], outputs[j, :, :]).sum()
      union = np.logical_or(gt_lst[i], outputs[j, :, :]).sum()
      eval[i, j] = intersection/union
  return eval

def FP_FN_0(output_dict: dict):

  out = 0
  for img in tqdm(output_dict.keys()):
    con_mat_shape = output_dict[img].shape
    if con_mat_shape[0] != con_mat_shape[1]:
      continue
    else:
      tps = output_dict[img] > 0.5
      if (tps.sum(axis = 0) == np.ones(con_mat_shape[0])).all() and (tps.sum(axis = 1) == np.ones(con_mat_shape[0])).all():
        out += 1
  return out

def TP(output_dict: dict):

    out = 0
    for img in tqdm(output_dict.keys()):
      cf_mat = output_dict[img].copy()
      con_mat_shape = cf_mat.shape
      for i in range(con_mat_shape[1]):
        idx = cf_mat[:, i].argmax()
        if cf_mat[idx, i] > 0.5:
          out += 1
          cf_mat[idx, :] = 0
          cf_mat[:, i] = 0
    return out

def FP(output_dict: dict):

  out = 0
  for img in tqdm(output_dict.keys()):
      cf_mat = output_dict[img].copy()
      con_mat_shape = cf_mat.shape
      for i in range(con_mat_shape[1]):
        if (cf_mat[:, i] > 0.5).sum() == 0:
          out += 1
  return out

def TP_0(output_dict: dict):

  out = 0
  for img in tqdm(output_dict.keys()):
      cf_mat = output_dict[img].copy()
      con_mat_shape = cf_mat.shape
      flag = True
      for i in range(con_mat_shape[1]):
        if (cf_mat[:, i] > 0.5).sum() != 0:
          flag = False
      if flag:
        out += 1
  return out