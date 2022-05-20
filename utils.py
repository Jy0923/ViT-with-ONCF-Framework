import os
import re
import glob
import torch
import random
import argparse
import numpy as np
from pathlib import Path


def arg_parse():
    """
    parse arguments from a command
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()

    return args


def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def remove_old_files(target_dir, thres = 3):
    """
    remove old pt files from a target directory
    - Args
        target_dir: a directory to remove files from
        thres: number of files to be remained
    """
    files = sorted(os.listdir(target_dir), key=lambda x: os.path.getctime(os.path.join(target_dir, x)))
    files = [os.path.join(target_dir, f) for f in files if f.endswith(".pt")]

    if len(files) <= 1:
        print("No Files to Remove")
        return 

    for i in range(0, len(files)-thres):
        os.remove(files[i])


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)