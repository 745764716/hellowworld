import os
import time
from util.args_loader import get_args
from util import metrics
import torch
# import faiss
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from numpy.linalg import norm


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(x)# Last Layer only


cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.npy"
feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
class_num = score_log.shape[1]

cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.npy"
feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)


ftrain = normalizer(prepos_feat(feat_log))
ftest = normalizer(prepos_feat(feat_log_val))


ood_feat_log_all = {}
ood_score_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
    ood_feat_log_all[ood_dataset] = ood_feat_log
    ood_score_log_all[ood_dataset] = ood_score_log

food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])


# t1 = 0.06
# score_log_exp = np.zeros_like(score_log)
# class_protytypes = np.zeros((class_num, ftrain.shape[1])).astype(np.float32)
#
# for k in range(class_num):
#     mask_k = (label_log == k)
#
#     # k_protype = normalizer(np.sum(ftrain*mask_k[:, np.newaxis], axis=0)) #norm + norm
#     k_protype = normalizer(np.sum(feat_log*mask_k[:, np.newaxis], axis=0)) #none + norm
#
#     k_protype = np.asarray(k_protype, dtype='float32')
#     class_protytypes[k, :] = k_protype
#     mask_k_reverse = 1-mask_k
#     score_log_exp[:, k] = np.exp(np.dot(ftrain, k_protype)/t1)*mask_k
#
# score_log_val_exp = np.exp(np.dot(ftest, class_protytypes.T)/t1)
# score_log_val_exp = score_log_val_exp/(score_log_val_exp+np.sum(score_log_exp, axis=0))
#
# scores_in = np.log(np.sum(score_log_val_exp, axis=1))
#
# all_results = []
# for ood_dataset, food in food_all.items():
#     food = normalizer(food)
#     ood_score_log_exp = np.exp(np.dot(food, class_protytypes.T)/t1)
#     ood_score_log_exp = ood_score_log_exp / (ood_score_log_exp + np.sum(score_log_exp, axis=0))
#     scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1))
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f't1={t1}')

import torch

t1 = 1500
score_log_val = torch.from_numpy(score_log_val).cuda()
scores_in = torch.logsumexp(score_log_val/t1, dim=1).cpu().numpy()

all_results = []
for ood_dataset, food in food_all.items():
    score_log_ood = torch.from_numpy(ood_score_log_all[ood_dataset]).cuda()
    scores_ood_test = torch.logsumexp(score_log_ood/t1, dim=1).cpu().numpy()
    print((scores_ood_test.shape))

    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'energy, t1={t1}')


# k = 50
#
# similarity_matrix_sorted = np.sort(-np.dot(ftest, ftrain.T), axis=1)
# scores_in = -similarity_matrix_sorted[:, k]
#
#
# all_results = []
# for ood_dataset, food in food_all.items():
#     food = normalizer(food)
#     scores_ood_test = -np.sort(-np.dot(food, ftrain.T), axis=1)[:, k]
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'k={k}')



# #################### KNN score OOD detection #################
#
# index = faiss.IndexFlatL2(ftrain.shape[1])
# index.add(ftrain)
# for K in [50]:
#
#     D, _ = index.search(ftest, K)
#     scores_in = -D[:,-1]
#     all_results = []
#     all_score_ood = []
#     for ood_dataset, food in food_all.items():
#         D, _ = index.search(food, K)
#         scores_ood_test = -D[:,-1]
#         all_score_ood.extend(scores_ood_test)
#         results = metrics.cal_metric(scores_in, scores_ood_test)
#         all_results.append(results)
#
#     metrics.print_all_results(all_results, args.out_datasets, f'KNN k={K}')
#     print()




# #################### SSD+ score OOD detection #################
# begin = time.time()
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
# prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
# ftrain_ssd = prepos_feat_ssd(ftrain)
# ftest_ssd = prepos_feat_ssd(ftest)
# food_ssd_all = {}
# for ood_dataset in args.out_datasets:
#     food_ssd_all[ood_dataset] = prepos_feat_ssd(food_all[ood_dataset])
#
# inv_sigma_cls = [None for _ in range(class_num)]
# covs_cls = [None for _ in range(class_num)]
# mean_cls = [None for _ in range(class_num)]
# cov = lambda x: np.cov(x.T, bias=True)
# for cls in range(class_num):
#     mean_cls[cls] = ftrain_ssd[label_log == cls].mean(0)
#     feat_cls_center = ftrain_ssd[label_log == cls] - mean_cls[cls]
#     inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
#
# def maha_score(X):
#     score_cls = np.zeros((class_num, len(X)))
#     for cls in range(class_num):
#         inv_sigma = inv_sigma_cls[cls]
#         mean = mean_cls[cls]
#         z = X - mean
#         score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
#     return score_cls.max(0)
#
# dtest = maha_score(ftest_ssd)
# all_results = []
# for name, food in food_ssd_all.items():
#     print(f"SSD+: Evaluating {name}")
#     dood = maha_score(food)
#     results = metrics.cal_metric(dtest, dood)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, 'SSD+')
# print(time.time() - begin)

