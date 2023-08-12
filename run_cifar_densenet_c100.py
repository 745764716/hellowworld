import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from numpy.linalg import norm
import torch

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# ours + spw
cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.npy"
feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
feat_log, score_log = feat_log.T, score_log.T
class_num = score_log.shape[1]

cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.npy"
feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val, score_log_val = feat_log_val.T, score_log_val.T

print(np.mean(np.argmax(score_log_val, axis=1) == label_log_val))

ood_feat_log_all = {}
ood_score_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
    ood_feat_log_all[ood_dataset] = ood_feat_log
    ood_score_log_all[ood_dataset] = ood_score_log


normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(x[:, range(282, 624)]) # Last Layer only

t1 = 7
t2 = 4
r = 1

ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)

val_score = softmax(score_log_val/t2, axis=1)*r

food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

score_log_exp = np.zeros_like(score_log)
class_protytypes = np.zeros((class_num, ftrain.shape[1]))

for k in range(class_num):
    mask_k = (label_log == k)
    mask_k_reverse = 1 - mask_k
    k_protype = normalizer(np.sum(ftrain*mask_k[:, np.newaxis], axis=0))
    k_protype = np.asarray(k_protype)
    class_protytypes[k, :] = k_protype
    score_log_exp[:, k] = np.exp(np.dot(normalizer(ftrain), k_protype)/t1)*mask_k

score_log_val_exp = np.exp((np.dot(normalizer(ftest), class_protytypes.T)/t1) * val_score)
score_log_val_exp = score_log_val_exp/np.sum(score_log_exp, axis=0)
scores_in = np.log(np.sum(score_log_val_exp, axis=1))

all_results = []
for ood_dataset, food in food_all.items():
    sood = softmax(ood_score_log_all[ood_dataset]/t2, axis=1)*r
    food = normalizer(food)
    ood_score_log_exp = np.exp((np.dot(food, class_protytypes.T)/t1) * sood)
    ood_score_log_exp = ood_score_log_exp / np.sum(score_log_exp, axis=0)
    scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1))

    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'ours+spw, t1={t1}, t2={t2}, r={r}')

# ours

cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.npy"
feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
feat_log, score_log = feat_log.T, score_log.T
class_num = score_log.shape[1]

cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.npy"
feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val, score_log_val = feat_log_val.T, score_log_val.T

print(np.mean(np.argmax(score_log_val, axis=1) == label_log_val))

ood_feat_log_all = {}
ood_score_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
    ood_feat_log_all[ood_dataset] = ood_feat_log
    ood_score_log_all[ood_dataset] = ood_score_log

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(x[:, range(282, 624)]) # Last Layer only

ftrain = normalizer(prepos_feat(feat_log))
ftest = normalizer(prepos_feat(feat_log_val))

food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

t1 = 0.02
score_log_exp = np.zeros_like(score_log)
class_protytypes = np.zeros((class_num, ftrain.shape[1]))

for k in range(class_num):
    mask_k = (label_log == k)
    mask_k_reverse = 1 - mask_k
    k_protype = normalizer(np.sum(feat_log[:, range(282, 624)]*mask_k[:, np.newaxis], axis=0)) #none + norm
    k_protype = np.asarray(k_protype)
    class_protytypes[k, :] = k_protype
    score_log_exp[:, k] = np.exp(np.dot(ftrain, k_protype)/t1)*mask_k


score_log_val_exp = np.exp(np.dot(ftest, class_protytypes.T)/t1)
score_log_val_exp = score_log_val_exp/np.sum(score_log_exp, axis=0)

scores_in = np.log(np.sum(score_log_val_exp, axis=1))

all_results = []
for ood_dataset, food in food_all.items():
    food = normalizer(food)
    ood_score_log_exp = np.exp(np.dot(food, class_protytypes.T)/t1)
    ood_score_log_exp = ood_score_log_exp / np.sum(score_log_exp, axis=0)

    scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1))
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'ours, t1={t1}')


# ours + ash

# def ash_s(x, threshold):
#     x = torch.from_numpy(x[:, :, np.newaxis, np.newaxis]).cuda()
#     assert x.dim() == 4
#     assert x.min() <= threshold <= x.max()
#     s1 = x.sum(dim=[1, 2, 3])
#     x[torch.where(x < threshold)] = 0.
#     s2 = x.sum(dim=[1, 2, 3])
#     scale = s1 / s2
#     x = x * torch.exp(scale[:, None, None, None])
#     return x
#
# normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
# t1 = 0.03
# ash_feat = lambda x: ash_s(x, threshold=0.51).cpu().numpy().squeeze()
# prepos_feat = lambda x: np.ascontiguousarray(x[:, range(282, 624)]) # Last Layer only
#
# ftrain = ash_feat(prepos_feat(feat_log))
# ftest = ash_feat(prepos_feat(feat_log_val))
#
# food_all = {}
# for ood_dataset in args.out_datasets:
#     food_all[ood_dataset] = ash_feat(prepos_feat(ood_feat_log_all[ood_dataset]))
#
# score_log_exp = np.zeros_like(score_log)
# class_protytypes = np.zeros((class_num, ftrain.shape[1]))
#
# for k in range(class_num):
#     mask_k = (label_log == k)
#     mask_k_reverse = 1 - mask_k
#     k_protype = normalizer(np.sum(ftrain*mask_k[:, np.newaxis], axis=0)) #none + norm
#     k_protype = np.asarray(k_protype)
#     class_protytypes[k, :] = k_protype
#     score_log_exp[:, k] = np.exp(np.dot(normalizer(ftrain), k_protype)/t1)*mask_k
#
#
# score_log_val_exp = np.exp(np.dot(normalizer(ftest), class_protytypes.T)/t1)
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
# metrics.print_all_results(all_results, args.out_datasets, f'ours, t1={t1}')


# knn
# normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

# clip_feat = lambda x: np.clip(x, a_min=np.min(x.flatten()), a_max=2.75)
# prepos_feat = lambda x: clip_feat(np.ascontiguousarray(x[:, range(282, 624)]))# Last Layer only
# k = 300

# prepos_feat = lambda x: np.ascontiguousarray(x[:, range(282, 624)])
# k = 300
#
# ftrain = normalizer(prepos_feat(feat_log))
# ftest = normalizer(prepos_feat(feat_log_val))
# food_all = {}
# for ood_dataset in args.out_datasets:
#     food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])
#
# similarity_matrix_sorted = np.sort(-np.dot(ftest, ftrain.T), axis=1)
# scores_in = -similarity_matrix_sorted[:, k]
#
# all_results = []
# for ood_dataset, food in food_all.items():
#     food = normalizer(food)
#     scores_ood_test = -np.sort(-np.dot(food, ftrain.T), axis=1)[:, k]
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'knn, k={k}')

# knn + ash
# from models.ash import *
# normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
# ash_feat = lambda x: ash_b(torch.from_numpy(x[:,:,np.newaxis,np.newaxis]).cuda(), percentile=60).cpu().numpy().squeeze()
# prepos_feat = lambda x: np.ascontiguousarray(x[:, range(282, 624)])
#
# ftrain = normalizer(ash_feat(prepos_feat(feat_log)))
# ftest = normalizer(ash_feat(prepos_feat(feat_log_val)))
# food_all = {}
# for ood_dataset in args.out_datasets:
#     food_all[ood_dataset] = normalizer(ash_feat(prepos_feat(ood_feat_log_all[ood_dataset])))
#
# k = 200
#
# similarity_matrix_sorted = np.sort(-np.dot(ftest, ftrain.T), axis=1)
# scores_in = -similarity_matrix_sorted[:, k]
#
# all_results = []
# for ood_dataset, food in food_all.items():
#     food = normalizer(food)
#     scores_ood_test = -np.sort(-np.dot(food, ftrain.T), axis=1)[:, k]
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'knn, k={k}')



# energy
# t1 = 1
# score_log_val = torch.from_numpy(score_log_val).cuda()
# scores_in = torch.logsumexp(score_log_val/t1, dim=1).cpu().numpy()
#
# all_results = []
# for ood_dataset, ood_score_log in ood_score_log_all.items():
#     ood_score_log = torch.from_numpy(ood_score_log).cuda().cuda()
#     scores_ood_test = torch.logsumexp(ood_score_log/t1, dim=1).cpu().numpy()
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'energy, t1={t1}')

# odin
# t1 = 10000
# score_log_val = score_log_val/t1
# score_log_val = score_log_val - np.max(score_log_val, axis=1, keepdims=True)
# score_log_val = np.exp(score_log_val) / np.sum(np.exp(score_log_val), axis=1, keepdims=True)
# scores_in = np.max(score_log_val, 1)
#
# all_results = []
# for ood_dataset, ood_score_log in ood_score_log_all.items():
#     ood_score_log = ood_score_log/t1
#     ood_score_log = ood_score_log - np.max(ood_score_log, axis=1, keepdims=True)
#     ood_score_log = np.exp(ood_score_log) / np.sum(np.exp(ood_score_log), axis=1, keepdims=True)
#     scores_ood_test = np.max(ood_score_log, 1)
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'odin, t1={t1}')

# msp
# score_log_val = score_log_val
# score_log_val = score_log_val - np.max(score_log_val, axis=1, keepdims=True)
# score_log_val = np.exp(score_log_val) / np.sum(np.exp(score_log_val), axis=1, keepdims=True)
# scores_in = np.max(score_log_val, 1)
#
# all_results = []
# for ood_dataset, ood_score_log in ood_score_log_all.items():
#     ood_score_log = ood_score_log_all[ood_dataset]
#     ood_score_log = ood_score_log - np.max(ood_score_log, axis=1, keepdims=True)
#     ood_score_log = np.exp(ood_score_log) / np.sum(np.exp(ood_score_log), axis=1, keepdims=True)
#     scores_ood_test = np.max(ood_score_log, 1)
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'msp')

# ################### Maha #################
# begin = time.time()
#
# ftrain = feat_log
# ftest = feat_log_val
#
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
# prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
# ftrain_ssd = prepos_feat_ssd(ftrain)
# ftest_ssd = prepos_feat_ssd(ftest)
# food_ssd_all = {}
# for ood_dataset in args.out_datasets:
#     food_ssd_all[ood_dataset] = prepos_feat_ssd(ood_feat_log_all[ood_dataset])
#
# inv_sigma_cls = [None for _ in range(class_num)]
# covs_cls = [None for _ in range(class_num)]
# mean_cls = [None for _ in range(class_num)]
# cov = lambda x: np.cov(x.T, bias=True)
#
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
# metrics.print_all_results(all_results, args.out_datasets, 'maha')
# print(time.time() - begin)

