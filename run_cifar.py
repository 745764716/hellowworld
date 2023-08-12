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

cache_name = f"cache/{args.in_dataset}_train_{args.name}_{args.imb_factor}_in_alllayers.npy"
feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
feat_log, score_log = feat_log.T.astype(np.float64), score_log.T.astype(np.float64)
class_num = score_log.shape[1]

class_nums = np.zeros(class_num)
for k in range(class_num):
    class_nums[k] = np.sum(label_log == k)

cache_name = f"cache/{args.in_dataset}_val_{args.name}_{args.imb_factor}_in_alllayers.npy"
feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val, score_log_val = feat_log_val.T.astype(np.float64), score_log_val.T.astype(np.float64)

print(np.mean(np.argmax(score_log_val, axis=1)==label_log_val))
print(np.mean(np.argmax(score_log, axis=1)==label_log))

ood_feat_log_all = {}
ood_score_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.imb_factor}_{args.name}_out_alllayers.npy"
    ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float64), ood_score_log.T.astype(np.float64)
    ood_feat_log_all[ood_dataset] = ood_feat_log
    ood_score_log_all[ood_dataset] = ood_score_log

food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])


# ftrain = normalizer(prepos_feat(feat_log))
# ftest = normalizer(prepos_feat(feat_log_val))
# t1 = 0.01
# score_log_exp = np.zeros_like(score_log)
# class_protytypes = np.zeros((class_num, ftrain.shape[1])).astype(np.float32)
#
# for k in range(class_num):
#     mask_k = (label_log == k)
#     k_protype = normalizer(np.sum(prepos_feat(feat_log)*mask_k[:, np.newaxis], axis=0)) #none + norm
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


# knn

# ftrain = normalizer(prepos_feat(feat_log))
# ftest = normalizer(prepos_feat(feat_log_val))
# k = 40
# # k = 50
# similarity_matrix_sorted = np.sort(-np.dot(ftest, ftrain.T), axis=1)
# scores_in = -similarity_matrix_sorted[:, k]
# all_results = []
# for ood_dataset, food in food_all.items():
#     food = normalizer(food)
#     scores_ood_test = -np.sort(-np.dot(food, ftrain.T), axis=1)[:, k]
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, f'k={k}')

# maxmaha

# begin = time.time()
#
# ftrain = prepos_feat(feat_log)
# ftest = prepos_feat(feat_log_val)
#
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
#
# prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
#
# ftrain_ssd = prepos_feat_ssd(ftrain)
# ftest_ssd = prepos_feat_ssd(ftest)
#
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

# GEM

# begin = time.time()
#
# ftrain = prepos_feat(feat_log)
# ftest = prepos_feat(feat_log_val)
#
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
#
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
# def gem_score(X):
#     score_cls = np.zeros((class_num, len(X)))
#     for cls in range(class_num):
#         inv_sigma = inv_sigma_cls[cls]
#         mean = mean_cls[cls]
#         z = X - mean
#         score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
#     return torch.logsumexp(torch.from_numpy(score_cls.T,).cuda(), dim=1).cpu().numpy()
#
# dtest = gem_score(ftest_ssd)
# all_results = []
#
# for name, food in food_ssd_all.items():
#     print(f"SSD+: Evaluating {name}")
#     dood = gem_score(food)
#     results = metrics.cal_metric(dtest, dood)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, 'gem')
# print(time.time() - begin)



# msp/odin

# t1 = 1000
# W_in = 1-np.dot(normalizer(softmax(score_log_val, axis=1)), normalizer(class_nums))
# score_log_val = score_log_val/t1
# score_log_val = score_log_val - np.max(score_log_val, axis=1, keepdims=True)
# score_log_val = np.exp(score_log_val) / np.sum(np.exp(score_log_val), axis=1, keepdims=True)
# if t1 == 1:
#     # scores_in = np.max(score_log_val - class_nums[np.newaxis, :] / feat_log.shape[0], 1)
#     scores_in = np.max(score_log_val, 1)
# else:
#     # W_in = 1
#     scores_in = np.max(score_log_val, 1) * W_in
#
# all_results = []
# for ood_dataset, ood_score_log in ood_score_log_all.items():
#     ood_score_log = ood_score_log_all[ood_dataset]
#     W_ood = 1 - np.dot(normalizer(softmax(ood_score_log, axis=1)), normalizer(class_nums))
#     ood_score_log = ood_score_log/t1
#     ood_score_log = ood_score_log - np.max(ood_score_log, axis=1, keepdims=True)
#     ood_score_log = np.exp(ood_score_log) / np.sum(np.exp(ood_score_log), axis=1, keepdims=True)
#     if t1 == 1:
#         scores_ood_test = np.max(ood_score_log - class_nums[np.newaxis, :]/feat_log.shape[0], 1)
#         # scores_ood_test = np.max(ood_score_log, 1)
#     else:
#         # W_ood = 1
#         scores_ood_test = np.max(ood_score_log, 1) * W_ood
#     results = metrics.cal_metric(scores_in, scores_ood_test)
#     all_results.append(results)
# if t1 == 1:
#     metrics.print_all_results(all_results, args.out_datasets, f'msp')
# else:
#     metrics.print_all_results(all_results, args.out_datasets, f'odin, t1={t1}')

# energy

t1 = 1
W_in = 1-np.dot(normalizer(softmax(score_log_val, axis=1)), normalizer(class_nums))
# W_in = 1
score_log_val = torch.from_numpy(score_log_val).cuda()
scores_in = torch.logsumexp(score_log_val/t1, dim=1).cpu().numpy()*W_in

all_results = []
for ood_dataset, ood_score_log in ood_score_log_all.items():
    W_ood = 1 - np.dot(normalizer(softmax(ood_score_log, axis=1)), normalizer(class_nums))
    # W_ood = 1
    ood_score_log = torch.from_numpy(ood_score_log).cuda()
    scores_ood_test = torch.logsumexp(ood_score_log/t1, dim=1).cpu().numpy()*W_ood
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'energy, t1={t1}')