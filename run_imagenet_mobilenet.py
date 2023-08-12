import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import numpy as np
from scipy import stats
from scipy.special import softmax
from models.mobilenetv2 import mobilenet_v2

args = get_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ours + spw
class_num = 1000
t1 = 0.1
t2 = 5
r = 40
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
prepos_feat = lambda x: np.ascontiguousarray(x).astype(np.float32)

feat_log_val = np.load(f"cache/imagenet_feature_mobilenet_v2_sorted/val.npy")
ftest = normalizer(prepos_feat(feat_log_val))

score_log_exp = []
class_protytypes = np.zeros((class_num, 1280))

model = mobilenet_v2(num_classes=1000, pretrained=True).classifier
model.cuda()
model.eval()

for k in range(class_num):
    k_ftrain = np.load(f"cache/imagenet_feature_mobilenet_v2_sorted/class_{k}.npy".format(k=k))
    k_protype = normalizer(prepos_feat(np.mean(k_ftrain, axis=0)))
    class_protytypes[k, :] = k_protype
    score_log_exp.append(np.sum(np.exp(np.dot(normalizer(k_ftrain), k_protype) / t1)))

val_score = softmax(model(torch.from_numpy(prepos_feat(feat_log_val)).cuda()).data.cpu().numpy()/t2, axis=1)*r

score_log_exp = np.asarray(score_log_exp)
score_log_val_exp = np.exp((np.dot(ftest, class_protytypes.T)/t1)*val_score)
score_log_val_exp = score_log_val_exp/score_log_exp

scores_in = np.log(np.sum(score_log_val_exp, axis=1))

food_all = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}

for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 1280))
    food_all[ood_dataset] = prepos_feat(ood_feat_log)

all_results = []
for ood_dataset, food in food_all.items():

    sood = softmax(model(torch.from_numpy(food).cuda()).data.cpu().numpy()/t2, axis=1)*r
    food = normalizer(food)

    ood_score_log_exp = np.exp((np.dot(food, class_protytypes.T)/t1)*sood)
    ood_score_log_exp = ood_score_log_exp / score_log_exp

    scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1))
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'ours + spw, t1={t1},t2={t2},r={r}')


# ours
class_num = 1000
t1 = 0.02
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
prepos_feat = lambda x: np.ascontiguousarray(x).astype(np.float32)

feat_log_val = np.load(f"cache/imagenet_feature_mobilenet_v2_sorted/val.npy")
ftest = normalizer(prepos_feat(feat_log_val))

score_log_exp = []
class_protytypes = np.zeros((class_num, 1280))

for k in range(class_num):
    k_ftrain = np.load(f"cache/imagenet_feature_mobilenet_v2_sorted/class_{k}.npy".format(k=k))
    k_protype = normalizer(prepos_feat(np.mean(k_ftrain, axis=0)))
    class_protytypes[k, :] = k_protype
    score_log_exp.append(np.sum(np.exp(np.dot(normalizer(k_ftrain), k_protype) / t1)))
score_log_exp = np.asarray(score_log_exp)
score_log_val_exp = np.exp(np.dot(ftest, class_protytypes.T)/t1)
# score_log_val_exp = score_log_val_exp/(score_log_val_exp + score_log_exp)
score_log_val_exp = score_log_val_exp/score_log_exp

scores_in = np.log(np.sum(score_log_val_exp, axis=1))


food_all = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}

for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 1280))
    food_all[ood_dataset] = normalizer(prepos_feat(ood_feat_log))

all_results = []
for ood_dataset, food in food_all.items():
    ood_score_log_exp = np.exp(np.dot(food, class_protytypes.T)/t1)
    # ood_score_log_exp = ood_score_log_exp / (ood_score_log_exp + score_log_exp)
    ood_score_log_exp = ood_score_log_exp / score_log_exp

    scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1))
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'ours, t1={t1}')
