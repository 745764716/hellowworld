from distfit import distfit
import numpy as np
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt

# t = [0.01, 0.02, 0.05, 0.1, 1]
# fig, axes = plt.subplots(1, 5, figsize=(10, 5), sharey=True, dpi=1000)
# for t1 in t:
#     scores_in, scores_ood_test, results = np.load(f"./results_{t1}.npy", allow_pickle=True)
#     sns.distplot(scores_in, color="dodgerblue", ax=axes[t.index(t1)], axlabel='ID')
#     sns.distplot(scores_ood_test, color="deeppink", ax=axes[t.index(t1)], axlabel='OOD')
#
# plt.title('Iris Histogram')
# plt.show()


# from pylab import *
# 
# sns.set(rc={"figure.figsize": (8, 4)}); np.random.seed(0)
# x = np.random.randn(100)
# 
# subplot(2,3,1)
# scores_in, scores_ood_test, results = np.load(f"./results_0.01.npy", allow_pickle=True)
# ax = sns.distplot(scores_in, color="dodgerblue", rug=False, hist=False, label='ID')
# ax = sns.distplot(scores_ood_test, color="deeppink", rug=False, hist=False, label='OOD')
# 
# subplot(2,3,2)
# scores_in, scores_ood_test, results = np.load(f"./results_0.02.npy", allow_pickle=True)
# ax = sns.distplot(scores_in, color="dodgerblue", rug=False, hist=False, label='ID')
# ax = sns.distplot(scores_ood_test, color="deeppink", rug=False, hist=False, label='OOD')
# 
# subplot(2,3,3)
# scores_in, scores_ood_test, results = np.load(f"./results_0.1.npy", allow_pickle=True)
# ax = sns.distplot(scores_in, color="dodgerblue", rug=False, hist=False, label='ID')
# ax = sns.distplot(scores_ood_test, color="deeppink", rug=False, hist=False, label='OOD')
# 
# subplot(2,3,4)
# scores_in, scores_ood_test, results = np.load(f"./results_0.5.npy", allow_pickle=True)
# ax = sns.distplot(scores_in, color="dodgerblue", rug=False, hist=False, label='ID')
# ax = sns.distplot(scores_ood_test, color="deeppink", rug=False, hist=False, label='OOD')
# 
# subplot(2,3,5)
# scores_in, scores_ood_test, results = np.load(f"./results_1.npy", allow_pickle=True)
# ax = sns.distplot(scores_in, color="dodgerblue", rug=False, hist=False, label='ID')
# ax = sns.distplot(scores_ood_test, color="deeppink", rug=False, hist=False, label='OOD')
# 
# # subplot(2,3,6)
# # ax = sns.kdeplot(x, shade=True, color="r")
# 
# ax.set_xlim(-100, 0);
# plt.show()

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
#
# np.random.seed(1)
#
# m, n = 10, 2
# # generate random data, with arbitrary covariance
# X = np.random.multivariate_normal([0 for i in range(n)], np.array([[1, 0.8], [0.8, 1]]), size=m)
# X = X - X.mean(axis=0) # cener the data
#
# X = X[X[:, 0].argsort(), :] # sort samples for plotting with increasing colors
#
# C = X.T @ X / m #
#
# eig_vals, eig_vecs = np.linalg.eig(C)
# D = np.diag(eig_vals) # eig_vals is a vector, but we want a matrix
# P = eig_vecs
#
# D_m12 = np.diag(np.diag(D)**(-0.5)) # 'm12' for power(minus 1/2)
#
# W_ZCA = P @ D_m12 @ P.T
#
# x = (W_ZCA @ X.T).T
# print((W_ZCA @ X.mean(axis=0).T).T)
# print(x.mean(axis=0))
#
# C = x.T @ x / m
# print(x)
# print(C)

# import torch
# checkpoint = torch.load(
#     "./checkpoints/{in_dataset}/{model_arch}/checkpoint_200_{imb_factor}.pth".format(
#         in_dataset="CIFAR-10-LT", model_arch="resnet32", imb_factor=200),
#     map_location='cpu')

# print(checkpoint)
# checkpoint2 = {'state_dict': {}}
#
# for key, value in checkpoint['state_dict_model'].items():
#     checkpoint2["state_dict"][key.replace("module.", "")] = value
#
# for key, value in checkpoint['state_dict_classifier'].items():
#     checkpoint2["state_dict"][key.replace("module.", "")] = value
#
# for key, value in checkpoint['state_dict_best']["feat_model"].items():
#     print(key)
#
# for key, value in checkpoint['state_dict_best']["classifier"].items():
#     print(key)

feat_log_val = np.load(f"val.npy")
print(feat_log_val.shape)