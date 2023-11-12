#### Reviewer EBSh
Q1: I think this is an explicit assumption on the prior distribution, which contradicts to $\spadesuit$ how can we obtain a tractable estimate for $\Phi (k)$ without presuming any particular prior distribution of $\hat{p}_{\boldsymbol{\theta}}\left(\mathbf{z}|k \right)$

A1: 

#### Reviewer WvTp
Q1: Clarify “Without loss of generality, we employ latent features z extracted from deep models as a surrogate for the original high-dimensional raw data x. This is because z is deterministic within the post-hoc framework”

A1: 

Q2: Discuss the role of Deep generative models for flexible density estimation in OOD detection

A2:

Q3: Tell more about why learning the natural parameter of an Exp. Family intractable

A3:

#### Reviewer LYsh
Q1: theoretical justification on the proposed method

A1:

Q2: Comparing with prior work regarding the technical novelty

A2:

Q3: List out all the assumptions made

A3:

#### Reviewer MqR8
We thank the reviewer for the valuable and constructive comments and we have updated the submission accordingly. Please kindly find the detailed responses below.

Q1: The main search problem of optimal coefficient for OOD scoring is remained as a hyperparameter search, which may constrain the practicality of the proposed score and How to choose $p$.

A1.1: Thank you for your advice. We believe that it is necessary to find the optimal norm coefficient $p$ since the feature distribution of different datasets produced by different network architectures is not necessarily same as each other. Therefore, it is not reasonable to use a universal norm coefficient for all datasets. We also find that SOTA post-hoc OOD detection methods [a,b,c,d,e,f] come with (one or more) hyper-parameter searching as well, where their searched hyper-parameters vary across datasets. 

A1.2: We use a subset of Tiny imagenet as the auxiliary OOD data. We remove those data whose labels coincide with ID and test-time OOD cases. The optimal p is selected from 1 and 3. We will add this details in the appendix. Please refer to Section A.3 in the revision for details.

[a] Extremely simple activation shaping for out-of-distribution detection. ICLR 2023

[b] LINe: Out-of-Distribution Detection by Leveraging Important Neurons. CVPR 2023

[c] Out-of-distribution detection with deep nearest neighbors. ICML 2022.

[d] Dice: Leveraging sparsification for out-of-distribution detection. ECCV 2022
 
[f] Out-of-Distribution Detection via Conditional Kernel Independence Model. NeurIPS 2022

Q2: For long-tailed OOD detection when the ID training data exhibits an imbalanced class distribution, I guess the accuracy of the importance sampling-based estimation may decrease given a limited number of tail-class predictions. Elaborate why their method still outperforms in the long-tailed scenarios.

A2:  Thanks for your advice. We agree with your insightful intuition. The table below shows that, compared with (II) estimation based on the full CIFAR-100 training dataset, estimation based on the long-tailed version of the CIFAR-100 training dataset (I) will result in worse OOD detection performance. However, this does not prevent our method from outperforming in this scenario, which implies the robustness and flexibility of the importance sampling-based estimation in our method. We have revised the submission accordingly by adding a discussion on the experiment results of Table 4. Please refer to Section 4.4.2 in the revision for details.

|      Baseline  |      SVHN    |              |      LSUN    |              |      iSUN    |              |     Textures    |              |     Places       |              |     Average    |              |
|:--------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:---------------:|:------------:|:----------------:|:------------:|:--------------:|:------------:|
|                |     FPR95    |     AUROC    |     FPR95    |     AUROC    |     FPR95    |     AUROC    |       FPR95     |     AUROC    |       FPR95      |     AUROC    |      FPR95     |     AUROC    |
|       (I)      |     40.16    |     91.00    |     45.72    |     87.64    |     41.89    |     90.42    |       40.50     |     86.80    |       91.74      |     58.44    |      52.00     |     82.86    |
|       (II)     |     35.66    |     91.51    |     42.40    |     89.81    |     40.41    |     90.59    |       34.54     |     88.90    |       89.28      |     62.84    |      48.46     |     84.73    |
