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

Q1: How to choose $p$

A1:

Q2: For long-tailed OOD detection when the ID training data exhibits an imbalanced class distribution, I guess the accuracy of the importance sampling-based estimation may decrease given a limited number of tail-class predictions. elaborate why their method still outperforms in the long-tailed scenarios.

A2: Yes. 

|         Method |      SVHN    |              |      LSUN    |              |      iSUN    |              |     Textures    |              |     Places       |              |     Average    |              |
|:--------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:---------------:|:------------:|:----------------:|:------------:|:--------------:|:------------:|
|                |     FPR95    |     AUROC    |     FPR95    |     AUROC    |     FPR95    |     AUROC    |       FPR95     |     AUROC    |       FPR95      |     AUROC    |      FPR95     |     AUROC    |
|       (a)      |     40.16    |     91.00    |     45.72    |     87.64    |     41.89    |     90.42    |       40.50     |     86.80    |       91.74      |     58.44    |      52.00     |     82.86    |
|       (b)      |     35.66    |     91.51    |     42.40    |     89.81    |     40.41    |     90.59    |       34.54     |     88.90    |       89.28      |     62.84    |      48.46     |     84.73    |
