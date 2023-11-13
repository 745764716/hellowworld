
Dear Area Chairs and Reviewers,

We would like to thank the reviewers again for their constructive and insightful comments, which help us a lot in improving the submission. We have uploaded the revised version and responded to all the reviewers in detail. We believe that the quality of the paper is improved and the contributions are solid. In particular, we would like to highlight some key materials we added:

>1.
>2.
>3.
>4.
>5.
>6.
>7. 

We understand that reviewers are busy during the response period, we would greatly appreciate it if the reviewers can kindly advise if our responses solve their concerns. If there are any other suggestions/questions, we will try our best to provide satisfactory answers. We are looking forward to any further discussion with the reviewers. Thank you for your time.

Best regards,

The authors


#### Reviewer EBSh
We appreciate the constructive suggestions provided by Reviewer EBSh. Our response is as follows.

Q1: I think this is an explicit assumption on the prior distribution, which contradicts $\spadesuit$ ( how can we obtain a tractable estimate for $\Phi (k)$ without presuming any particular prior distribution of $\hat{p}_{\boldsymbol{\theta}}\left(\mathbf{z}|k \right)$ )

A1: Thank you for your concern. We argue that the use of exponential family of distribution doe not contradict to $\spadesuit$ since our method answers $\spadesuit$ by designing an importance sampling-based estimator of $\Phi (k)$. The estimator itself does not rely on any prior knowledge on data distribution and therefore can be ideally applied to any forms of density functions $g_{\boldsymbol{\theta}}(\mathbf{z}, k)$.

#### Reviewer WvTp

We thank Reviewer WvTp for your insightful suggestions. As to the weaknesses and minor issues you pointed out, we took them very seriously, and have updated parts of the paper to improve it. Please see below for details.

Q1: Clarify “Without loss of generality, we employ latent features z extracted from deep models as a surrogate for the original high-dimensional raw data $\mathbf{x}$. This is because $\mathbf{z}$ is deterministic within the post-hoc framework”

A1: We are sorry that our presentation makes you confused. Here, what we intend to express is as follows. The latent feature space can be a suitable surrogate of raw input space for density estimation because 1) the latent feature space is lower-dimensional and therefore computationally much more efficient and 2) the pre-trained encoder $Enc(\cdot)$ is deterministic such that $p(\mathbf{z}|\mathbf{x})=1$ iff $\mathbf{z}=Enc(\mathbf{x})$. We modify our presentation in the revised version.

Q2: Discuss the role of Deep generative models (DGMs) for flexible density estimation in OOD detection

A2: We thank you for bringing deep generative models into our eyes. We add the discussion on DGMs for flexible density estimation in OOD detection in Section A.8 of the revised version. We agree that using DGMs for latent density estimation is, of course, a valid and natural option. However, this practice requires to train DGMs from scratch, therefore potentially violating the original intention of post-hoc OOD detection, i.e., using the pre-trained model at hand for OOD detection without re-training. Wile it is theoretically possible for zero-shot density estimation with pre-trained diffusion models based on Eq.(1) in [a], the computation is intractable due to the integral. Although the authors in [a] use a simplified ELBO for approximation, there is no theoretical guarantee that the ELBO can align with the ID data density not to mention the time-inefficient inference of diffusion models. We will leave this challenge as our future work.

[a] Your Diffusion Model is Secretly a Zero-Shot Classifier. ICCV 2023.
 
Q3: Tell more about why learning the natural parameter of an Exp. Family intractable

A3: Thank you for your advice. We add the elaboration on this point in the revised version. In short, Learning the natural parameter of an Exp. Family is intractable because we need to solve the following equtation where an integral in a high-dimensional space is involved.

$$\psi(\boldsymbol{\eta}_k)=\int \exp (\mathbf{z}^\top\boldsymbol{\eta}\_k)-g\_{\psi}(\mathbf{z}){\rm d}\mathbf{z}$$


#### Reviewer LYsh
We thank Reviewer LYsh for your thorough suggestions and we have updated the submission accordingly. Our response is as follows.

Q1: The second weakness is that the prior work (Morteza & Li, 2022) already proposed a method based on Gaussian assumptions and Mahalanobis distance. The extension in this paper, at least logically, is relatively straightforward, i.e., from Gaussian to Exponential Family and from Mahalaobis distance to Bregman-divergence (an extension). Maybe it's worth adding a section summarizing the paper's technical novelty.

A1: We thank you for your advice. The contributions of our method are summaried in section A.9 of the revision and as follows:

>1. We propose a novel theoretical framework grounded in Bregman divergence to provide a unified perspective on density-based score design, where the Mahalaobis distance used in GEM and MaxMaha is a special case.
>2. We reframe the design of density function as a search for the optimal norm coefficient p. In this way, compared to GEM, MaxMaha and Energy, all of which impose a fixed distributional assumption for all datasets, the distributional assumption is mild, flexible and switchable.

We included the above summary in Section A.9 of the revision.

Q2: List out all the assumptions made

A2: Than you for your advice. The assumptions made in our method are listed in section A.10 of the revision and as below:

>Assumption 1: the uniform class prior on ID classes.

We note that Assumption 1 is also made in prior post-hoc ood detection methods either explicitly or implicitly [a]. Experiments in Section 4.4.2 show that our method still outperforms in long-tailed OOD detection even with Assumption 1.

>Assumption 2:  $g_\varphi(\cdot)$ is a constant function and the cumulant function $\psi(\cdot) = \frac{1}{2}\|\|\cdot\|\|_{p}^{2}$

Assumption 2 made here aims to reduce the complexity of the exponential family distribution. While it is possible to parameterize the exponential family distribution in a more complicated manner, our proposed simple version already performs best on a wide range of datasets and settings.

[a] Detecting Out-of-distribution Data through In-distribution Class Prior. ICML 2023

Q3: theoretical justification on the proposed method

A3: 

#### Reviewer MqR8
We thank reviewer MqR8 for your valuable comments and we have updated the submission accordingly. Please kindly find the detailed responses below. 

Q1: 1) The main search problem of optimal coefficient for OOD scoring is remained as a hyperparameter search, which may constrain the practicality of the proposed score, and 2) How to choose $p$.

A1.1: Thank you for your concern. We believe that it is necessary to find the optimal norm coefficient $p$ since the feature distribution of different datasets produced by different network architectures could not be necessarily same as each other. Therefore, it is not reasonable to use a universal norm coefficient for all datasets. We also observe that SOTA post-hoc OOD detection methods [a,b,c,d,e,f] come with (one or more) hyper-parameter searching as well, where their searched hyper-parameter values vary across datasets. 

A1.2: Simialr to [d], We use a subset of Tiny imagenet as the auxiliary OOD data. We remove those data whose labels coincide with ID cases. The searching space of $p$ is (1,3]. We will add this details in the appendix. Please refer to Section A.3 in the revision for details.

[a] Extremely simple activation shaping for out-of-distribution detection. ICLR 2023

[b] LINe: Out-of-Distribution Detection by Leveraging Important Neurons. CVPR 2023

[c] Out-of-distribution detection with deep nearest neighbors. ICML 2022.

[d] Dice: Leveraging sparsification for out-of-distribution detection. ECCV 2022
 
[f] Out-of-Distribution Detection via Conditional Kernel Independence Model. NeurIPS 2022

Q2: For long-tailed OOD detection when the ID training data exhibits an imbalanced class distribution, I guess the accuracy of the importance sampling-based estimation may decrease given a limited number of tail-class predictions. Elaborate why their method still outperforms in the long-tailed scenarios.

A2:  We agree with your insightful intuition. The table below shows that, compared with estimation based on the full CIFAR-100 training dataset (ii), estimation based on the long-tailed version of the CIFAR-100 training dataset (i) will result in worse OOD detection performance. However, this does not prevent our method from outperforming in this scenario, which implies the robustness and flexibility of the importance sampling-based estimation in our method. We have revised the submission accordingly by adding a discussion on the experiment results of Table 4. Please refer to Section 4.4.2 in the revision for details.

|      Baseline  |      SVHN    |              |      LSUN    |              |      iSUN    |              |     Textures    |              |     Places       |              |     Average    |              |
|:--------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:---------------:|:------------:|:----------------:|:------------:|:--------------:|:------------:|
|                |     FPR95    |     AUROC    |     FPR95    |     AUROC    |     FPR95    |     AUROC    |       FPR95     |     AUROC    |       FPR95      |     AUROC    |      FPR95     |     AUROC    |
|       (i)      |     40.16    |     91.00    |     45.72    |     87.64    |     41.89    |     90.42    |       40.50     |     86.80    |       91.74      |     58.44    |      52.00     |     82.86    |
|       (ii)     |     35.66    |     91.51    |     42.40    |     89.81    |     40.41    |     90.59    |       34.54     |     88.90    |       89.28      |     62.84    |      48.46     |     84.73    |
