Dear Area Chairs and Reviewers,

We would like to thank the reviewers again for their constructive and insightful comments, which help us a lot in improving the submission. We have uploaded the revised version and responded to all the reviewers in detail. We believe that the quality of the paper is improved and the contributions are solid. In particular, we would like to highlight some key materials we added:

>1. improve the confusing presentations and the use of notations .
>2. Discussion on Deep generative models (DGMs) as our related work
>3. elaboration on why learning the natural parameter of an Exp. Family is intractable
>4. contribution summary
>5. list of assumptions made in our method
>6. theoretical justification of our method
>7. The searching Strategy of the norm coefficient $p$
>8. More dicussion on experiment results of long-tailed OOD detection

We understand that reviewers are busy during the response period, we would greatly appreciate it if the reviewers can kindly advise if our responses solve their concerns. If there are any other suggestions/questions, we will try our best to provide satisfactory answers. We are looking forward to any further discussion with the reviewers. Thank you for your time.

Best regards,

The authors


#### Reviewer EBSh
We appreciate the constructive suggestions provided by Reviewer EBSh. As to the weaknesses you pointed out, we took them very seriously. Our response is as follows.

Q1: I think this is an explicit assumption on the prior distribution, which contradicts $\spadesuit$ ( how can we obtain a tractable estimate for $\Phi (k)$ without presuming any particular prior distribution of $\hat{p}_{\boldsymbol{\theta}}\left(\mathbf{z}|k \right)$ )

A1: Thank you for your concern. We argue that the use of the exponential family of distribution does not contradict to $\spadesuit$ since our method answers $\spadesuit$ by designing an importance sampling-based estimator of $\Phi (k)$. The estimator itself does not rely on any prior knowledge of data distribution and therefore can be ideally applied to any forms of density functions $g_{\boldsymbol{\theta}}(\mathbf{z}, k)$.



#### Reviewer WvTp

We thank Reviewer WvTp for your insightful suggestions. As to the weaknesses and minor issues you pointed out, we took them very seriously, and have updated parts of the paper to improve it. Please see below for details.

Q1: Clarify “Without loss of generality, we employ latent features z extracted from deep models as a surrogate for the original high-dimensional raw data $\mathbf{x}$. This is because $\mathbf{z}$ is deterministic within the post-hoc framework”

A1: We are sorry that our presentation makes you confused. Here, what we intend to express is as follows. Since the pre-trained encoder $Enc(\cdot)$ is deterministic such that $p(\mathbf{z}|\mathbf{x})=1$ iff $\mathbf{z}=Enc(\mathbf{x})$, we consider the latent feature space as a suitable surrogate of the raw data space for density estimation since the former is lower-dimensional and therefore computationally much more efficient than the latter.

Q2: Discuss the role of Deep generative models (DGMs) for flexible density estimation in OOD detection

A2: We thank you for bringing deep generative models into our eyes. We add the discussion on DGMs for flexible density estimation in OOD detection in Section A.8 of the revised version. We agree that using DGMs for latent density estimation is, of course, a valid and natural option. However, this practice requires to train DGMs from scratch, therefore potentially violating the original intention of post-hoc OOD detection, i.e., using the pre-trained model at hand for OOD detection without re-training. Wile it is theoretically possible for zero-shot density estimation with pre-trained diffusion models based on Eq.(1) in [a], the computation is intractable due to the integral. Although the authors in [a] use a simplified ELBO for approximation, there is no theoretical guarantee that the ELBO can align with the ID data density not to mention the time-inefficient inference of diffusion models. We will leave this challenge as our future work.

[a] Your Diffusion Model is Secretly a Zero-Shot Classifier. ICCV 2023.
 
Q3: Tell more about why learning the natural parameter of an Exp. Family is intractable

A3: Thank you for your advice. We add the elaboration on this point in the revised version. In short, Learning the natural parameter of an Exp. Family is intractable because we need to solve the following equtation where an integral in a high-dimensional space is involved.

$$\psi(\boldsymbol{\eta}_k)=\int \exp (\mathbf{z}^\top\boldsymbol{\eta}\_k)-g\_{\psi}(\mathbf{z}){\rm d}\mathbf{z}$$


#### Reviewer LYsh
We thank Reviewer LYsh for your thorough suggestions. As to the weaknesses and minor issues you pointed out, we took them very seriously, and have updated parts of the paper to improve it. Our response is as follows.

Q1: The second weakness is that the prior work (Morteza & Li, 2022) already proposed a method based on Gaussian assumptions and Mahalanobis distance. The extension in this paper, at least logically, is relatively straightforward, i.e., from Gaussian to Exponential Family and from Mahalaobis distance to Bregman-divergence (an extension). Maybe it's worth adding a section summarizing the paper's technical novelty.

A1: We thank you for your advice. The contributions of our method are summaried in section A.9 of the revision and as follows:

>1. We propose a novel theoretical framework grounded in Bregman divergence to provide a unified perspective on density-based score design, where the Mahalaobis distance used in GEM and MaxMaha is a special case.
>2. We reframe the design of density function as a search for the optimal norm coefficient p. In this way, compared to GEM, MaxMaha and Energy, all of which impose a fixed distributional assumption for all datasets, the distributional assumption is mild, flexible and switchable.

We included the above summary in Section A.9 of the revision.

Q2: List out all the assumptions made

A2: Than you for your advice. The assumptions made in our method are listed in section A.10 of the revision and as below:

>**Assumption 1.** the uniform class prior on ID classes.

We note that Assumption 1 is also made in prior post-hoc ood detection methods either explicitly or implicitly [a]. Experiments in Section 4.4.2 show that our method still outperforms in long-tailed OOD detection even with Assumption 1.

>**Assumption 2.**  $g_\varphi(\cdot)$ is a constant function and the cumulant function $\psi(\cdot) = \frac{1}{2}\|\|\cdot\|\|_{p}^{2}$

Assumption 2 made here aims to reduce the complexity of the exponential family distribution. While it is possible to parameterize the exponential family distribution in a more complicated manner, our proposed simple version suffices to perform well.

[a] Detecting Out-of-distribution Data through In-distribution Class Prior. ICML 2023

Q3: theoretical justification

A3: Here, same as KNN [b], we provice a theoretical analysis on our method by analyzing the average OOD detection performance of our algorithm in Section A.11 and as follows. 
>**Setup.** We consider the OOD detection task as a special binary classification task, where the negative samples (OOD) are only available in the testing stage. We assume the input is from feature embeddings space $\mathcal{Z}$ and the labeling set $\mathcal{G}= \left\lbrace 0(OOD),1(ID) \right\rbrace$. In the inference stage, the testing set $\left\lbrace(\mathbf{z},g)\right\rbrace$ is drawn i.i.d. from $P_{\mathcal{Z} \mathcal{G}}$. Denote the marginal distribution on $\mathcal{Z}$ as $\mathcal{P}$, We adopt the Huber contamination model [] to model the fact that we may encounter both ID and OOD data in test time:
$$\mathcal{P}=(1-\epsilon)\mathcal{P}_I+\epsilon \mathcal{P}_O$$
where $\mathcal{P}_I$ and $\mathcal{P}_o$ are the underlying distributions of feature embeddings for ID and OOD data, respectively, and $\epsilon$ is a constant controlling the fraction of OOD samples in testing. We use $\mathcal{p}_I(\mathbf{z})$ and $\mathcal{p}_o(\mathbf{z})$ to denote the probability density function, where $\mathcal{p}_I(\mathbf{z}) = \mathcal{p}(z|g = 1)$ and $\mathcal{p}_o(\mathbf{z}) = \mathcal{p}(z|g = 0)$. It is natural to approximate $\mathcal{p}_I(\mathbf{z})$ as $\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})$ in Eq.(11). Following KNN, we apprximate OOD distribution by modeling OOD data with an equal chance to appear outside of the high-density region of ID data, i.e., $\mathcal{p}_o(\mathbf{z})=c_o \mathbf{I}\left\lbrace\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})<\beta\right\rbrace$. Given the fact that $\mathcal{p}_o(\mathbf{z})=0$ if $z \in \left\lbrace Enc(\mathbf{x})|\mathbf{x}\in X\_I \right\rbrace$, the empirical esitimator of $\beta$ is given by $\hat\beta = \min\_{(\mathbf{x},\mathbf{y})\in \mathcal{D}\_{\rm in}} \mathcal{p}\_{\boldsymbol{\theta}}(Enc(\mathbf{x}))$ with $Enc(\cdot)$ as the encoder of a pre-train model.

>**Main result 1.** By the Bayesian rule, the probability of $\mathbf{z}$ being ID data is:

$$\begin{aligned}
\mathcal{p}(g = 1|\mathbf{z}) &= \frac{\mathcal{p}(\mathbf{z}|g = 1)\cdot\mathcal{p}(g = 1)}{\mathcal{p}(\mathbf{z}|g = 1)\cdot\mathcal{p}(g = 1)+\mathcal{p}(\mathbf{z}|g = 0)\cdot\mathcal{p}(g = 0)} \\
&= \frac{(1-\epsilon)\mathcal{p}(\mathbf{z}|g = 1)}{(1-\epsilon)\mathcal{p}(\mathbf{z}|g = 1)+\epsilon\mathcal{p}(\mathbf{z}|g = 0)} \\
&\approx \frac{(1-\epsilon)\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})}{(1-\epsilon)\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})+\epsilon c_o\mathbf{I}\left\lbrace\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})<\hat\beta\right\rbrace}
\end{aligned}$$

>**Main result 2.** Given a pre-defined threshold $\lambda$, it can be easily checked that:

$$\mathbf{I}\left\lbrace\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})\ge\lambda\right\rbrace\approx \mathbf{I}\left\lbrace\mathcal{p}(g = 1|z)\ge\frac{(1-\epsilon)\lambda}{(1-\epsilon)\lambda+\epsilon c_o\mathbf{I}(\lambda<\hat\beta)}\right\rbrace $$

[b] Out-of-distribution detection with deep nearest neighbors. ICML 2022.

[c] Robust estimation of a location parameter. Annals of Mathematical Statistics, 1964.
 
Q4：it would be good if the authors could make the notations more distinguishable.

A4: Thank you for your advice. we have improve our use of notations in the revision.

#### Reviewer MqR8
We thank reviewer MqR8 for your valuable comments. We updated the submission accordingly. Please kindly find the detailed responses below. 

Q1: 1) The main search problem of optimal coefficient for OOD scoring remains as a hyperparameter search, which may constrain the practicality of the proposed score, and 2) How to choose $p$.

A1.1: We believe that the ability to search the optimal norm coefficient $p$ is a strength of our method indeed. This is because the latent feature distribution of different datasets could not be necessarily the same as each other. By simply adjusting the value of the norm coefficient, we can succeed in finding the (relatively) most suitable distribution from the exponential family for each dataset in a computationally efficient manner. We also observe that SOTA post-hoc OOD detection methods [a,b,c,d,e,f] come with (one or more) hyper-parameters as well, where their searched values vary across datasets. 

A1.2: Similar to [d], We use a subset of Tiny ImageNet as the auxiliary OOD data. We remove those data whose labels coincide with ID cases. We empirically find that setting (1,3] as the searching space of $p$ suffices to work well on CIFAR and ImageNet datasets. We will add the details in the appendix. Please refer to Section A.3 in the revision for details.

[a] Extremely simple activation shaping for out-of-distribution detection. ICLR 2023

[b] LINe: Out-of-Distribution Detection by Leveraging Important Neurons. CVPR 2023

[c] Out-of-distribution detection with deep nearest neighbours. ICML 2022.

[d] Dice: Leveraging sparsification for out-of-distribution detection. ECCV 2022
 
[f] Out-of-Distribution Detection via Conditional Kernel Independence Model. NeurIPS 2022

Q2: For long-tailed OOD detection when the ID training data exhibits an imbalanced class distribution, I guess the accuracy of the importance sampling-based estimation may decrease given a limited number of tail-class predictions. Elaborate why their method still outperforms in the long-tailed scenarios.

A2:  We agree with your insightful intuition since class imbalance is widely recognized as a challenging setting. As shown in the Table below, all methods that involve the use of ID training data suffer from a decrease in their averaged OOD detection performance when the ID training data is long-tailed. Note that we keep using the network pre-trained on the long-tailed version of CIFAR-100 for fair comparison. Even so, it can be found that the performance degeneration of our method is less noticeable than other counterparts. We suspect that the flexibility of the norm coefficient provides us with the chance to find a compromised distribution from the exponential family. We have revised the submission accordingly by adding the discussion above. Please refer to Section A.12 in the revision for details.


|      Baseline  |   MaxMaha    |              |      KNN     |              |      GEM     |              |       Ours      |              |   
|:--------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:---------------:|:------------:|
|                |     FPR95    |     AUROC    |     FPR95    |     AUROC    |     FPR95    |     AUROC    |       FPR95     |     AUROC    |
|       (i)      |     40.16    |     91.00    |     45.72    |     87.64    |     41.89    |     90.42    |       52.00     |     82.86    |
|       (ii)     |     35.66    |     91.51    |     42.40    |     89.81    |     40.41    |     90.59    |       48.46     |     84.73    |
