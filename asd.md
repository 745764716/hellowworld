Dear Area Chairs and Reviewers,

We would like to thank the reviewers again for their constructive and insightful comments, which helps us a lot in improving the submission. We have uploaded the revised version and responded to all the reviewers in detail. We believe that the quality of the paper is improved and the contributions are solid. In particular, we would like to highlight some key materials we added:

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
We appreciate the insightful comments provided by Reviewer EBSh. Please see our responses to your concerns below.

Q1: I think this is an explicit assumption on the prior distribution, which contradicts $\spadesuit$ ( how can we obtain a tractable estimate for $\Phi (k)$ without presuming any particular prior distribution of $\hat{p}_{\boldsymbol{\theta}}\left(\mathbf{z}|k \right)$ )

A1: We would like to emphasize that: 1) the mild assumption of the exponential family is introduced to guide the design of density function ($\clubsuit$), where we search against the given dataset for the best choice of $l_p$ to determine the optimal density function. 2) $\spadesuit$ targets at the computation of the normalizing constant $\Phi(k)$ in Eq.(11) that involves the integral over high-dimensional feature space. Different from prior work that specifies $\hat{p}_{\boldsymbol{\theta}}\left(\mathbf{z}|k \right)$ as a pre-defined distribution, e.g., Guassian, to simplify $\Phi(k)$ as a known value, we alternatively design an importance sampling-based estimator of $\Phi (k)$ as the solution to $\spadesuit$ without loss of generalization. Note that our proposed estimator does not rely on any prior knowledge of data distribution and therefore can be ideally applied to any forms of density functions. 

#### Reviewer WvTp

We thank Reviewer WvTp for your constructive comments. To address some of your concerns, we have updated our paper. Please see below for our point-to-point rebuttal.

Q1: Clarify “Without loss of generality, we employ latent features z extracted from deep models as a surrogate for the original high-dimensional raw data $\mathbf{x}$. This is because $\mathbf{z}$ is deterministic within the post-hoc framework”

A1: Since this paper mainly focuses on the feature space for determining OOD data, we use different notations to explicitly discriminate raw input data $\mathbf{x}$ from the latent feature $\mathbf{z}$ to make things clearer. Note that some prior papers sometimes use $\mathbf{z}$ and $\mathbf{x}$ interchangeably. To clarify, the term "deterministic" here means that, for any given input image x, we can always have a deterministic representation z since the pre-trained encoder is fixed in the setting of post-hoc OOD detection. This motivates us to choose the lower-dimensional feature space as a suitable surrogate of the raw data space X for more computationally efficient density estimation. 

Q2: Discuss the role of Deep generative models (DGMs) for flexible density estimation in OOD detection

A2: We agree that using DGMs for density estimation is, of course, a valid and intuitive option. However, aligning with our response in A1, this practice requires training DGMs from scratch to reconstruct high-dimensional $\mathbf{x}$, therefore bringing more computational overheads. Please kindly note that our paper focuses on the task of Post-hoc OOD detection where only pre-trained models at hand are expected to be used to detect OOD data from streaming data at the interference stage. 

We also explore the possibility of integrating pre-trained Diffusion models [a,b] into zero-shot class-conditioned density estimation based on Eq.(1) in [c]. Unfortunately, the computation is intractable due to the integral. Although authors in [c] use a simplified ELBO for approximation, there is no theoretical guarantee that the ELBO can align with the data density not to mention the computational-inefficient inference of diffusion models. We will leave this challenge as our future work.

[a] Scalable Diffusion Models with Transformers. ICCV 2023.

[b] High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022

[c] Your Diffusion Model is Secretly a Zero-Shot Classifier. ICCV 2023.

>Discussion: Since density estimation plays a key role in our method, our work is related to deep generative models that achieve empirically promising results based on neural networks. Generally, there are two families of DGMs for density estimation: 1) autoregressive models [] that decompose the density into the product of conditional densities based on probability chain rule where Each conditional probability is modelled by a parametric density (e.g., Gaussian or mixture of Gaussian) whose parameters are learned by neural networks, and 2) normalizing flows [] that represent input as an invertible transformation of a latent variable with known density with the invertible transformation as a composition of a series of simple functions. While using DGMs for density estimation seems to a valid and intuitive option for density-based OOD detection, this requires to train a DGM from scratch and therefore violates the principle of post-hoc OOD detection, i.e., only pre-trained models at hand are expected to be used to detect OOD data from streaming data at the interference stage. Besides, [] finds that DGMS tend to assign higher probabilities or densities to OOD images than images from the training distribution. We also explore the possibility of integrating pre-trained Diffusion models [] into zero-shot class-conditioned density estimation based on Eq.(1) in []. Unfortunately, the computation is intractable due to the integral. Although authors in [] use a simplified ELBO for approximation, there is no theoretical guarantee that the ELBO can align with the data density not to mention the computational-inefficient inference of diffusion models. We will leave this challenge as our future work.
 
Q3: Tell more about why learning the natural parameter of an Exp. Family is intractable

A3: Thank you for your advice. given the fact that $\int \hat{p}_{\boldsymbol{\theta}}\left(\mathbf{z}|k \right) {\rm d}\mathbf{z} =1$, we then have:

$$ \int \exp \left\lbrace \mathbf{z}^\top\boldsymbol{\eta}\_k-\psi(\boldsymbol{\eta}_k)-g\_{\psi}(\mathbf{z})\right\rbrace{\rm d}\mathbf{z}=1 $$

This means that, for any known $\psi(\cdot)$ and $g\_{\psi}(\cdot)$, one can learn the natural parameter $\boldsymbol{\eta}_k$ by solving the following equation:

$$\exp \psi(\boldsymbol{\eta}_k)=\int \exp \left\lbrace \mathbf{z}^\top\boldsymbol{\eta}\_k-g\_{\psi}(\mathbf{z})\right\rbrace{\rm d}\mathbf{z}$$

Since the right side of the equation includes the integral over latent feature space that is high-dimensional, learning the natural parameter of an Exp. Family is said to be intractable.

We add the elaboration above in the revised version.

#### Reviewer LYsh
We thank Reviewer LYsh for your thorough comments. As to the weaknesses and minor issues you pointed out, we took them very seriously, and have updated parts of the paper to improve it. Our response is as follows.

Q1: The second weakness is that the prior work (Morteza & Li, 2022) already proposed a method based on Gaussian assumptions and Mahalanobis distance. The extension in this paper, at least logically, is relatively straightforward, i.e., from Gaussian to Exponential Family and from Mahalaobis distance to Bregman-divergence (an extension). Maybe it's worth adding a section summarizing the paper's technical novelty.

A1: We thank you for your advice. The contributions of our method are summarised as follows:

>1 It is always non-trivial to generalize from a specific distribution/distance to a broader distribution/distance family since this will trigger an important question to the optimal design of the underlying distribution ($\clubsuit$).To answer this question, we explore the conjugate relationship as a guideline for the design. Compared with other hand-crafted choices, our proposed $l_p$ norm is general and well-defined, offering simplicity in determining its conjugate pair. By searching the optimal value of p for each dataset, we can flexibly model ID data in a data-driven manner instead of blindly adopting a narrow Gaussian distributional assumption in prior work, i.e., GEM and Maha.

>2 Our proposed framework reveals the core components in density estimation for OOD detection, which was overlooked by most heuristic-based OOD papers. In this way, The framework not only inherits prior work including GEM and Maha but also motivates further work to explore more effective designing principles of density functions for OOD detection.

>3 We demonstrate the superior performance of our method on several OOD detection benchmarks (CIFAR10/100 and ImageNet-1K), different model architectures (DenseNet, ResNet, and MobileNet), and different pre-training protocols (standard classification, long-tailed classification and Contrastive learning).

We included the summary above in Section A.9 of the revision.

Q2: List out all the assumptions made

A2: Thank you for your advice. The assumptions made in our method are given as follows:

>**Assumption 1.** The uniform ID class prior.

Assumption 1 is also made in many post-hoc OOD detection methods either explicitly or implicitly [a]. Experiments in Section 4.4.2 show that our method still outperforms in long-tailed scenarios with Assumption 1.

>**Assumption 2.**  $g_\varphi(\cdot)=const$ and $\psi(\cdot) = \frac{1}{2}\|\|\cdot\|\|_{p}^{2}$

Assumption 2 helps to reduce the complexity of the exponential family distribution. While it is possible to parameterize the exponential family distribution in a more complicated manner, our proposed simple version suffices to perform well.

We listed the assumptions above in Section A.10 of the revision.

[a] Detecting Out-of-distribution Data through In-distribution Class Prior. ICML 2023

Q3: theoretical justification

A3: Here, same as KNN [b], we provide a theoretical analysis of our method by analyzing the average OOD detection performance of our algorithm in Section A.11 and as follows. 
>**Setup.** We consider the OOD detection task as a special binary classification task, where the negative samples (OOD) are only available in the testing stage. We assume the input is from feature embeddings space $\mathcal{Z}$ and the labeling set $\mathcal{G}= \left\lbrace 0(OOD),1(ID) \right\rbrace$. In the inference stage, the testing set $\left\lbrace(\mathbf{z},g)\right\rbrace$ is drawn i.i.d. from $P_{\mathcal{Z} \mathcal{G}}$. Denote the marginal distribution on $\mathcal{Z}$ as $\mathcal{P}$, We adopt the Huber contamination model [] to model the fact that we may encounter both ID and OOD data in test time:
$$\mathcal{P}=(1-\epsilon)\mathcal{P}_I+\epsilon \mathcal{P}_O$$
where $\mathcal{P}_I$ and $\mathcal{P}_o$ are the underlying distributions of feature embeddings for ID and OOD data, respectively, and $\epsilon$ is a constant controlling the fraction of OOD samples in testing. We use $\mathcal{p}_I(\mathbf{z})$ and $\mathcal{p}_o(\mathbf{z})$ to denote the probability density function, where $\mathcal{p}_I(\mathbf{z}) = \mathcal{p}(z|g = 1)$ and $\mathcal{p}_o(\mathbf{z}) = \mathcal{p}(z|g = 0)$. It is natural to approximate $\mathcal{p}_I(\mathbf{z})$ as $\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})$ in Eq.(11). Following KNN, we apprximate OOD distribution by modelling OOD data with an equal chance to appear outside of the high-density region of ID data, i.e., $\mathcal{p}_o(\mathbf{z})=c_o \mathbf{I}\left\lbrace\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})<\beta\right\rbrace$. Given the fact that $\mathcal{p}_o(\mathbf{z})=0$ if $z \in \left\lbrace Enc(\mathbf{x})|\mathbf{x}\in X\_I \right\rbrace$, the empirical esitimator of $\beta$ is given by $\hat\beta = \min\_{(\mathbf{x},\mathbf{y})\in \mathcal{D}\_{\rm in}} \mathcal{p}\_{\boldsymbol{\theta}}(Enc(\mathbf{x}))$ with $Enc(\cdot)$ as the encoder of a pre-train model.

>**Main result 1.** By the Bayesian rule, the probability of $\mathbf{z}$ being ID data is:

$$\begin{aligned}
\mathcal{p}(g = 1|\mathbf{z}) &= \frac{\mathcal{p}(\mathbf{z}|g = 1)\cdot\mathcal{p}(g = 1)}{\mathcal{p}(\mathbf{z}|g = 1)\cdot\mathcal{p}(g = 1)+\mathcal{p}(\mathbf{z}|g = 0)\cdot\mathcal{p}(g = 0)} \\
&= \frac{(1-\epsilon)\mathcal{p}(\mathbf{z}|g = 1)}{(1-\epsilon)\mathcal{p}(\mathbf{z}|g = 1)+\epsilon\mathcal{p}(\mathbf{z}|g = 0)} \\
&\approx \frac{(1-\epsilon)\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})}{(1-\epsilon)\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})+\epsilon c_o\mathbf{I}\left\lbrace\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})<\hat\beta\right\rbrace}
\end{aligned}$$

>**Main result 2.** Given a pre-defined threshold $\lambda$, it can be easily checked that:

$$\mathbf{I}\left\lbrace\mathcal{p}\_{\boldsymbol{\theta}}(\mathbf{z})\ge\lambda\right\rbrace = \mathbf{I}\left\lbrace\mathcal{p}(g = 1|z)\ge\frac{(1-\epsilon)\lambda}{(1-\epsilon)\lambda+\epsilon c_o\mathbf{I}(\lambda<\hat\beta)}\right\rbrace $$

[b] Out-of-distribution detection with deep nearest neighbours. ICML 2022.

[c] Robust estimation of a location parameter. Annals of Mathematical Statistics, 1964.

Q 4: Explain why the algorithm tends to perform well for small p values and derive some for simple cases like the Gaussian one.

Since the searching process of the coefficient $p$ is data-driven, the optimal value of $p$ should vary from dataset to dataset. Therefore, while extensive experiments show that small P values tend to be beneficial to OOD detection on CIFAR and ImageNet, this observation does not necessarily hold for all datasets. It can be seen from Figure 4 (c) and Figure 4 (d) that $p$=2, which corresponds to the Gaussian case, performs noticeably interior to $p$=2.2 in Cifar10 and $p$=2.6 in Cifar100. This indicates the suboptimal of using Gaussian and further emphasizes searching for a better alternative.

Q5: It would be good if the authors could make the notations more distinguishable.

A5: Thank you for your advice. we have improved our use of notations in the revision.

#### Reviewer MqR8
We thank reviewer MqR8 for your valuable comments. We updated the submission accordingly. Please kindly find the detailed responses below. 

Q1: 1) The main search problem of optimal coefficient for OOD scoring remains as a hyperparameter search, which may constrain the practicality of the proposed score, and 2) How to choose $p$.

A1.1: We believe that the ability to search the optimal norm coefficient $p$ is a strength of our method indeed. This is because the latent feature distribution of different datasets could not be necessarily the same as each other. By simply adjusting the value of the norm coefficient, we can succeed in finding the (relatively) most suitable distribution from the exponential family for each ID dataset in a computationally efficient manner. We also observe that SOTA post-hoc OOD detection methods [a,b,c,d,e,f] come with (one or more) hyper-parameters as well, where their searched values vary across datasets. 

A1.2: Similar to [d], We use a subset of Tiny ImageNet as the auxiliary OOD data. We remove those data whose labels coincide with ID cases. We empirically find that our method suffices to work well on CIFAR and ImageNet datasets with (1,3] as the searching space of $p$. We will add the details in the appendix. Please refer to Section A.3 in the revision for details.

[a] Extremely simple activation shaping for out-of-distribution detection. ICLR 2023

[b] LINe: Out-of-Distribution Detection by Leveraging Important Neurons. CVPR 2023

[c] Out-of-distribution detection with deep nearest neighbours. ICML 2022.

[d] Dice: Leveraging sparsification for out-of-distribution detection. ECCV 2022
 
[f] Out-of-Distribution Detection via Conditional Kernel Independence Model. NeurIPS 2022

Q2: For long-tailed OOD detection when the ID training data exhibits an imbalanced class distribution, I guess the accuracy of the importance sampling-based estimation may decrease given a limited number of tail-class predictions. Elaborate why their method still outperforms in the long-tailed scenarios.

A2:  We agree with your insightful intuition since class imbalance is widely recognized as a challenging setting. As shown in the Table below, all methods that involve the use of ID training data suffer from a decrease in their averaged OOD detection performance when the ID training data is with class imbalance. Note that we keep using the network pre-trained on the long-tailed version of CIFAR-100 for fair comparison. Even so, our method consistently outperforms in both scenarios, which implies the robustness of our method. We suspect the reason is that the flexibility of the norm coefficient provides us with the chance to find a compromised distribution from the exponential family. We have revised the submission accordingly by adding the discussion above. Please refer to Section A.12 in the revision for details.

|                |      Maha    |              |      KNN     |              |      GEM     |              |       Ours      |              |   
|:--------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:---------------:|:------------:|
|                |     FPR95    |     AUROC    |     FPR95    |     AUROC    |     FPR95    |     AUROC    |       FPR95     |     AUROC    |
|       class imbalance      |     71.76    |     75.22    |     58.11    |     81.75    |     66.82    |     76.97    |      52.00     |     82.86    |
|       class balance     |     67.39    |     77.16    |     56.67    |     82.93    |     60.11    |     79.22    |      48.46     |     84.02    |
