
## Usage

### 1. Dataset Preparation for Large-scale Experiment 

#### In-distribution dataset

##### ResNet50:
Run python imagenet_resnet/feature_extraction.py
Run python imagenet_resnet/feature_sort_resnet_50.py
Place the sorted features in the folder of cache/imagenet_feature_resnet_50_sorted

##### MobileNetv2:
Run python imagenet_mobilenet/feature_extraction.py
Run python imagenet_mobilenet/feature_sort_mobilenet_v2.py
Place the sorted features in the folder of cache/imagenet_feature_mobilenet_v2_sorted

For convenience, the ImageNet Features can be download from https://1drv.ms/f/s!AtOtfUBMB8Oemu4ceTd_TpKwcJZUDg?e=6PW8wS.

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/ood_data`.

### 2. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset


We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_data/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_data/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_data/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/LSUN`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/iSUN`.
* [LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/LSUN_fix`.


[//]: # (For example, run the following commands in the **root** directory to download **LSUN**:)

[//]: # (```)

[//]: # (cd datasets/ood_datasets)

[//]: # (wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

[//]: # (tar -xvzf LSUN.tar.gz)

[//]: # (```)


### 3.  Pre-trained model

Pre-trained models are placed in the knn-ood-master/checkpoints folder.

## Preliminaries
It is tested Python 3.4 environment, and requries some packages to be installed:
* [PyTorch 1.2.0](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [numpy](http://www.numpy.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [ylib](https://github.com/sunyiyou/ylib) (Manually download and copy to the current folder)

## Demo
### 1. Demo code for Cifar-10 Experiment 
Run python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365_sub --name densenet  --model-arch densenet --epochs 1002

Run python run_cifar_densenet.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365_sub --name densenet  --model-arch densenet

### 2. Demo code for Cifar-100 Experiment 
Run python feat_extract.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365_sub --name densenet  --model-arch densenet --epochs 1002

Run python run_cifar_densenet_c100.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365_sub --name densenet  --model-arch densenet

### 3. Demo code for ImageNet Experiment on ResNet50
Run python feat_extract_largescale.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50

Run python run_imagenet.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50

### 4. Demo code for Cifar-100 Experiment on MobileNetv2
Run python feat_extract_largescale.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name mobilenetv2  --model-arch mobilenetv2

Run python run_imagenet_mobilenet.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name mobilenetv2  --model-arch mobilenetv2


