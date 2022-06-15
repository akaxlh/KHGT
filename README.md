# Knowledge-Enhanced Graph Transformer

This repository contains TensorFlow codes and datasets for the paper:

>Lianghao Xia, Chao Huang, Yong Xu, Peng Dai, Xiyue Zhang, Hongsheng Yang, Jian Pei, Liefeng Bo (2021). Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation, <a href='https://ojs.aaai.org/index.php/AAAI/article/view/16576'> Paper in AAAI</a>, <a href='https://arxiv.org/abs/2110.04000'> Paper in ArXiv</a>. In AAAI'21, Online, February 2-9, 2021.

## Introduction

Knowledge-Enhanced Graph Transformer (KHGT) is a graph-based recommender system focusing on multi-behavior user-item relation learning. The main characteristics of this model includes: i) it incorporates item-side knowledge information to enhance the item embedding learning, ii) the method explicitly models the cross-type dependencies between different user behavior categories, using a hierarchical graph transformer framework, iii) its time-aware graph neural architecture captures the dynamic characteristics of multi-typed user-item interactions.

## Citation

If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{xia2021khgt,
  author    = {Xia, Lianghao and
               Huang, Chao and
               Xu, Yong and
               Dai, Peng and
               Zhang, Xiyue and
               Yang, Hongsheng and
               Pei, Jian and
               Bo, Liefeng},
  title     = {Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation},
  booktitle = {Proceedings of the 35th AAAI Conference on Artificial Intelligence,
  			  AAAI 2021,
              Online, February 2-9, 2021.},
  year      = {2021},
}
```

## Environment

The codes of KHGT are implemented and tested under the following development environment:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

## Datasets
We employed three datasets to evaluate KHGT, <i>i.e. Yelp, MovieLens, Online Retail</i>. For Yelp and Movielens data, <i>like</i> behavior is taken as the target behavior, and <i>purchase</i> behavior is taken as the target behavior for the Online Retail data. The last target behavior for the test users are left out to compose the testing set. We filtered out users and items with too few interactions.

## How to Run the Codes
Please unzip the datasets first. Also you need to create the `History/` and the `Models/` directories. The command to train KHGT is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper. For Movielens and Online Retail data, we conducted sub-graph sampling to efficiently handle the large-scale multi-behavior user-item graphs.

* Yelp
```
python labcode_yelp.py
```

* MovieLens, training
```
python labcode_ml10m.py --data ml10m --graphSampleN 1000 --save_path model_name
```
- MovieLens, testing
```
python labcode_ml10m.py --data ml10m --graphSampleN 5000 --epoch 0 --load_model model_name
```

* Online Retail, training
```
python labcode_retail.py --data retail --graphSampleN 15000 --reg 1e-1 --save_path model_name
```
- Online Retail, testing
```
python labcode_retail.py --data retail --graphSampleN 30000 --epoch 0 --load_model model_name
```

Important arguments:
* `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5}`.

* `graphSampleN`: It denotes the number of nodes to sample in each step of sub-graph sampling. It is tuned from the set `{1000, 5000, 10000, 15000, 20000, 30000, 40000, 50000}`

## Acknowledgements
We thank the anonymous reviewers for their constructive feedback and comments. This work is supported by
National Nature Science Foundation of China (62072188,
61672241), Natural Science Foundation of Guangdong
Province (2016A030308013), Science and Technology Program of Guangdong Province (2019A050510010).
