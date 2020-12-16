# KHGT
Code for KHGT accepted by AAAI2021

Please unzip the data files in Datasets/ first.

To run KHGT on Yelp data, use
```
python labcode_yelp.py
```

For MovieLens data, use the following command to train
```
python labcode_ml10m.py --data ml10m --graphSampleN 1000 --save_path XXX
```
and use this command to test with larger sampled sub-graphs
```
python labcode_ml10m.py --data ml10m --graphSampleN 5000 --epoch 0 --load_model XXX
```

For Online Retail data, use this command to train
```
python labcode_retail.py --data retail --graphSampleN 15000 --reg 1e-1 --save_path XXX
```
and also load the model to test it with larger sampled sub-graphs
```
python labcode_retail.py --data retail --graphSampleN 30000 --epoch 0 --load_model XXX
```
