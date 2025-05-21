# Fused-Temporal-Graph-Neural-Network-for-Fake-News-Early-Detection
Fused-Temporal Graph Neural Network for Fake News Early Detection
# Fast_Training_on_Dynamic_Heterogeneous_Information_Network_for_Fake_News_Detection

1. Install packages from root with conda. 

For `osx-arm64`:
```
conda create --name gnn_ftgnn --file gnn_ftgnn_osx-arm64.txt
conda activate gnn_ftgnn
```

For `linux64`:
```
conda create --name gnn_ftgnn --file gnn_ftgnn_linux64.txt
conda activate gnn_ftgnn
```

or make sure you have the following packages installed:

* matplotlib
* numpy
* pandas
* python 3.6
* torch 1.4.0+cu100
* torch-cluster 1.4.7
* torch-geometric 1.6.2
* torch-scatter 1.5.3
* torch-sparse 0.5.1
* tqdm
* Scikit_learn
  
2. Extract and preprocess data set

Set `data_set` to `Twitter`(https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0) or `Weibo`(https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0).

```
cd data/script
python extract_data.py
```

3. Change parameters in `main.py`:

  `with_author_test = True` to include author information during test.
  `data_set` to `liar_dataset` or `FakeNewsNet`.
  `version` to `no_finetuning` or `with_finetuning`.
  `model_type` to `bipartite` or `heterogeneous`. 

4. Run model from root
```
python main.py --dataset weibo --model completed --cuda 1 --batch 32 --epoch 5 --lr 0.001
```

