# Dynamic Partial Removal

This project provides the code to replicate the experiments in the paper:

> <cite> Dynamic Partial Removal: A Neural Network Heuristic for Large Neighborhood Search[arxiv link](https://arxiv.org/pdf/2005.09330.pdf) </cite>

The proposed approach introduces a novel Large-Neighborhood-Search Heuristic, namely Dynamic Partial Removal (DPR), for Combinatorial Optimization problems. This heuristic is efficient due to spatial and temporal exploration. A neural network architecture - Hierarchical Recurrent Graph Convolutional Network (HRGCN) - is introduced to perform the DPR heuristic, and this network is trained by reinforcement learning approach (PPO).

We have applied this algorithm to Capacited Vehicle Routing Problem with Time-Windows (CVRPTW) as an example. This algorithm is able to solve large scale problems, and outperforms the traditional LNS heuristics. This algorithm can also be generalized to mixed-scale problems, say, to train the model by the data with scale A, and apply the trained model to problem with scale B. 

Welcome to cite our work (bib):

``` 
@misc{chen2020dynamic,
    title={Dynamic Partial Removal: A Neural Network Heuristic for Large Neighborhood Search},
    author={Mingxiang Chen and Lei Gao and Qichang Chen and Zhixin Liu},
    year={2020},
    eprint={2005.09330},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

Run "train.py" to train the model. A pre-trained model is provided under "model/ppo_1/ppo.pth". "test.ipynb" provides the test code for solving benchmark problems (Solomon benchmark and Gehring & Homberger benchmark). Please unzip "data.zip" before testing the model. Our synthetic dataset are provided in "validation_set.zip".

Example:
```
python train.py --epoch 10000
```
By 物界科技 WaterMirror Ltd. www.water-mirror.com

**We Are Hiring!! ML, OR, please email at liuzhixin AT watermirror.ai**
