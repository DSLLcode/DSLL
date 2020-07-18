# Deep Streaming Label Learning (DSLL)

- Exploring and exploiting the knowledge from past labels and historical models to understand and develop emerging new labels.

- Framework: [PyTorch](https://pytorch.org/)

 
## Dependence

- python 3.7
- Pytoch 1.4
- liac-arff (pip install liac-arff)

## Running
`cd ./`    # open the folder where the code is located

`python3 DSLL.py`   # run DSLL model

## Results
![image](https://github.com/DSLLcode/DSLL/blob/master/results/Figure1.jpg)
**Figure 1.** Performance comparison of learning new labels with different batch sizes by considering 50\% of labels as past labels. *m* indicates the number of new labels.

<br/> 


**Table 1.** Ranking performance of each comparison algorithm for learning new labels with different batch sizes by regarding 50\% of labels as past labels. \#label denotes the number of new labels. ![](http://latex.codecogs.com/gif.latex?%20\downarrow(\uparrow)) 
means the smaller (larger) the value, the better the performance. 

![image](https://github.com/DSLLcode/DSLL/blob/master/results/Table1.png)
<br/> 
More detailed results can be found in the Supplementary Materials.

<br/> 
## Cite
```

@incollection{icml2020_230,
Abstract = {In multi-label learning, each instance can be associated with multiple and non-exclusive labels. Previous studies assume that all the labels in the learning process are fixed and static; however, they ignore the fact that the labels will emerge continuously in changing environments. In order to fill in these research gaps, we propose a novel deep neural network (DNN) based framework, Deep Streaming Label Learning (DSLL), to classify instances with newly emerged labels effectively. DSLL can explore and incorporate the knowledge from past labels and historical models to understand and develop emerging new labels. DSLL consists of three components: 1) a streaming label mapping to extract deep relationships between new labels and past labels with a novel label-correlation aware loss; 2) a streaming feature distillation propagating feature-level knowledge from the historical model to a new model; 3) a senior student network to model new labels with the help of knowledge learned from the past. Theoretically, we prove that DSLL admits tight generalization error bounds for new labels in the DNN framework. Experimentally, extensive empirical results show that the proposed method performs significantly better than the existing state-of-the-art multi-label learning methods to handle the continually emerging new labels.},
Author = {Wang, Zhen and Liu, Liu and Tao, Dacheng},
Booktitle = {Proceedings of Machine Learning and Systems 2020},
Pages = {378--387},
Title = {Deep Streaming Label Learning},
Year = {2020}}
```
