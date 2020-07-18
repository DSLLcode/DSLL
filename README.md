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
Author = {Wang, Zhen and Liu, Liu and Tao, Dacheng},
Booktitle = {International Conference on Machine Learning (ICML 20202)},
Pages = {378--387},
Title = {Deep Streaming Label Learning},
Year = {2020}}
```
