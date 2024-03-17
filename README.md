# 新型學習演算法 (New-Learning-Algorithm-2021-NCCU)

New Learning Algorithm (2021), by Prof. TSAIH RUA-HUAN at Management Information Systems in NCCU

## Introduction
The **new learning algorithm** is a revised learning mechanism based on simple two layer net, aiming to overcome the overfitting and learning dilemma issues.

## Learning Goals
Predicting copper price.

## Data
Cleaned data provided by teacher with 18 variables and copper price

## Environment
- GPU: CUDA 12.0
- Python 3.11
- Pytorch


## MODULES
The modules used in new learning algorithm
### NOTE
:notebook: **Definition of acceptable**: The maximum residual error for training samples does not exceed a predefined threshold.
### LTS (Lecture 11)
- Select all samples that fit the learning goal, denoted as {ns}
- Select k samples, denoted as {ks}, that is not in {ns}
- Take {ns} + {ks} as training data
### Weight Tuning Module (Lecture 6)
- If acceptable: go to Reorganise module
- If not acceptable: go to Cramming

### Reorganising Module (Lecture 7)
- Removing irrelavent nodes
- Coping with Overfitting problem

### Cramming Module (Lecture 9)
- Ruled based adding nodes
- For each case that did not fit well to the model, assign three nodes for the case in the model, where the weights for each nodes is predefined

## NEW LEARNING ALGORITHM
`````python  
"""
NOTE:
# n: picked data to train in traing data
# N: all training data
"""
1. Start and Initialise model with hidden node size 1 and do Weight tuning
2. Do LTS with k = 1 and get selected training data, denoted as S. 
3. Save weight
    3.1 If S residuals satisfied learning goal (max(eps) <= learning goal), Go step 5; 
    3.2 Otherwise, there is one and only one sample in S that are not satisfy the learning goal    
4. Weight tune the current model
    4.1 IF acceptable: go step 5
    4.2 Otherwise, restore weight and do cramming to get acceptable SLFN
5. Reorganise SLFN
6. GO to step 2
`````


## Result

1. Full learning algorithm 
    Train loss / Test loss
    - epoch 50               # for each module
    - learning goal: e ** 2  # for all residual square < learning goals in train data
    - learning rate: 0.01    # for each module

    |        Dataset       | Full learning algorithm | Train time            |
    | -------------------- | ----------------------- | --------------------- |
    |       Copper         |        1.366/19.586     |   250 min (not sure)  |

2. Benchmark: simple fully connected net (2 ~ 3)\
    Trainloss / Testloss/ Epochs / Traintime(Min)

    |  Dataset   | Two layer net           |  
    | ---------- | ----------------------- |  
    |   Copper   | 1.075/19.261/2000/4     | 

    - learning rate: 0.001, 0.01 will explode
    - hidden nodes: 50



## Conclusion 
1. The full learning algorithm might not out perform other models

### Others
1. Benchmark model with no initialisation and 10000 epochs
    - Two layer net on copper
        ![Image](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU/blob/main/baseline_result/Two-Layer-Net%20Loss.png)
    - Three layer net on copper
        ![Image](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU/blob/main/baseline_result/Three-Layer-Net%20Loss.png)


