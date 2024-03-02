# 新型學習演算法 (New-Learning-Algorithm-2021-NCCU)

New Learning Algorithm (2021), by Prof. TSAIH RUA-HUAN at Management Information Systems in NCCU

## Introduction
The learning algorithm is a revised learning mechanism based on simple two layer net. To overcome the overfitting and learning dilemma issues, two module, reorganizing module and cramming, are proposed to cope with each issues respectively.

## Learning Goals
Predicting copper price.

## Data
- Cleaned data provided by teacher with 18 covariate with dependent variable as price
- Inspect with [HOUSE PRICE data](https://github.com/KJJHHH/New-Learning-Algorithm/tree/main/DATA_HOUSE_PRICE)

## Environment
- GPU: CUDA 12.0
- Python 3.11
- Pytorch


## Mechanism

### LTS (Lecture 11)
- Select the {ns} samples that fit the learning goal 
- Select {ks} of k element that is not in {ns}
- Take {ns} + {ks} as training data
- Keep doing full learning algorithm to get all training data

### Weight Tuning Module (Lecture 6)
- Simple learning
- If acceptable: go to Reorganise module
- If not acceptable: go to Cramming

### Reorganising Module (Lecture 7)
- Complex learning
- Removing irrelavent nodes
- Coping with Overfitting problem

### Cramming Module (Lecture 9)
- Ruled based adding nodes
- For each case that did not fit well to the model, assign three nodes for the case in the model, where the weights for each nodes is predefined

### Full learning algorithm (Lecture 11)
```
Notation
# n: picked data to train in traing data
# N: all training data
```
1. Start and INitilasie with hidden node size 1: Initilialise Module
2. Let n = obtaining_LTS, n += 1 
    - The obtaining_LTS
    - if n > N break 
3. Selecting _LTS(n). I(n) = the picked data indexes
4. If the learning goal for picked train data satisfied (max(eps) <= learning goal), Go step 7; Otherwise, there is oen and onlly one k in n that cause contradicton and k = [n]    
5. Save weight
6. Weight tune the current SLFN
    - IF acceptable: go step 7
    - Otherwise, restore weight cram to get acceptable SLFN
7. Reorganise SLFN
8. GO to step 2


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


