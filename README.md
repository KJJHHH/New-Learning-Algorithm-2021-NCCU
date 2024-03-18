# 新型學習演算法 (New-Learning-Algorithm-2021-NCCU)

New Learning Algorithm (2021), by Prof. TSAIH RUA-HUAN at Management Information Systems in NCCU

- [INTRODUCTION](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU?tab=readme-ov-file#introduction)
- [DATA](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU?tab=readme-ov-file#data)
- [NEW LEARNING ALGORITHM](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU?tab=readme-ov-file#new-learning-algorithm)
- [RESULT](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU?tab=readme-ov-file#result)
- [ENVIRONMENT](https://github.com/KJJHHH/New-Learning-Algorithm-2021-NCCU?tab=readme-ov-file#environment)

## INTRODUCTION
The **new learning algorithm** is a revised learning mechanism based on simple two layer net, aiming to overcome the overfitting and learning dilemma issues.\
The goal of this project is to **predict copper price** using new leanring algorithm

## DATA
Including 18 variables and copper price

## NEW LEARNING ALGORITHM
### Algorithm
`````python  
"""
NOTE:
1. Definition of acceptable: The maximum residual error for training samples does not exceed a predefined threshold.
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

### MODULES
The modules used in new learning algorith
- LTS (Lecture 11)
    - Select all samples that fit the learning goal, denoted as {ns}
    - Select k samples, denoted as {ks}, that is not in {ns}
    - Take {ns} + {ks} as training data
- Weight Tuning Module (Lecture 6)
    - If acceptable: go to Reorganise module
    - If not acceptable: go to Cramming
- Reorganising Module (Lecture 7)
    - Removing irrelavent nodes
    - Coping with Overfitting problem
- Cramming Module (Lecture 9)
    - Ruled based adding nodes
    - For each case that did not fit well to the model, assign three nodes for the case in the model, where the weights for each nodes is predefined


## RESULT
train loss: 0.00041519435840462924
test loss: 0.002123679771900428
train residual max 0.026442381296536088
test residual max 0.04143138922202543

|                         | Train Loss | Train Max Epsilon | Test Loss  | Test Max Epsilon | Train Time  | 
| ----------------------- | ---------- | ----------------- | ---------- | -----------------| ----------- |
| Full learning algorithm |   0.00016  |      0.00118      | 0.000496   |       0.00500    |    22 min   |
| Two Layer Net           |   0.00041  |      0.02644      | 0.00212    |       0.04143    |   4 min     |

## ENVIRONMENT
- GPU: CUDA 12.0
- Python 3.11
- Pytorch
