# EMNIST Dataset Classification 

some blurb about stuff


### Table of Contents:
1. EDA
2. Feature Selection vs Dimensionality Reduction
    1. Mutual Information
    2. PCA
3. Standard Classification
4. CNN

### Some notes on how to use this:
- I'm assuming you have a level of rudimentary understanding of how one may build or use a classification algorithm. This information probably won't be super useful to you if this is your first time encountering ML concepts.
- Probably don't copy this code directly. I don't mind if you do, BUT I am handwaving over a lot of important details here. A lot of the approaches presented in this notebook will be absent of important steps in favor of having a cleaner presentation in the notebook

## EDA


```python
from scipy.io import loadmat

full = loadmat('EMNIST/emnist-byclass.mat')
```


```python
data = full['dataset']
del full
```

Hard to explain exactly what's going on with the structure of the dataset, but the relevant indices are as follows:

data[0][0][0][0][0][0][:] - these are the preallocated training data

data[0][0][0][0][0][1][:] - these are our training labels

data[0][0][1][0][0][0][:] - test data

data[0][0][1][0][0][1][:] - test labels

Let's take a quick look at what these images look like!


```python
data[0][0][0][0][0][0][1]
```




    array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  19,  27,   8,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,  47, 157,
           189, 126,  33,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,  33, 126,
           208, 247, 252, 244, 159,  22,   3,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  36,
            91, 207, 245, 250, 251, 252, 207,  46,   8,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
            67, 175, 221, 252, 221, 143, 143, 223, 244, 114,  32,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,
            19,  52, 175, 249, 246, 207,  91,  12,  19, 154, 242, 114,  32,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             3,  47, 122, 219, 247, 231, 175,  48,   7,   9,  46, 208, 219,
            50,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,  20, 100, 175, 248, 246, 195, 123,  12,   0,  21,  82,
           231, 204,  34,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,  12, 123, 231, 247, 244, 164,  47,  20,   0,   0,
            38, 127, 245, 139,   9,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   3,  20, 123, 222, 251, 234, 131,  33,   1,   0,
             0,   2,  82, 172, 229,  82,   2,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   1,  36,  95, 231, 252, 219, 159,  23,   0,
             0,   0,   0,  22, 143, 218, 171,  22,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,  22, 159, 219, 252, 219,  91,  35,
             1,   0,   0,   0,  12, 123, 229, 226,  83,   3,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   1,  47, 207, 244, 246, 163,
            35,   9,   0,   0,   1,   7,  48, 175, 242, 209,  46,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 159, 246, 249,
           175,  36,   1,   0,   2,   9,  47,  95, 207, 247, 239, 190,  41,
             1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,  82, 231,
           234, 187,  67,   3,   1,   8,  77, 139, 209, 232, 253, 254, 253,
           246, 163,  33,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9,
           139, 243, 159,  80,   5,  19,  52,  96, 218, 250, 254, 253, 234,
           218, 222, 236, 243, 126,   8,   0,   0,   0,   0,   0,   0,   0,
             4,  32, 203, 221,  52,  12,  12, 122, 219, 234, 253, 254, 246,
           219,  96,  39,  64, 132, 243, 204,  34,   4,   0,   0,   0,   0,
             0,   0,   4,  37, 215, 217,  40,  12,  48, 175, 248, 252, 255,
           253, 219, 163,  36,   5,  24,  80, 220, 219,  50,   9,   0,   0,
             0,   0,   0,   0,   4,  37, 215, 222,  84,  99, 207, 247, 254,
           254, 250, 218,  91,  35,   1,   0,   1,  11, 141, 237, 102,  28,
             0,   0,   0,   0,   0,   0,   2,  20, 158, 238, 227, 234, 253,
           254, 244, 220, 139,  77,   8,   1,   0,   0,   0,   4, 114, 229,
            95,  26,   0,   0,   0,   0,   0,   0,   0,   1,  33, 126, 203,
           215, 217, 204, 126,  79,  11,   2,   0,   0,   0,   0,   0,   0,
            50, 212, 112,  32,   0,   0,   0,   0,   0,   0,   0,   0,   7,
            46, 114, 125, 127, 114,  46,  22,   1,   0,   0,   0,   0,   0,
             0,   0,  39, 196,  82,  21,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   4,   4,   4,   4,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,  31, 153,  33,   4,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   5,  26,   5,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0], dtype=uint8)




```python
data[0][0][0][0][0][1][1]
```




    array([36], dtype=uint8)




```python
from PIL import Image

im = Image.fromarray(data[0][0][0][0][0][0][1].reshape((28,28)).T)
im
```




    
![png](EMNIST_Classification_files/EMNIST_Classification_7_0.png)
    



Not to spoil everything, but the main things you should know going forward are the following:

- Each image is composed of 28x28 black and white pixels (so just the one channel)

- Each pixel can take a value on [0,255] which indicates its intensity/brightness

- 62 possible classes

Our 62 possible classes correspond to:

Index 0-9: nums 0-9

Index 10-35: A-Z

Index 36-61: a-z

### So, how do we formulate this problem in the context of classification algorithms?

The straightforward answer is to simply treat every value/pixel as its own feature, in this case meaning we would end up with 784 features. However this presents its sets of problems due to that high dimensionality (algorithm runtime, memory limitations, curse of dimensionality, etc.), so we'll need to think about strategies to get around that.

## Feature Selection and Dimensionality Reduction

One way that we could think about circumventing this problem is by reducing the number of features that we're using to predict in one of two ways:

1. Selecting only the most important features (i.e., features that provide the most information) and using them to form our model.

2. Find some sort of alternate representation of our data, with fewer features, that encapsulates some meaningful information about the original data.

Let's look at a quick example of each.


### Feature Selection: Mutual Information
Blurb on how it works, maybe


```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.feature_selection import mutual_info_classif
```


```python
mi_X = data[0][0][0][0][0][0][0:50000]
mi_y = data[0][0][0][0][0][1][0:50000]
mi = mutual_info_classif(mi_X,mi_y)
mi
```

    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)





    array([0.00000000e+00, 1.45944931e-03, 2.62843451e-04, 5.27375616e-04,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.84151372e-03,
           4.48791464e-03, 3.81074157e-03, 8.13158004e-03, 0.00000000e+00,
           0.00000000e+00, 3.59986858e-04, 0.00000000e+00, 0.00000000e+00,
           1.34225668e-03, 1.99643969e-03, 2.91168555e-03, 7.95578549e-04,
           0.00000000e+00, 0.00000000e+00, 3.57667132e-03, 1.46492012e-03,
           0.00000000e+00, 0.00000000e+00, 1.06452307e-03, 9.09570429e-04,
           0.00000000e+00, 2.30816752e-05, 0.00000000e+00, 0.00000000e+00,
           5.61052153e-03, 3.60676754e-03, 7.90976344e-03, 2.32159855e-02,
           1.87324802e-02, 2.19408511e-02, 1.25425291e-02, 9.73606160e-03,
           7.24677719e-03, 1.33210825e-02, 1.90640668e-02, 2.10070398e-02,
           1.72119816e-02, 3.15931980e-02, 3.34485305e-02, 3.36392512e-02,
           3.63396207e-02, 3.16761853e-02, 1.78325413e-02, 1.27931249e-02,
           8.05799268e-03, 4.31594089e-03, 0.00000000e+00, 0.00000000e+00,
           4.01640908e-03, 0.00000000e+00, 3.56559580e-03, 8.00801598e-03,
           2.14126631e-02, 3.24993088e-02, 4.17435390e-02, 5.39716336e-02,
           5.05365368e-02, 5.16340883e-02, 4.88621134e-02, 5.12642266e-02,
           6.85482086e-02, 9.22996583e-02, 1.12455503e-01, 1.30432464e-01,
           1.40182818e-01, 1.45604808e-01, 1.42296142e-01, 1.36752556e-01,
           1.26304653e-01, 9.93915100e-02, 8.00315822e-02, 5.43832377e-02,
           2.75362443e-02, 1.32251683e-02, 5.94685006e-03, 7.06677348e-03,
           1.54658481e-03, 2.39771829e-03, 2.24428458e-02, 3.00238444e-02,
           4.17486181e-02, 4.93958571e-02, 5.91543347e-02, 6.69264840e-02,
           7.51945205e-02, 7.70066494e-02, 8.76064498e-02, 1.08254347e-01,
           1.33754166e-01, 1.67619107e-01, 2.02759777e-01, 2.12979572e-01,
           2.33751176e-01, 2.32071874e-01, 2.28154897e-01, 2.16309118e-01,
           1.87562975e-01, 1.62571468e-01, 1.26481283e-01, 8.81546398e-02,
           5.92245504e-02, 2.59362641e-02, 1.15217549e-02, 1.54772037e-03,
           3.61033382e-04, 6.38827598e-03, 1.52035527e-02, 3.52660395e-02,
           6.04559472e-02, 7.19888097e-02, 8.04255425e-02, 7.77115668e-02,
           8.45454499e-02, 9.40556847e-02, 1.26535196e-01, 1.63091471e-01,
           1.96072542e-01, 2.36288319e-01, 2.64880959e-01, 2.79110324e-01,
           2.94250073e-01, 2.91199283e-01, 2.74338423e-01, 2.61179848e-01,
           2.41395121e-01, 2.12102761e-01, 1.67511778e-01, 1.28740028e-01,
           7.90478628e-02, 4.80183255e-02, 1.35734332e-02, 1.10783335e-03,
           0.00000000e+00, 3.07792444e-03, 3.60137272e-02, 4.99494430e-02,
           7.56809512e-02, 8.88607126e-02, 9.88356842e-02, 1.04540698e-01,
           1.15040458e-01, 1.40845548e-01, 1.73137759e-01, 2.06018946e-01,
           2.47392825e-01, 2.92822143e-01, 3.19408157e-01, 3.37467641e-01,
           3.46090130e-01, 3.37871043e-01, 3.23589444e-01, 3.02632894e-01,
           2.82241333e-01, 2.44460515e-01, 2.01140943e-01, 1.51601159e-01,
           1.07595480e-01, 6.34836083e-02, 1.41025034e-02, 0.00000000e+00,
           2.61121280e-03, 1.32346903e-02, 4.69696903e-02, 7.11168756e-02,
           8.53884135e-02, 1.12049879e-01, 1.21393418e-01, 1.34877629e-01,
           1.50732019e-01, 1.79751865e-01, 2.22379819e-01, 2.58461901e-01,
           2.99324703e-01, 3.26060701e-01, 3.46803247e-01, 3.53158065e-01,
           3.57341539e-01, 3.55424844e-01, 3.47144248e-01, 3.19991207e-01,
           2.95340483e-01, 2.63571810e-01, 2.22509929e-01, 1.79168357e-01,
           1.29430359e-01, 7.98635195e-02, 2.16687820e-02, 4.90845860e-04,
           5.04795227e-03, 1.50636729e-02, 5.52560378e-02, 9.00652069e-02,
           1.14354425e-01, 1.39141412e-01, 1.55567793e-01, 1.58005315e-01,
           1.93774629e-01, 2.24488898e-01, 2.64271902e-01, 3.02097137e-01,
           3.32616242e-01, 3.43783801e-01, 3.43071353e-01, 3.31548872e-01,
           3.35104754e-01, 3.42315098e-01, 3.25187282e-01, 3.07979190e-01,
           2.77937395e-01, 2.60863644e-01, 2.30149859e-01, 1.97590696e-01,
           1.56750304e-01, 9.85761637e-02, 2.74064660e-02, 0.00000000e+00,
           1.12446604e-03, 1.04759114e-02, 7.32474896e-02, 1.00357104e-01,
           1.45238700e-01, 1.70285921e-01, 1.91606233e-01, 1.95299955e-01,
           2.23350093e-01, 2.60162561e-01, 2.90919206e-01, 3.18101436e-01,
           3.37061850e-01, 3.28044739e-01, 3.16520837e-01, 2.90985620e-01,
           2.88680447e-01, 2.86170440e-01, 2.75387627e-01, 2.57671034e-01,
           2.38784336e-01, 2.39816292e-01, 2.30164187e-01, 2.13890607e-01,
           1.90560199e-01, 1.34854950e-01, 2.83175707e-02, 0.00000000e+00,
           0.00000000e+00, 1.28313403e-02, 7.20784616e-02, 1.20519936e-01,
           1.62092520e-01, 2.02577480e-01, 2.06744288e-01, 2.15867489e-01,
           2.32899482e-01, 2.63815806e-01, 2.90830436e-01, 3.08855303e-01,
           3.13211086e-01, 2.92488586e-01, 2.72162761e-01, 2.45174207e-01,
           2.43754908e-01, 2.38405135e-01, 2.19770290e-01, 1.97597723e-01,
           2.01329817e-01, 2.21151400e-01, 2.35155302e-01, 2.43000081e-01,
           2.28814672e-01, 1.72574115e-01, 3.05763758e-02, 0.00000000e+00,
           3.25124100e-03, 1.51443237e-02, 8.79756986e-02, 1.39545575e-01,
           1.92409017e-01, 2.21902999e-01, 2.17250441e-01, 2.12845893e-01,
           2.16453532e-01, 2.40154277e-01, 2.60335734e-01, 2.68281205e-01,
           2.73398499e-01, 2.50781528e-01, 2.28718185e-01, 2.12868056e-01,
           2.02175765e-01, 2.00111850e-01, 1.82912290e-01, 1.58735401e-01,
           1.70118787e-01, 2.04059453e-01, 2.40475674e-01, 2.71357364e-01,
           2.71222195e-01, 2.09704952e-01, 4.44515922e-02, 7.64595158e-03,
           0.00000000e+00, 2.05688614e-02, 9.86198520e-02, 1.66356988e-01,
           2.06211887e-01, 2.26993170e-01, 2.01430435e-01, 1.93887797e-01,
           2.00101239e-01, 2.12843278e-01, 2.20027578e-01, 2.31734972e-01,
           2.35363019e-01, 2.36081465e-01, 2.20446861e-01, 2.11759646e-01,
           2.03898132e-01, 2.01833341e-01, 1.81760133e-01, 1.56093672e-01,
           1.61694470e-01, 1.96267861e-01, 2.50469300e-01, 2.82377997e-01,
           3.01839163e-01, 2.43132117e-01, 4.76726613e-02, 1.20745595e-03,
           0.00000000e+00, 2.38057819e-02, 1.27718163e-01, 2.03651707e-01,
           2.30327449e-01, 2.27092516e-01, 1.95868137e-01, 1.77412395e-01,
           1.84254081e-01, 1.88699550e-01, 1.85846510e-01, 2.00840722e-01,
           2.31414925e-01, 2.43453223e-01, 2.50412953e-01, 2.39232577e-01,
           2.37929284e-01, 2.33379604e-01, 2.02228055e-01, 1.77152380e-01,
           1.74186308e-01, 2.01834033e-01, 2.53199991e-01, 2.95724065e-01,
           3.03541755e-01, 2.43262652e-01, 5.19558324e-02, 0.00000000e+00,
           0.00000000e+00, 4.07436379e-02, 1.61197046e-01, 2.38133293e-01,
           2.60258495e-01, 2.31900035e-01, 1.96464164e-01, 1.92759972e-01,
           1.78966536e-01, 1.79731431e-01, 1.68980281e-01, 1.85740888e-01,
           2.40161884e-01, 2.71865716e-01, 2.73660574e-01, 2.74834713e-01,
           2.64822428e-01, 2.39935948e-01, 2.09894169e-01, 1.71204433e-01,
           1.67496186e-01, 2.03700009e-01, 2.48272772e-01, 2.78962170e-01,
           2.92793278e-01, 2.24721152e-01, 5.77244642e-02, 0.00000000e+00,
           4.84295489e-03, 5.31265370e-02, 1.97601209e-01, 2.64530820e-01,
           2.76449317e-01, 2.38611744e-01, 2.06161058e-01, 2.03355276e-01,
           1.85489117e-01, 1.70261010e-01, 1.71956790e-01, 2.10672524e-01,
           2.58409823e-01, 2.85949822e-01, 2.85189211e-01, 2.57925729e-01,
           2.38076333e-01, 2.14337404e-01, 1.82025565e-01, 1.58770927e-01,
           1.61660989e-01, 1.95644676e-01, 2.35371148e-01, 2.52816856e-01,
           2.53246848e-01, 1.88943265e-01, 4.63264812e-02, 1.51299761e-04,
           5.34844040e-03, 6.47713730e-02, 2.20515998e-01, 2.97347155e-01,
           2.77014175e-01, 2.47983770e-01, 2.16913603e-01, 1.96095365e-01,
           1.71899460e-01, 1.59153041e-01, 1.80970109e-01, 2.20316805e-01,
           2.56177388e-01, 2.55696579e-01, 2.41901686e-01, 2.28641743e-01,
           2.06917743e-01, 2.01078850e-01, 1.66979731e-01, 1.70768827e-01,
           1.84201321e-01, 2.10219232e-01, 2.38520216e-01, 2.29865499e-01,
           2.12719520e-01, 1.47880999e-01, 2.97791213e-02, 3.25626405e-03,
           0.00000000e+00, 6.87157135e-02, 2.25197076e-01, 2.86172240e-01,
           2.60396062e-01, 2.34097711e-01, 2.03182491e-01, 1.69718497e-01,
           1.52256881e-01, 1.57190120e-01, 1.94449156e-01, 2.16486807e-01,
           2.40420623e-01, 2.46079594e-01, 2.28260976e-01, 2.13632338e-01,
           2.04035237e-01, 1.82817142e-01, 1.76673908e-01, 1.99216422e-01,
           2.20030338e-01, 2.45868532e-01, 2.51318500e-01, 2.19505593e-01,
           1.87431887e-01, 1.19688492e-01, 2.60497312e-02, 0.00000000e+00,
           0.00000000e+00, 6.39041287e-02, 2.04839275e-01, 2.55312018e-01,
           2.38596343e-01, 2.11371284e-01, 1.88049933e-01, 1.61969611e-01,
           1.55681026e-01, 1.87915761e-01, 2.15594563e-01, 2.33209715e-01,
           2.36292731e-01, 2.24380703e-01, 2.18341024e-01, 2.09412004e-01,
           2.07227394e-01, 2.02691115e-01, 2.19468304e-01, 2.53582421e-01,
           2.67164536e-01, 2.77102361e-01, 2.58351609e-01, 2.12419273e-01,
           1.57273389e-01, 9.52866147e-02, 2.07976982e-02, 3.57062300e-04,
           0.00000000e+00, 4.89339034e-02, 1.65835978e-01, 2.02235300e-01,
           1.97461405e-01, 1.98664706e-01, 1.82789875e-01, 1.72163516e-01,
           1.88080307e-01, 2.29406459e-01, 2.46691910e-01, 2.38150363e-01,
           2.17995983e-01, 2.11179786e-01, 2.15480486e-01, 2.18987603e-01,
           2.28357016e-01, 2.39398425e-01, 2.61250573e-01, 2.85016158e-01,
           3.04695140e-01, 3.02083574e-01, 2.52828962e-01, 1.99682892e-01,
           1.43048828e-01, 7.95068841e-02, 1.94561181e-02, 0.00000000e+00,
           0.00000000e+00, 3.30192533e-02, 1.30540717e-01, 1.59541612e-01,
           1.73763231e-01, 1.86601464e-01, 1.90340171e-01, 1.98764648e-01,
           2.17667142e-01, 2.55945388e-01, 2.48764296e-01, 2.32044214e-01,
           1.99454569e-01, 2.06265734e-01, 2.16037384e-01, 2.46133278e-01,
           2.57256687e-01, 2.80628295e-01, 2.95883768e-01, 3.07254163e-01,
           3.21264258e-01, 2.90218396e-01, 2.43604325e-01, 1.76956553e-01,
           1.17075522e-01, 6.92108139e-02, 1.64648617e-02, 0.00000000e+00,
           0.00000000e+00, 3.26837039e-02, 1.08103206e-01, 1.33491274e-01,
           1.50483339e-01, 1.69842912e-01, 1.82179105e-01, 1.96000196e-01,
           2.27623383e-01, 2.44139488e-01, 2.53050582e-01, 2.35388771e-01,
           2.28781976e-01, 2.21591625e-01, 2.43917526e-01, 2.56232470e-01,
           2.79470176e-01, 2.90062788e-01, 2.99333863e-01, 3.08146763e-01,
           2.93483402e-01, 2.67636712e-01, 2.16562364e-01, 1.54960855e-01,
           1.10083758e-01, 6.27658166e-02, 9.67796168e-03, 7.21724787e-03,
           1.05592512e-04, 3.34046817e-02, 9.71219149e-02, 1.18353406e-01,
           1.36537583e-01, 1.55057517e-01, 1.66266740e-01, 1.82765483e-01,
           2.03455273e-01, 2.26595570e-01, 2.56464524e-01, 2.59307754e-01,
           2.51410326e-01, 2.55561887e-01, 2.65361795e-01, 2.71344986e-01,
           2.72589622e-01, 2.72010991e-01, 2.72097679e-01, 2.67661141e-01,
           2.51770961e-01, 2.24092783e-01, 1.88899493e-01, 1.41674768e-01,
           9.46249425e-02, 5.41538220e-02, 1.26157244e-02, 1.50371646e-03,
           2.83343018e-03, 2.49007172e-02, 7.92555447e-02, 1.06976566e-01,
           1.23320851e-01, 1.29703930e-01, 1.46384808e-01, 1.53498934e-01,
           1.81804857e-01, 2.17935453e-01, 2.49946047e-01, 2.64997591e-01,
           2.65551045e-01, 2.63356191e-01, 2.58131571e-01, 2.47149192e-01,
           2.41430421e-01, 2.40468129e-01, 2.33899967e-01, 2.28029899e-01,
           2.08966476e-01, 1.84329587e-01, 1.47168997e-01, 1.15337347e-01,
           7.16527010e-02, 4.64950325e-02, 1.43332087e-02, 0.00000000e+00,
           0.00000000e+00, 2.35467331e-02, 6.62881895e-02, 8.58686991e-02,
           1.04071995e-01, 1.15558885e-01, 1.25108848e-01, 1.33020660e-01,
           1.59674334e-01, 1.86934785e-01, 2.17076048e-01, 2.30205331e-01,
           2.36126155e-01, 2.37140114e-01, 2.27027107e-01, 2.09459826e-01,
           1.93122261e-01, 1.86644950e-01, 1.87762202e-01, 1.82845364e-01,
           1.67126879e-01, 1.44097333e-01, 1.24739853e-01, 8.71592969e-02,
           6.20212282e-02, 2.78974227e-02, 6.50527632e-03, 0.00000000e+00,
           0.00000000e+00, 1.12821494e-02, 4.49658024e-02, 6.81567328e-02,
           8.51609001e-02, 1.01140115e-01, 1.04468722e-01, 1.11593284e-01,
           1.18723866e-01, 1.50512107e-01, 1.66367178e-01, 1.85898432e-01,
           1.88428152e-01, 1.82177363e-01, 1.72183369e-01, 1.53593335e-01,
           1.43025307e-01, 1.34906640e-01, 1.35296485e-01, 1.41967277e-01,
           1.33488762e-01, 1.09588459e-01, 9.14739792e-02, 6.48502602e-02,
           3.70093488e-02, 1.95717563e-02, 0.00000000e+00, 8.95464752e-04,
           0.00000000e+00, 0.00000000e+00, 1.88646391e-02, 5.02678079e-02,
           6.20176238e-02, 7.59958438e-02, 8.36406983e-02, 8.30082047e-02,
           8.57063865e-02, 8.60203324e-02, 1.07893324e-01, 1.10775263e-01,
           1.12879560e-01, 1.07380084e-01, 9.55657188e-02, 9.05192476e-02,
           7.59349803e-02, 7.92004042e-02, 7.92304059e-02, 8.90890733e-02,
           8.87328377e-02, 6.92131021e-02, 5.75452176e-02, 3.38745112e-02,
           1.19445482e-02, 5.07153977e-03, 2.17828359e-03, 1.95499116e-04,
           0.00000000e+00, 1.36536536e-03, 4.01367873e-03, 1.32558700e-02,
           2.53086624e-02, 3.21919756e-02, 2.98694392e-02, 2.91846477e-02,
           2.45166238e-02, 2.36932803e-02, 1.57654983e-02, 1.80763258e-02,
           1.89750997e-02, 1.47279592e-02, 1.70797516e-02, 1.70498222e-02,
           1.19226980e-02, 1.54063364e-02, 1.38992043e-02, 1.90641676e-02,
           1.90086065e-02, 1.08221019e-02, 1.13823596e-02, 5.62571644e-03,
           3.77801733e-03, 1.76641411e-03, 5.11441349e-04, 0.00000000e+00,
           7.11439216e-03, 3.10007503e-03, 0.00000000e+00, 2.61706451e-03,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 6.18057177e-04, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           3.60213462e-03, 8.15114422e-04, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 1.58615543e-03, 0.00000000e+00, 0.00000000e+00,
           2.09803510e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])



For a quick visualization of what this looks like:


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(mi.reshape(28,28).T);
```


    
![png](EMNIST_Classification_files/EMNIST_Classification_15_0.png)
    


Above, brighter spots correspond to areas that are more correlated with our target values.


```python
indexList = np.flip(np.argsort(mi))
```


```python
indexList
#indexList holds the indices of the 'most important' features in descending order
```




    array([184, 185, 183, 186, 182, 156, 209, 210, 213, 157, 155, 236, 212,
           208, 211, 237, 181, 214, 158, 552, 187, 154, 235, 238, 264, 263,
           579, 215, 551, 524, 360, 159, 207, 525, 332, 578, 180, 423, 550,
           359, 188, 128, 580, 153, 388, 265, 129, 239, 234, 262, 553, 577,
           240, 451, 241, 405, 406, 523, 331, 160, 549, 576, 127, 387, 216,
           497, 424, 396, 242, 379, 130, 378, 292, 604, 266, 606, 605, 377,
           303, 603, 304, 291, 607, 581, 496, 628, 602, 627, 126, 380, 395,
           206, 261, 189, 629, 522, 131, 217, 452, 290, 368, 233, 599, 179,
           404, 498, 630, 407, 243, 548, 598, 575, 432, 541, 433, 601, 479,
           495, 416, 358, 570, 526, 415, 608, 600, 470, 293, 330, 350, 626,
           542, 386, 425, 152, 631, 514, 547, 461, 469, 267, 161, 569, 574,
           268, 554, 349, 361, 333, 275, 434, 632, 132, 302, 633, 460, 376,
           289, 381, 245, 521, 351, 244, 397, 480, 442, 269, 515, 367, 408,
           352, 657, 488, 125, 656, 321, 571, 414, 320, 274, 453, 634, 100,
           353, 487, 260, 101, 543, 369, 319, 348, 340, 655, 246, 218, 443,
           513, 276, 572, 294, 435, 520, 462, 102, 635, 568, 341, 658, 313,
           597, 450, 389, 205, 489, 609, 232, 190, 178, 285, 573, 273, 422,
           322, 431, 468, 318, 270, 471, 494, 519, 490, 516, 625, 540, 286,
           654, 426, 582, 459, 288, 103, 546, 259, 486, 518, 409, 247, 463,
            99, 295, 287, 317, 444, 499, 133, 323, 481, 517, 403, 441, 382,
           305, 659, 491, 636, 492, 436, 258, 545, 312, 398, 151, 478, 301,
           464, 324, 385, 339, 596, 399, 454,  98, 493, 257, 507, 354, 296,
           357, 325, 314, 272, 162, 437, 347, 297, 316, 527, 544, 467, 539,
           509, 394, 271, 219, 508, 370, 329, 427, 124, 567, 342, 413, 231,
           458, 315, 204, 660, 371, 284, 230, 248, 538, 417, 610, 345, 684,
           512, 482, 485, 662, 104, 472, 653, 661, 537, 683, 346, 375, 400,
           637, 344, 440, 298, 663, 465, 510, 595, 566, 685, 410, 624, 326,
           430, 177, 373, 191, 372, 343, 355, 555, 466, 356, 536, 150, 277,
           686, 511, 402, 428, 383, 439, 229, 401, 300, 565, 455, 374,  97,
           134, 384, 664, 438, 682, 311, 594, 506, 123, 105, 256, 483, 328,
           412, 366, 652, 535, 429, 411, 299, 203, 500, 457, 220, 327, 484,
           202, 593, 583, 687, 623, 456, 163, 176, 681, 564, 445, 638, 622,
            73, 228, 665, 528, 688,  74, 691, 611, 149,  72, 283, 201,  75,
           592, 690, 689, 175, 249,  96, 563, 692, 651, 534,  71, 621, 192,
           135, 338, 122, 106,  76, 650, 666, 620, 174, 255, 473, 680, 591,
           556, 649, 639, 148, 200, 712,  70, 173, 679, 711, 584, 693,  95,
           562, 710, 164, 713, 619, 147, 678, 648, 677, 227,  77, 146, 310,
           221, 590, 714, 501, 612, 121,  69, 694, 715, 199, 719, 145, 720,
           107, 282,  94, 667, 709, 647, 708, 172, 676, 120, 706, 707, 118,
            78, 193, 529, 618, 718, 717, 136, 119,  93, 705, 716, 144,  92,
           226, 254, 117, 640, 171, 721, 557, 449,  68, 675,  91, 646, 695,
           421, 477, 165, 585, 668, 704, 116, 108,  90, 390, 722, 198,  79,
           613,  63, 393, 362,  65,  67,  64, 703, 143,  89, 505,  66, 137,
           334, 170, 641, 418, 674, 306,  88,  62, 365, 696,  48, 142, 115,
           723,  47,  46, 589, 533, 561,  61, 733,  49,  45, 278,  87, 734,
           446, 735, 250, 669,  80, 222, 474, 109, 732, 617, 736, 337, 737,
           645,  35,  86,  37, 194,  60,  43, 502, 309, 697, 530, 747,  42,
           748, 740, 702,  36, 739,  50,  44, 742, 743, 558, 738, 745, 114,
           281, 197, 741, 642, 166, 746, 138,  41, 731, 169,  81, 253,  51,
           614,  38, 724, 744, 110, 750, 673, 749, 225,  39, 586,  10,  52,
            59,  34, 307,  40, 587, 756,  83, 670, 113,  82, 751,  32, 420,
           725, 196, 392,   8,  53,  56, 730,   7,   9, 752,  33, 772,  22,
            58, 447, 280, 757, 141,  18, 616, 759, 168,  85, 726, 780,  17,
           753, 777, 111,  84, 615,  23,   1, 729,  16, 335, 224, 139,  26,
            27, 699, 773,  19, 766,   3, 754, 195, 112,  13, 503,   2, 727,
           419, 588,  29,  11,  12,  31,  30,  28,  25,  24,  21,  20,   4,
             5,   6,  15,  14, 783, 391,  54, 765, 700, 701, 728, 755, 758,
           760, 761, 762, 763, 764, 767,  55, 768, 769, 770, 771, 774, 775,
           776, 778, 779, 781, 698, 672, 671, 644,  57, 140, 167, 223, 251,
           252, 279, 308, 336, 363, 364, 782, 448, 475, 476, 504, 531, 532,
           559, 560, 643,   0])



Let's see how well this works! Using Random Forest for our model, lets look how our model's accuracy increases with added features:


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

toColumns = list(zip(*mi_X))

scores=[]
for i in range(2,65):
    #need to snag the appropriate features
    count=1
    data = np.array(toColumns[indexList[0]])
    while count<i:   
        data = np.column_stack((data,np.array(toColumns[indexList[count]]).reshape(len(toColumns[i]),1)))
        count+=1
    #then do the modeling
    X_train,X_test,y_train,y_test = train_test_split(data,mi_y, test_size=0.3)
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    temp = rf.score(X_test,y_test)
    print(temp)
    scores.append(temp)
```

    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.1324


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.13566666666666666


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.14886666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.16


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.181


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.20233333333333334


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.2118


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.2244


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.23093333333333332


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.233


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.25666666666666665


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.26306666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.2698


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.276


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.2812


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.2809333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.2858


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.29406666666666664


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3376


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.35033333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3474


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3595333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3569333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3742


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3828666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3934


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.3924


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.40313333333333334


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.41113333333333335


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.4478


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.45126666666666665


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.45286666666666664


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.4570666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.4614666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.46686666666666665


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.4679333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5096


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5086666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5168666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5222


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5186666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.524


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5231333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5170666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.523


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5244666666666666


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5274


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5393333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5339333333333334


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5325333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.544


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5528666666666666


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5548666666666666


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5597333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5862666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5823333333333334


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5966666666666667


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5918


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5898


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5971333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.5933333333333334


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.6027333333333333


    <ipython-input-30-cf41432da07d>:17: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)


    0.599



```python
import matplotlib.pyplot as plt
plt.plot(range(2,65),scores, label='Random Forest')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy Score')
plt.title('MI Feature Selection vs Accuracy')
plt.legend()
plt.show();
```


    
![png](EMNIST_Classification_files/EMNIST_Classification_21_0.png)
    


This graph is actually fairly interesting, especially in comparison to PCA which you will see in just a moment. A couple of things to note is how irregular the changes are. Sometimes with a new feature you see a huge increase in accuracy, other times small increase, and sometimes we even get a decrease in accuracy with a new feature. Additionally, our accuracy isn't even that good. We can certainly do better than this!

### PCA


Going to be using SKLearn LogReg, a non-SKLearn LogReg, and then RF additionally.


```python
#non-SKLearn LogReg:
#sigmoid activation
def sigmoid(z):
    return 1/ (1+np.exp(-z))

#cross entropy loss (J):

def cost(theta, x, y):
    h = sigmoid(x @ theta)
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h)
    )
    grad = 1 / m * ((y - h) @ x)
    #grad returns a value for each bi per row
    return cost, grad

def fit(x, y, max_iter=5000, alpha=0.1):
    x = np.insert(x, 0, 1, axis=1)
    thetas = []
    classes = np.unique(y)
    costs = np.zeros(max_iter)

    for c in classes:
        # one vs. rest binary classification
        #basically a boolean map for each class
        binary_y = np.where(y == c, 1, 0)
        
        theta = np.zeros(x.shape[1])
        for epoch in range(max_iter):
            costs[epoch], grad = cost(theta, x, binary_y)
            theta = theta + alpha * grad
            
        thetas.append(theta)
    return thetas, classes, costs

def predict(classes, thetas, x):
    #inserts column of 1s at index 0 to multiple against b0
    x = np.insert(x, 0, 1, axis=1)
    preds = [np.argmax(
        [sigmoid(xi @ theta) for theta in thetas]) for xi in x]
    return [classes[p] for p in preds]

def score(classes, theta, x, y):
    return (predict(classes, theta, x) == y).mean()
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.metrics import accuracy_score

scoresSKLR = []
#scoresMeLR = []
scoresRF = []

for i in range(1,30):
    pca = decomposition.PCA(n_components=i)
    X_train,X_test,y_train,y_test = train_test_split(mi_X,mi_y, test_size=0.3)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    lrSK = LogisticRegression(max_iter=10000)
    lrSK.fit(X_train,y_train)
    #thetas, classes, costs = fit(X_train,y_train)
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    scoresSKLR.append(lrSK.score(X_test,y_test))
    #scoresMeLR.append(score(classes, thetas, X_test, y_test))
    scoresRF.append(rf.score(X_test,y_test))
```

    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)
    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/sklearn/utils/validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    <ipython-input-62-c71176684ca0>:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train,y_train)



```python
plt.plot(range(1,len(scoresRF)+1),scoresRF, label='Random Forest')
plt.plot(range(1,len(scoresSKLR)+1), scoresSKLR, label='Logistic Regression')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy Score')
plt.title('PCA Dimensionality Reduction vs Accuracy')
plt.legend()
plt.show();
```


    
![png](EMNIST_Classification_files/EMNIST_Classification_26_0.png)
    


Some blurb

### A quick look at a CNN


```python
from scipy.io import loadmat

full = loadmat('EMNIST/emnist-byclass.mat')
data = full['dataset']
del full
arrs = data[0][0][0][0][0][0]
labels = data[0][0][0][0][0][1]
```


```python
arrs = arrs.reshape((arrs.shape[0],28,28,1))
X=np.empty((len(arrs),28,28))
for i in range(len(arrs)):
    X[i]= arrs[i].T
```


```python
train_X, test_X, train_y, test_y = train_test_split(X,labels, test_size=.33)

```


```python
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

train_norm = train_X.astype('float32')
test_norm = test_X.astype('float32')
# normalize to range [0-1]
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0
```


```python
train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)
```


```python
batch_size = 64
epochs = 10
num_classes = 62
```


```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
```


```python
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#using adam optimizer
tf.config.run_functions_eagerly(True)
#not sure why but need to set the above switch in order for things to run
```


```python
emnist_train = model.fit(train_norm, train_y, batch_size=batch_size,epochs=epochs,verbose=1)
```

    /opt/anaconda3/envs/metis/lib/python3.8/site-packages/tensorflow/python/data/ops/dataset_ops.py:3703: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable.debug_mode()`.
      warnings.warn(


    Epoch 1/10
    7307/7307 [==============================] - 861s 118ms/step - loss: 0.6765 - accuracy: 0.7892
    Epoch 2/10
    7307/7307 [==============================] - 842s 115ms/step - loss: 0.3737 - accuracy: 0.8615
    Epoch 3/10
    7307/7307 [==============================] - 840s 115ms/step - loss: 0.3464 - accuracy: 0.8696
    Epoch 4/10
    7307/7307 [==============================] - 2664s 365ms/step - loss: 0.3291 - accuracy: 0.8744
    Epoch 5/10
    7307/7307 [==============================] - 1194s 163ms/step - loss: 0.3137 - accuracy: 0.8789
    Epoch 6/10
    7307/7307 [==============================] - 850s 116ms/step - loss: 0.3000 - accuracy: 0.8842
    Epoch 7/10
    7307/7307 [==============================] - 2797s 383ms/step - loss: 0.2924 - accuracy: 0.8854
    Epoch 8/10
    7307/7307 [==============================] - 836s 114ms/step - loss: 0.2821 - accuracy: 0.8894
    Epoch 9/10
    7307/7307 [==============================] - 847s 116ms/step - loss: 0.2750 - accuracy: 0.8910
    Epoch 10/10
    7307/7307 [==============================] - 1920s 263ms/step - loss: 0.2640 - accuracy: 0.8948



```python
model.evaluate(test_norm,test_y)
```

    7198/7198 [==============================] - 223s 31ms/step - loss: 0.4052 - accuracy: 0.8628





    [0.40515509247779846, 0.8628331422805786]




```python

```
