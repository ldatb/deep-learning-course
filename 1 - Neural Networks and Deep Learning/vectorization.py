'''
Vectorization is basically the art of getting rid of explicit folders in your code. 
In the deep learning era safety in deep learning in practice, you often find yourself training on relatively large data sets, 
because that's when deep learning algorithms tend to shine. And so, it's important that your code very quickly because otherwise, 
if it's running on a big data set, your code might take a long time to run then you just find yourself waiting a very long time to get the result. 
So in the deep learning era, I think the ability to perform vectorization has become a key skill. Let's start with an example.

So, what is Vectorization? In logistic regression you need to compute Z equals W transpose X plus B, where W was this column vector and X is also this vector. 
Maybe there are very large vectors if you have a lot of features. So, W and X were both these R and no R, NX dimensional vectors. 
So, to compute W transpose X, if you had a non-vectorized implementation, you would do something like Z equals zero. 
And then for I in range of X. So, for I equals 1, 2 NX, Z plus equals W I times XI. And then maybe you do Z plus equal B at the end. 
So, that's a non-vectorized implementation. Then you find that that's going to be really slow. 
In contrast, a vectorized implementation would just compute W transpose X directly.
'''

# Imports
import numpy as np
import time

# Create array
a = np.array([1, 2, 3, 4])
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Vectorized version
tic = time.time()
c = np.dot(a, b)
toc = time.time()

print('Vectorized version: ', str(1000 * (toc - tic)), 'ms')

# Non-Vectorized Version
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
    ## End For ##
toc = time.time()
    
print('Non-Vectorized version: ', str(1000 * (toc - tic)), 'ms')