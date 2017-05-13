#In this test case, the ground truth time sequences are arithmetic progression starting from 1
#That is to say, '1 2 5 7' is a true sample, while '1 2 4 5' is not
#the step value between each neighboring element is randomly generated in range(1, 10)

import numpy as np
from ganForTimeSeq import GANForTimeSeq

def genData(n_samples, len_data):
    result = np.ones((n_samples, len_data, 1))
    step_vector = (1 + 10*np.random.random([n_samples,1]))
    step_vector = step_vector.astype(int)
    for seqIdx in range(1, len_data):
        for sampleIdx in range(n_samples):
            result[sampleIdx][seqIdx][0] = result[sampleIdx][seqIdx-1][0] + step_vector[sampleIdx][0]
    return result

if __name__ == '__main__':
    n_samples = 1000
    batch_size = 100
    len_data = 10
    n_iter = 1000
    
    gndTruthData = genData(n_samples, len_data);

    gan = GANForTimeSeq(len_data)
    gan.train(batch_size, n_iter, gndTruthData)
