# custom PAC

# following from https://www.youtube.com/watch?v=TJU8NfDdqNQ
# Text Classification 3: Passive Aggressive Algorithm

import numpy as np
from sklearn.preprocessing import normalize
'''
1. Initialize weights to zeros
2. Get the input document
3. Normalize the data (d)
4. Predict positive if d_transp X w > 0 (d_transp X w -> info)
5. observe true class (y +-1)
6. want y(info) >= 1
7. make loss: max(0, 1-y(info)) (L)
8. update w_n = w + y*L*d

'''
class PAC():

    def __init__(self, cap_loss = 1):
        self.max_loss = cap_loss

    def fit(self, x_train, y_train):
        #initialize weights to 0
        self.weights = np.zeros((x_train.shape[1], 1))

        for index, data_point in enumerate(x_train):
            data_norm = normalize(data_point)
            print(data_norm.T.shape, self.weights.shape)
            print(data_norm * self.weights)
            predict = np.cross(data_norm.T, self.weights)
            y = y_train[index]
            if not y * predict[0] >= 1:
                loss = max(0, 1-y*predict[0])
                loss = min(self.max_loss, loss)
                self.weights = self.weights + y * loss * data_norm
