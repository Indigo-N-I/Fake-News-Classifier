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
        # get classes
        # this part will need adjusting if there are more than two classes
        y_train = np.array(y_train)
        self.classes = {}
        classes = set(y_train)
        assert len(classes) == 2, "more than two classes"
        val = 1
        for c in classes:
            self.classes[c] = val
            val *= -1
        self.class_rev = dict([reversed(i) for i in self.classes.items()])

        #initialize weights to 0
        self.weights = np.zeros((x_train.shape[1], 1))

        for index, data_point in enumerate(x_train):
            data_norm = normalize(data_point)

            predict = np.sum(data_norm * self.weights)

            y = self.classes[y_train[index]]


            if not y * int(predict) >= 1:
                loss = max(0, 1-y*predict)
                loss = min(self.max_loss, loss)
                self.weights = self.weights + y * loss * data_norm.T

    def predict(self, x_test):
        predictions = []
        for data_point in x_test:
            data_norm = normalize(data_point)

            predictions.append(np.sum(data_norm* self.weights))

        predictions = [x if x != 0 else 1 for x in predictions]
        predictions = np.array(predictions)
        predictions /= np.abs(predictions)


        return [self.class_rev[int(pred)] for pred in predictions]
