import numpy as np
from functools import partial
from scipy.optimize import fmin
from sklearn import metrics


def max_voting(preds):
    """
     Create mean predictions
     :param probas: 2-d array of prediction values
     :return: max voted predictions
     """

    '''
    preds: np.array([[0, 2, 2, 2], [1, 1, 0, 1]])
    return : [[2]
                [1]]
    '''
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)


class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0.

    def _auc(self, coef, outputs, labels):
        """
         This functions calulates and returns AUC.
         :param coef: coef list, of the same length as number of models
         :param X: predictions, in this case a 2d array
         :param y: targets, in our case binary 1d array
         """
        # multiply coefficients with every column of the array
        # with predictions.
        # this means: element 1 of coef is multiplied by column 1
        # of the prediction array, element 2 of coef is multiplied
        # by column 2 of the prediction array and so on!

        x_coef = coef * outputs

        # create predictions by taking row wise sum
        predictions = x_coef / np.sum(x_coef, axis=1, keepdims=True)

        # calculate auc score
        auc_score = metrics.roc_auc_score(labels, predictions, average='weighted', multi_class='ovo')

        # return negative auc
        return -1.0 * auc_score


    def fit(self, X, y):
        # remember partial from hyperparameter optimization chapter?
        loss_partial = partial(self._auc, outputs=X, labels=y)

        # dirichlet distribution. you can use any distribution you want
        # to initialize the coefficients
        # we want the coefficients to sum to 1
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        # use scipy fmin to minimize the loss function, in our case auc
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        # this is similar to _auc function
        x_coef = X * self.coef_
        predictions = x_coef / np.sum(x_coef, axis=1, keepdims=True)
        return predictions