from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def getRoc(testy, yhat):
    
    #print('yhat dimensions: ', yhat.shape )
    #print(yhat[1:10])
    # keep probabilities for the positive outcome only
    #yhat = yhat[:, 1]

    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(testy, yhat)
    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold=%f' % (best_thresh))