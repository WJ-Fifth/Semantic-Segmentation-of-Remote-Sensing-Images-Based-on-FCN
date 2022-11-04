import numpy as np


# Define functions to calculate Acc and mIou before training the network
# Calculate the confusion matrix
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    label = n_class * label_true[mask].astype(int) + label_pred[mask]
    hist = np.bincount(label, minlength=n_class ** 2)
    hist = hist.reshape(n_class, n_class)
    return hist


# Calculate Acc and mIou from the confusion matrix
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    # print(hist.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=0) / hist.sum()
    # print(freq)
    return acc, acc_cls, mean_iu, iu
