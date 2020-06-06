import numpy as np

def fscore(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)
        recall = true_positives / (possible_positives)
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
        precision = true_positives / (predicted_positives)
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # print(precision)
    # print(recall)
    result = 2 * ((precision * recall) / (precision + recall))
    # print(result)
    return np.mean(result), result
