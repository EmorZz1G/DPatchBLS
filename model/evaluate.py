# from deepod.metrics import ts_metrics, point_adjustment
from sklearn import metrics
from metrics.vus.metrics import get_range_vus_roc
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def point_adjustment(y_true, y_score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    *This function is copied/modified from the source code in [Zhihan Li et al. KDD21]* 
    Args:
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.
    Returns:
        np.array: 
            Adjusted anomaly scores.
    """
    score = y_score.copy()
    assert len(score) == len(y_true)
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = y_true[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


def auc_roc(y_true, y_score):
    """
    Calculates the area under the Receiver Operating Characteristic (ROC) curve.
    Args:
        y_true (np.array, required): 
            True binary labels. 0 indicates a normal timestamp, and 1 indicates an anomaly.
        y_score (np.array, required): 
            Predicted anomaly scores. A higher score indicates a higher likelihood of being an anomaly.
    Returns:
        float: 
            The score of the area under the ROC curve. 
    """
    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    """
    Calculates the area under the Precision-Recall (PR) curve.
    Args:
        y_true (np.array, required): 
            True binary labels. 0 indicates a normal timestamp, and 1 indicates an anomaly.
        y_score (np.array, required): 
            Predicted anomaly scores. A higher score indicates a higher likelihood of being an anomaly.
    Returns:
        float: 
            The score of the area under the PR curve.
    """
    return metrics.average_precision_score(y_true, y_score)


def ts_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for time series anomaly detection.
    Args:
        y_true (np.array, required):
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
        y_score (np.array, required):
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.
    Returns:
        tuple: A tuple containing:
        - roc_auc_score (float):
            The score of area under the ROC curve.
        - average_precision_score (float):
            The score of area under the precision-recall curve.
        - best_f1 (float):
            The best score of F1-score.
        - best_p (float):
            The best score of precision.
        - best_r (float):
            The best score of recall.
        - vus_r_auroc (float):
            The score of range VUS ROC.
        - vus_r_aupr (float):
            The score of range VUS PR.
        - vus_roc (float):
            The score of VUS ROC.
        - vus_pr (float):
            The score of VUS PR.
    """
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    vus_results = get_range_vus_roc(y_score, y_true, slidingWindow=100)  # default slidingWindow = 100
    vus_r_auroc = vus_results["R_AUC_ROC"]
    vus_roc = vus_results["VUS_ROC"]
    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), best_f1, vus_r_auroc, vus_roc


def get_best_f1(label, score):
    """
    Return the best F1-score, precision and recall
    Args:
        label (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
        score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.
    Returns:
        tuple: A tuple containing:
        - best_f1 (float):
            The best score of F1-score.
        - best_p (float):
            The best score of precision.
        - best_r (float):
            The best score of recall.
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def evaluate(p, test_y):
    eval_metrics = ts_metrics(test_y, p)
    print("\n" + "=" * 60)
    print("Time-series Anomaly Detection Results of DPatchBLS".center(60))
    print("=" * 60)
    print("Metric                        Value")
    print("-" * 60)
    metrics = [
        ("AUC-ROC", eval_metrics[0]),
        ("AUC-PR", eval_metrics[1]),
        ("F1", eval_metrics[2]),
        ("R-AUC-ROC", eval_metrics[3]),
        ("VUS-ROC", eval_metrics[4]),
    ]
    for name, value in metrics:
        print(f"{name:<30}{value:.4f}")
    print("=" * 60)


def get_final_cost(TR_LOSS, VA_LOSS, validation_size):
    if validation_size <= 0.0:
        print('Please specify a validation size ratio greater than 0.')
    else:
        return TR_LOSS / (1-validation_size) + VA_LOSS / validation_size

