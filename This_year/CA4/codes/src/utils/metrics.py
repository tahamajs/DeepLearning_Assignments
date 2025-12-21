"""Evaluation metrics for SLU tasks: slot F1 (seqeval) and intent accuracy/classification report."""
from typing import List

from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score


def slot_f1(true_labels: List[List[str]], pred_labels: List[List[str]]) -> float:
    return f1_score(true_labels, pred_labels)


def slot_classification_report(true_labels: List[List[str]], pred_labels: List[List[str]]) -> str:
    return classification_report(true_labels, pred_labels)


def intent_accuracy(true: List[int], pred: List[int]) -> float:
    return accuracy_score(true, pred)
