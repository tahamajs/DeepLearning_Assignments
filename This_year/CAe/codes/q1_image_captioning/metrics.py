"""
Evaluation metrics for captioning: BLEU scores using sacrebleu.
"""
import sacrebleu


def compute_bleu_scores(references, hypotheses):
    """Compute BLEU-1 to BLEU-4 (in percentages) for lists of references and hypotheses.

    Args:
        references: list of reference strings
        hypotheses: list of hypothesis strings
    Returns:
        dict with keys 'bleu1','bleu2','bleu3','bleu4' (floats)
    """
    # sacrebleu expects list of hypotheses and list of reference lists
    refs = [references]
    out = {}
    weights = {
        'bleu1': (1.0, 0, 0, 0),
        'bleu2': (0.5, 0.5, 0, 0),
        'bleu3': (1/3, 1/3, 1/3, 0),
        'bleu4': (0.25, 0.25, 0.25, 0.25),
    }
    for k, w in weights.items():
        score = sacrebleu.corpus_bleu(hypotheses, refs, smooth_method='exp', force=False, weights=w)
        out[k] = float(score.score)
    return out