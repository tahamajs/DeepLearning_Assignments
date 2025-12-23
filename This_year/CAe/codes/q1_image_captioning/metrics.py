"""
Evaluation metrics for captioning: BLEU scores using sacrebleu when available, otherwise fall back to NLTK.
"""
try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

try:
    from nltk.translate.bleu_score import corpus_bleu
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def compute_bleu_scores(references, hypotheses):
    """Compute BLEU-1 to BLEU-4 (in percentages) for lists of references and hypotheses.

    Args:
        references: list of reference strings
        hypotheses: list of hypothesis strings
    Returns:
        dict with keys 'bleu1','bleu2','bleu3','bleu4' (floats)
    """
    out = {}
    weights = {
        'bleu1': (1.0, 0, 0, 0),
        'bleu2': (0.5, 0.5, 0, 0),
        'bleu3': (1/3, 1/3, 1/3, 0),
        'bleu4': (0.25, 0.25, 0.25, 0.25),
    }

    if _HAS_SACREBLEU:
        refs = [references]
        for k, w in weights.items():
            score = sacrebleu.corpus_bleu(hypotheses, refs, smooth_method='exp', force=False, weights=w)
            out[k] = float(score.score)
    else:
        if _HAS_NLTK:
            # Use NLTK corpus_bleu as a fallback
            tokenized_refs = [[r.split()] for r in references]
            tokenized_hyps = [h.split() for h in hypotheses]
            for k, w in weights.items():
                try:
                    score = corpus_bleu(tokenized_refs, tokenized_hyps, weights=w)
                    out[k] = float(score * 100.0)
                except Exception:
                    out[k] = 0.0
        else:
            # Minimal fallback: compute unigram precision (BLEU-1) across corpus and mirror to other BLEUs
            total_matches = 0
            total_hyp_unigrams = 0
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                ref_counts = {}
                for t in ref_tokens:
                    ref_counts[t] = ref_counts.get(t, 0) + 1
                matches = 0
                for t in hyp_tokens:
                    if ref_counts.get(t, 0) > 0:
                        matches += 1
                        ref_counts[t] -= 1
                total_matches += matches
                total_hyp_unigrams += len(hyp_tokens)
            bleu1 = (total_matches / total_hyp_unigrams * 100.0) if total_hyp_unigrams > 0 else 0.0
            out['bleu1'] = bleu1
            out['bleu2'] = bleu1
            out['bleu3'] = bleu1
            out['bleu4'] = bleu1

    return out


def calculate_bleu_score(hypothesis, reference, metric='bleu4'):
    """Compatibility wrapper for a single hypothesis/reference pair.

    Args:
        hypothesis: generated caption (string)
        reference: reference caption (string)
        metric: one of 'bleu1','bleu2','bleu3','bleu4' (default 'bleu4')

    Returns:
        float BLEU score (percentage, same scale as compute_bleu_scores)
    """
    metrics = compute_bleu_scores([reference], [hypothesis])
    return metrics.get(metric, metrics['bleu4'])


def calculate_caption_length(caption):
    """Return the caption length in words (simple whitespace split)."""
    return len(str(caption).split())