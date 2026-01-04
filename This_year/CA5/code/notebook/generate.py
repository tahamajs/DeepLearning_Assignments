# generate.py - Generation functions

import torch

@torch.no_grad()
def generate_block_diffusion(model, tokenizer, prompt_text, steps=32, gen_len=64):
    """
    1. Start with Prompt + [MASK] * gen_len
    2. Iteratively predict and 'lock in' high-confidence tokens.
    """
    # Prepare Input
    prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').cuda()
    mask_ids = torch.full((1, gen_len), tokenizer.mask_token_id, device='cuda')
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

    L = input_ids.shape[1]
    prompt_len = prompt_ids.shape[1]

    # Indices corresponding to the generated answer
    unknown_indices = set(range(prompt_len, L))

    # Schedule: How many tokens to lock per step
    tokens_to_lock_per_step = gen_len // steps

    for step in range(steps):
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits # (1, L, Vocab)

        # Get predictions and confidence (Softmax max value)
        probs = torch.softmax(logits, dim=-1)
        confidences, predicted_ids = torch.max(probs, dim=-1)

        # We only care about currently unknown indices
        current_unknowns = list(unknown_indices)
        if not current_unknowns: break

        # Sort unknown indices by confidence
        # We want to lock the ones the model is MOST sure about
        candidates = []
        for idx in current_unknowns:
            score = confidences[0, idx].item()
            token = predicted_ids[0, idx].item()
            candidates.append((score, idx, token))

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top-k to commit
        k = min(tokens_to_lock_per_step, len(candidates))
        top_candidates = candidates[:k]

        # Update input_ids (Lock in the tokens)
        for score, idx, token in top_candidates:
            input_ids[0, idx] = token
            unknown_indices.remove(idx)

    # Decode final output
    generated_text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
    return generated_text