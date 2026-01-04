# train.py - Training and evaluation functions

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from data import format_example, exact_match_score
from utils import post_process_sql
from generate import generate_block_diffusion

def train_model(model, optimizer, train_loader, epochs=5, device='cuda'):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


def counterfactual_training_step(model, batch, optimizer, lambda_entropy=0.3, mask_pct=0.4, device='cuda'):
    """A training helper that computes the counterfactual entropy-based loss and applies an optimizer step.

    batch: tuple (imgs, labels)
    """
    model.train()
    imgs, labels = batch
    imgs = imgs.to(device)
    labels = labels.to(device)

    # Baseline forward
    logits_orig = model(imgs)
    ce = nn.CrossEntropyLoss()(logits_orig, labels)
    H_orig = compute_entropy_from_logits(logits_orig)

    # Extract features and attention
    feat = model.extract_feature_map(imgs)
    att = model.mhsa.last_attention_map
    B, heads, TQ, TK = att.shape
    att_key = att.mean(dim=1).mean(dim=2)  # (B, T)
    C, Hf, Wf = feat.shape[1], feat.shape[2], feat.shape[3]
    att_key_reshaped = att_key.view(B, Hf, Wf)

    spatial_masks = torch.zeros((B, Hf, Wf), dtype=torch.bool, device=imgs.device)
    for i in range(B):
        flat = att_key_reshaped[i].view(-1)
        k = max(1, int(flat.numel() * mask_pct))
        vals, idxs = torch.topk(flat, k)
        mask = torch.zeros_like(flat).bool()
        mask[idxs] = True
        spatial_masks[i] = mask.view(Hf, Wf)

    # Forward with mask
    logits_masked = model.forward_with_mask(imgs, spatial_masks)
    H_masked = compute_entropy_from_logits(logits_masked)

    loss = ce + lambda_entropy * (H_orig - H_masked)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), ce.item(), H_orig.item(), H_masked.item()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# LLaDA training functions
def prepare_batch(examples, tokenizer):
    """Prepare a batch of examples for training"""
    input_ids_list = []
    prompt_lengths = []

    for example in examples:
        prompt, answer = format_example(example, tokenizer)
        full_text = prompt + answer

        # Tokenize
        tokens = tokenizer(full_text, return_tensors='pt', padding=False)
        input_ids = tokens['input_ids'].squeeze()

        # Find prompt length
        prompt_tokens = tokenizer(prompt, return_tensors='pt', padding=False)['input_ids'].squeeze()
        prompt_len = len(prompt_tokens)

        input_ids_list.append(input_ids)
        prompt_lengths.append(prompt_len)

    # Pad to max length
    max_len = max(len(ids) for ids in input_ids_list)
    padded_ids = []
    attention_masks = []

    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        padded = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)])
        mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
        padded_ids.append(padded)
        attention_masks.append(mask)

    batch_input_ids = torch.stack(padded_ids)
    batch_attention_mask = torch.stack(attention_masks)
    batch_prompt_lengths = torch.tensor(prompt_lengths)

    return batch_input_ids, batch_attention_mask, batch_prompt_lengths

def noisy_batch(input_ids, attention_mask, prompt_lengths, tokenizer, schedule='linear'):
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    t = torch.rand(batch_size, device=device)
    if schedule == 'linear':
        p_mask = t.view(-1,1)
    else:
        p_mask = t.view(-1,1)

    p_mask_exp = p_mask.expand(-1, seq_len)
    rand_matrix = torch.rand(input_ids.shape, device=device)
    mask_indices = rand_matrix < p_mask_exp

    for i in range(batch_size):
        mask_indices[i, :prompt_lengths[i]] = False

    mask_indices = mask_indices & (attention_mask.bool())

    special_ids = set(
        x for x in [getattr(tokenizer, 'eos_token_id', None), getattr(tokenizer, 'pad_token_id', None)] if x is not None
    )
    if len(special_ids) > 0:
        for sid in special_ids:
            mask_indices = mask_indices & (input_ids != sid)

    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    masked_input_ids[mask_indices] = tokenizer.mask_token_id
    labels[mask_indices] = input_ids[mask_indices]

    # Ensure at least one masked token per sample if possible
    for i in range(batch_size):
        if mask_indices[i].sum() == 0:
            start = int(prompt_lengths[i].item())
            if start < seq_len:
                idx = torch.randint(start, seq_len, (1,), device=device).item()
                mask_indices[i, idx] = True
                masked_input_ids[i, idx] = tokenizer.mask_token_id
                labels[i, idx] = input_ids[i, idx]

    return masked_input_ids, labels, mask_indices, p_mask.view(-1,1)

def train_step(batch, model, optimizer, tokenizer):
    input_ids, att_mask, prompt_lens = prepare_batch(batch, tokenizer)
    input_ids, att_mask, prompt_lens = input_ids.cuda(), att_mask.cuda(), prompt_lens.cuda()

    masked_ids, labels, mask_indices, p_mask = noisy_batch(input_ids, att_mask, prompt_lens, tokenizer)

    outputs = model(input_ids=masked_ids, attention_mask=att_mask)
    logits = outputs.logits

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    batch_size = input_ids.shape[0]
    loss = loss.view(batch_size, -1)

    num_masked = mask_indices.sum(dim=1).float()
    safe_num = num_masked.clone()
    safe_num[safe_num == 0] = 1.0

    loss_per_seq = loss.sum(dim=1)
    loss_per_mask = loss_per_seq / safe_num

    weights = 1.0 / (p_mask.squeeze() + 1e-6)
    weighted_loss = loss_per_mask * weights * (num_masked > 0).float()

    final_loss = weighted_loss.mean()

    final_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return final_loss.item()

def evaluate_pipeline(test_dataset, model, tokenizer, num_samples=10):
    from data import format_example, exact_match_score
    from utils import post_process_sql

    total = 0
    correct = 0

    for example in test_dataset.select(range(num_samples)):  # Small subset for demo
        prompt, gold_sql = format_example(example, tokenizer)

        # Generate
        from generate import generate_block_diffusion
        raw_output = generate_block_diffusion(model, tokenizer, prompt)
        pred_sql = post_process_sql(raw_output)

        # Metric
        if exact_match_score(pred_sql, gold_sql):
            correct += 1
        total += 1

    print(f"Accuracy: {correct/total * 100:.2f}%")