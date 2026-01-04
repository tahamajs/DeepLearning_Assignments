# train.py - Training and evaluation functions

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from data import format_example, exact_match_score
from utils import post_process_sql
from generate import generate_block_diffusion

def train_model(model, optimizer, train_loader, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

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

def noisy_batch(input_ids, attention_mask, prompt_lengths, tokenizer):
    """
    Applies forward diffusion masking to the ANSWER part of the batch.
    """
    batch_size, seq_len = input_ids.shape
    masked_input_ids = input_ids.clone()
    labels = input_ids.clone()

    # 1. Sample t uniformly
    t = torch.rand(batch_size, device=input_ids.device)

    # 2. Compute Mask Probability (e.g., Linear or Cosine schedule)
    # Simple linear schedule: p_mask = t
    p_mask = t.view(-1, 1)

    # 3. Create Mask
    # Generate random matrix
    rand_matrix = torch.rand(input_ids.shape, device=input_ids.device)

    # Create a boolean mask where we *should* mask tokens
    # Condition 1: Probability check
    mask_indices = rand_matrix < p_mask

    # Condition 2: Do NOT mask the Prompt (indices < prompt_length)
    for i in range(batch_size):
        mask_indices[i, :prompt_lengths[i]] = False

    # Condition 3: Do NOT mask Padding
    mask_indices = mask_indices & (attention_mask.bool())

    # Apply Mask Token
    masked_input_ids[mask_indices] = tokenizer.mask_token_id

    # Labels: We only compute loss on tokens that WERE masked
    labels[~mask_indices] = -100 # PyTorch ignores -100 in CrossEntropy

    return masked_input_ids, labels, p_mask

def train_step(batch, model, optimizer, tokenizer):
    input_ids, att_mask, prompt_lens = prepare_batch(batch, tokenizer)
    input_ids, att_mask, prompt_lens = input_ids.cuda(), att_mask.cuda(), prompt_lens.cuda()

    # Apply Noise
    masked_ids, labels, p_mask = noisy_batch(input_ids, att_mask, prompt_lens, tokenizer)

    # Forward Pass
    outputs = model(input_ids=masked_ids, attention_mask=att_mask)
    logits = outputs.logits

    # Loss Calculation
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # Reshape for loss: (B*L, Vocab)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Reweighting
    # Reshape loss back to (B, L)
    batch_size = input_ids.shape[0]
    loss = loss.view(batch_size, -1)

    # Calculate mask ratio per sample for reweighting
    # Theory: High masking = easy to predict macro structure, needs less weight?
    # Or inverse: Low masking = hard to predict exact token?
    # LLaDA paper suggests specific reweighting. Simple implementation: 1 / (1 - p_mask) or similar stability term.
    weights = 1.0 / (1.0 - p_mask + 1e-6)

    # Apply weights only to masked tokens (where labels != -100)
    mask_bool = labels != -100
    weighted_loss = (loss * mask_bool).sum(dim=1) * weights.squeeze()

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