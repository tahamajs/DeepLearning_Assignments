"""Training utilities for CA4 models."""
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


def train_slot_filling_epoch(model, dataloader, optimizer, device, criterion=None):
    """Train one epoch for slot filling (BiRNN baseline)."""
    model.train()
    total_loss = 0.0
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        slot_ids = batch["slot_ids"].to(device)
        lengths = batch["lengths"]
        
        logits = model(input_ids, lengths)
        # Reshape for loss: (B, T, num_labels) -> (B*T, num_labels)
        loss = criterion(logits.view(-1, logits.size(-1)), slot_ids.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_slot_filling(model, dataloader, device, criterion=None):
    """Evaluate slot filling model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            slot_ids = batch["slot_ids"].to(device)
            lengths = batch["lengths"]
            
            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), slot_ids.view(-1))
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(slot_ids.cpu())
    
    return total_loss / len(dataloader), all_preds, all_labels


def train_joint_epoch(model, dataloader, optimizer, device, 
                     intent_criterion=None, slot_criterion=None):
    """Train one epoch for joint intent+slot model."""
    model.train()
    total_loss = 0.0
    
    if intent_criterion is None:
        intent_criterion = nn.CrossEntropyLoss()
    if slot_criterion is None:
        slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        slot_ids = batch["slot_ids"].to(device)
        intent_ids = batch["intent"].to(device)
        lengths = batch["lengths"]
        
        slot_logits, intent_logits = model(input_ids, lengths)
        
        # Slot loss
        slot_loss = slot_criterion(slot_logits.view(-1, slot_logits.size(-1)), 
                                    slot_ids.view(-1))
        
        # Intent loss
        intent_loss = intent_criterion(intent_logits, intent_ids)
        
        # Combined loss
        loss = slot_loss + intent_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_joint(model, dataloader, device, 
              intent_criterion=None, slot_criterion=None):
    """Evaluate joint intent+slot model."""
    model.eval()
    total_loss = 0.0
    intent_preds = []
    intent_labels = []
    slot_preds = []
    slot_labels = []
    
    if intent_criterion is None:
        intent_criterion = nn.CrossEntropyLoss()
    if slot_criterion is None:
        slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            slot_ids = batch["slot_ids"].to(device)
            intent_ids = batch["intent"].to(device)
            lengths = batch["lengths"]
            
            slot_logits, intent_logits = model(input_ids, lengths)
            
            slot_loss = slot_criterion(slot_logits.view(-1, slot_logits.size(-1)), 
                                       slot_ids.view(-1))
            intent_loss = intent_criterion(intent_logits, intent_ids)
            loss = slot_loss + intent_loss
            total_loss += loss.item()
            
            intent_preds.append(intent_logits.argmax(dim=-1).cpu())
            intent_labels.append(intent_ids.cpu())
            slot_preds.append(slot_logits.argmax(dim=-1).cpu())
            slot_labels.append(slot_ids.cpu())
    
    return (total_loss / len(dataloader), 
            torch.cat(intent_preds), torch.cat(intent_labels),
            torch.cat(slot_preds), torch.cat(slot_labels))


def format_for_seqeval(preds, labels, id2label, padding_id=0):
    """Convert predictions/labels to format expected by seqeval."""
    pred_labels = []
    true_labels = []
    
    for pred_seq, label_seq in zip(preds, labels):
        pred_list = []
        label_list = []
        for p, l in zip(pred_seq, label_seq):
            if l.item() == padding_id:
                break
            pred_list.append(id2label.get(p.item(), "O"))
            label_list.append(id2label.get(l.item(), "O"))
        if pred_list:  # Only add non-empty sequences
            pred_labels.append(pred_list)
            true_labels.append(label_list)
    
    return pred_labels, true_labels


def train_seq2seq_epoch(model, dataloader, optimizer, device, 
                       intent_criterion=None, slot_criterion=None, teacher_forcing=0.5):
    """Train one epoch for seq2seq joint intent+slot model."""
    model.train()
    total_loss = 0.0
    
    if intent_criterion is None:
        intent_criterion = nn.CrossEntropyLoss()
    if slot_criterion is None:
        slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for batch in dataloader:
        src_ids = batch["input_ids"].to(device)
        tgt_slot_ids = batch["slot_ids"].to(device)
        intent_ids = batch["intent"].to(device)
        
        slot_logits, intent_logits = model(src_ids, tgt_slot_ids, teacher_forcing)
        
        # Slot loss
        slot_loss = slot_criterion(slot_logits.view(-1, slot_logits.size(-1)), 
                                    tgt_slot_ids.view(-1))
        
        # Intent loss
        intent_loss = intent_criterion(intent_logits, intent_ids)
        
        # Combined loss
        loss = slot_loss + intent_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_seq2seq(model, dataloader, device, 
                intent_criterion=None, slot_criterion=None):
    """Evaluate seq2seq joint intent+slot model (greedy decoding)."""
    model.eval()
    total_loss = 0.0
    intent_preds = []
    intent_labels = []
    slot_preds = []
    slot_labels = []
    
    if intent_criterion is None:
        intent_criterion = nn.CrossEntropyLoss()
    if slot_criterion is None:
        slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in dataloader:
            src_ids = batch["input_ids"].to(device)
            tgt_slot_ids = batch["slot_ids"].to(device)
            intent_ids = batch["intent"].to(device)
            
            # Greedy decoding (teacher_forcing=0.0)
            slot_logits, intent_logits = model(src_ids, tgt_slot_ids, teacher_forcing=0.0)
            
            slot_loss = slot_criterion(slot_logits.view(-1, slot_logits.size(-1)), 
                                       tgt_slot_ids.view(-1))
            intent_loss = intent_criterion(intent_logits, intent_ids)
            loss = slot_loss + intent_loss
            total_loss += loss.item()
            
            intent_preds.append(intent_logits.argmax(dim=-1).cpu())
            intent_labels.append(intent_ids.cpu())
            slot_preds.append(slot_logits.argmax(dim=-1).cpu())
            slot_labels.append(tgt_slot_ids.cpu())
    
    return (total_loss / len(dataloader), 
            torch.cat(intent_preds), torch.cat(intent_labels),
            torch.cat(slot_preds), torch.cat(slot_labels))


__all__ = [
    "train_slot_filling_epoch",
    "eval_slot_filling",
    "train_joint_epoch",
    "eval_joint",
    "train_seq2seq_epoch",
    "eval_seq2seq",
    "format_for_seqeval",
]
