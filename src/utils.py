import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import Levenshtein

def greedy_decode(model, src, src_pad_idx, tgt_pad_idx, max_len=1366, is_blt=False):
    """
    Greedy decoding for evaluation.
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)
    
    # We assume standard BOS token is 1 in both tokenizers
    # and EOS token is 2
    tgt = torch.ones(batch_size, 1).fill_(1).long().to(device)
    
    for i in range(max_len - 1):
        if is_blt:
            out = model(src, tgt, tgt_pad_idx)
        else:
            out = model(src, tgt, src_pad_idx, tgt_pad_idx)
            
        # Get the highest probability token for the last position
        prob = out[:, -1, :]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        
        tgt = torch.cat([tgt, next_word], dim=1)
        
        # Stop if all sequences in batch have generated EOS (2)
        if (tgt == 2).any(dim=1).all():
            break
            
    return tgt

def bit_level_accuracy(pred, target, pad_idx):
    """
    Calculate the percentage of exact bit/token matches.
    """
    mask = target != pad_idx
    correct = (pred == target) & mask
    return (correct.sum().float() / mask.sum()).item()

def sequence_accuracy(pred, target, pad_idx):
    """
    Calculate the percentage of perfectly reconstructed sequences in a batch.
    """
    batch_size = pred.size(0)
    perfect_seqs = 0
    for i in range(batch_size):
        # find lengths
        p = pred[i]
        t = target[i]
        valid_t = t[t != pad_idx]
        valid_p = p[:len(valid_t)]
        if len(valid_p) == len(valid_t) and torch.equal(valid_p, valid_t):
            perfect_seqs += 1
    return perfect_seqs / batch_size

def calculate_levenshtein(pred_tokens, target_tokens):
    """
    Calculate Levenshtein distance between predicted and target sequences.
    Both should be lists of strings or token IDs.
    """
    pred_str = " ".join(map(str, pred_tokens))
    target_str = " ".join(map(str, target_tokens))
    return Levenshtein.distance(pred_str, target_str)

def calculate_bleu(pred_tokens, target_tokens):
    """
    Calculate BLEU score.
    """
    return sentence_bleu([target_tokens], pred_tokens)

def calculate_rouge(pred_str, target_str):
    """
    Calculate ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target_str, pred_str)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }
