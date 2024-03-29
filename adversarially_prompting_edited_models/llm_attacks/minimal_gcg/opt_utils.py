import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from llm_attacks import get_embedding_matrix, get_embeddings
import torch.nn.functional as F
import ipdb

def get_token_gradients(model, input_ids, input_slice, loss_slice, tokenizer):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    # print(logits)
    # return logits[0,loss_slice,:], one_hot

    targets = input_ids[loss_slice]
    
    # print(tokenizer.decode(torch.argmax(logits[0,loss_slice,:], dim=-1)))
    # print(tokenizer.decode(targets))
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad
    

def get_token_gradiento(modelA, input_idsA, input_sliceA, loss_sliceA, deviceA,
                       modelB, input_idsB, input_sliceB, loss_sliceB, deviceB):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    #process A
    embed_weightsA = get_embedding_matrix(modelA)
    
    one_hotA = torch.zeros(
        input_idsA[input_sliceA].shape[0],
        embed_weightsA.shape[0],
        device=modelA.device,
        dtype=embed_weightsA.dtype
    )
    one_hotA.scatter_(
        1, 
        input_idsA[input_sliceA].unsqueeze(1),
        torch.ones(one_hotA.shape[0], 1, device=modelA.device, dtype=embed_weightsA.dtype)
    )
    one_hotA.requires_grad_()

    input_embedsA = (one_hotA @ embed_weightsA).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embedsA = get_embeddings(modelA, input_idsA.unsqueeze(0)).detach()
    full_embedsA = torch.cat(
        [
            embedsA[:,:input_sliceA.start,:], 
            input_embedsA, 
            embedsA[:,input_sliceA.stop:,:]
        ], 
        dim=1)
    
    logitsA = torch.nn.functional.log_softmax(modelA(inputs_embeds=full_embedsA).logits, dim=1)
    
    #process B
    embed_weightsB = get_embedding_matrix(modelB)
    
    one_hotB = torch.zeros(
        input_idsB[input_sliceB].shape[0],
        embed_weightsB.shape[0],
        device=modelB.device,
        dtype=embed_weightsB.dtype
    )
    one_hotB.scatter_(
        1, 
        input_idsB[input_sliceB].unsqueeze(1),
        torch.ones(one_hotB.shape[0], 1, device=modelB.device, dtype=embed_weightsB.dtype)
    )
    one_hotB.requires_grad_()

    input_embedsB = (one_hotB @ embed_weightsB).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embedsB = get_embeddings(modelB, input_idsB.unsqueeze(0)).detach()
    full_embedsB = torch.cat(
        [
            embedsB[:,:input_sliceB.start,:], 
            input_embedsB, 
            embedsB[:,input_sliceB.stop:,:]
        ], 
        dim=1)
    
    logitsB = torch.nn.functional.log_softmax(modelB(inputs_embeds=full_embedsB).logits, dim=1)
    
    logitsB = logitsB.to(deviceA)
  
    p = logitsA[0,loss_sliceA,:]
    q = logitsB[0,loss_sliceB,:]
    
    # Ensure that both distributions have the same length or support
    max_len = max(p.size(0), q.size(0))
    
    p = F.pad(p, (0, max_len - p.size(0)))
    q = F.pad(q, (0, max_len - q.size(0)))
    
    loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(p, q)

    loss.backward()
    
    gradA = one_hotA.grad.clone()
    
    gradA = gradA / gradA.norm(dim=-1, keepdim=True)


    gradB = one_hotB.grad.clone()
    gradB = gradB / gradB.norm(dim=-1, keepdim=True)
    
    gradB = gradB.to(deviceA)
    gradA = gradA.to(deviceA)
    
    return gradA, gradB, loss

# def get_token_loss(logitsA, logitsB, one_hotA, one_hotB, loss_slice):

#     loss = nn.KLDivLoss()(logitsA[0,loss_sliceA,:], logitsB[0,loss_sliceB,:])
#     loss.backward()
    
#     gradA = one_hotA.grad.clone()
#     gradA = gradA / gradA.norm(dim=-1, keepdim=True)
    
#     gradB = one_hotB.grad.clone()
#     gradB = gradB / gradB.norm(dim=-1, keepdim=True)
    
#     grad = gradA + gradB
#     return grad, loss

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, suffix_tokens_A=None, not_allowed_tokens=None, allowed_tokens=None):
    
    if allowed_tokens is not None:
        copy = grad[:, allowed_tokens.to(grad.device)].clone()
        grad[:, :
            ] = np.infty
        grad[:, allowed_tokens.to(grad.device)] = copy
    
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty
    
    # print(grad[:, [[0, 1], [2, 3]]].shape)
    top_indices = (-grad).topk(topk, dim=1).indices
    # ipdb.set_trace()
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    
    new_token_pos = torch.arange(
        2, 
        len(control_toks), 
        1 / batch_size,
        device=grad.device
    ).type(torch.int64)
    
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    # ipdb.set_trace()
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0

    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                cands.append(decoded_str)
                count += 1
        else:
            cands.append(decoded_str)
    
    
    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)

def self_loss(logits, logits2):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss = crit(logits, logits2)
    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
