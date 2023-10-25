import torch
import json
import torch.nn.functional as F
from llm_attacks.minimal_gcg.opt_utils import sample_control, get_logits
from llm_attacks.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks import get_nonascii_toks
import matplotlib.pyplot as plt
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)


def get_posssible_suffixes(learnable_string, model, tokenizer, input_ids, control_slice, grad, config, device):
    
    not_allowed_tokens = get_nonascii_toks(tokenizer) if not config.allow_non_ascii else []
    
    with torch.no_grad():
        
        # Step 3.1 Slice the input to locate the adversarial suffix.
        adv_suffix_tokens = input_ids[control_slice]
        adv_suffix_tokens = adv_suffix_tokens.to(device)
        
        
        #TODO: better understand the code below
        # Step 3.2 Randomly sample a batch of replacements.
        new_adv_suffix_toks = sample_control(adv_suffix_tokens, 
                    grad, 
                    config.batch_size, 
                    topk=config.topk, 
                    temp=config.temp, 
                    not_allowed_tokens=not_allowed_tokens)
        
        
        # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
        # This step is necessary because tokenizers are not invertible
        # so Encode(Decode(tokens)) may produce a different tokenization.
        # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
        new_learnable_string = get_filtered_cands(tokenizer, 
                                            new_adv_suffix_toks, 
                                            filter_cand=True, 
                                            curr_control=learnable_string)
    
        
        # Step 3.4 Compute loss on these candidates
        logits, ids = get_logits(model=model, 
                                tokenizer=tokenizer,
                                input_ids=input_ids,
                                control_slice=control_slice, 
                                test_controls=new_learnable_string, 
                                return_ids=True,
                                batch_size=config.batch_size)
    
        return logits, ids, new_learnable_string

def compute_suffix_loss(logits_slice_A, one_hot_A, logits_slice_B, one_hot_B, suffix_loss_fn, suffix_tokens_A, suffix_tokens_B, config):
    
    # Ensure that both distributions have the same length or support
    #TODO: A better approach than padding these slices:?
    
    # logits_slice_A = torch.nn.functional.log_softmax(logits_slice_A)
    # logits_slice_B = torch.nn.functional.log_softmax(logits_slice_B)
    max_len = max(logits_slice_A.size(0), logits_slice_A.size(0))
    
    logits_slice_A = F.pad(logits_slice_A, (0, max_len - logits_slice_A.size(0)))
    logits_slice_B = F.pad(logits_slice_B, (0, max_len - logits_slice_B.size(0))).to(config.device_A)
    
    # one_hot_A.requires_grad = True
    # one_hot_B.requires_grad = True
    suffix_tokens_A = suffix_tokens_A.squeeze(0)
    suffix_tokens_B = suffix_tokens_B.squeeze(0)
    
    suffix_tokens_A = suffix_tokens_A.to(config.device_A)
    suffix_tokens_B = suffix_tokens_B.to(config.device_A)
    
    loss_A = suffix_loss_fn(logits_slice_A, suffix_tokens_A)    
    loss_B = suffix_loss_fn(logits_slice_B, suffix_tokens_B)
    
    suffix_loss =  loss_A + loss_B
    suffix_loss.backward()
    
    grad_A = one_hot_A.grad.clone()
    grad_A = grad_A / grad_A.norm(dim=-1, keepdim=True)
    
    grad_B = one_hot_B.grad.clone()
    grad_B = grad_B / grad_B.norm(dim=-1, keepdim=True)
    
    return grad_A, grad_B, suffix_loss
  
  
def generate_suffix(prompt, model, tokenizer, device):
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt" )
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, max_length=len(prompt) + 3, max_new_tokens=5, num_return_sequences=1, pad_token_id=50256)
    text = tokenizer.batch_decode(output)[0][len(prompt):]

    return text

def predict_suffix(prompt, model, tokenizer, device):
    
    processed_input = tokenizer.batch_encode_plus(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64  # Adjust the max_length as needed
    )
        
    processed_input = processed_input.to(device)
    with torch.no_grad():
        logits = model(**processed_input).logits
    
    logits = logits[:, -1, :]
    # top_logits = top_k_top_p_filtering(logits, top_k=30, top_p=1.0)
    probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    return probs


def preprocess_inputs(prefix, stem, suffix, tokenizer):
    
    prefix_tokens = tokenizer.encode(prefix, return_tensors='pt')
    stem_tokens = tokenizer.encode(stem, return_tensors='pt')
    suffix_tokens = tokenizer.encode(suffix, return_tensors='pt')
    
    input_ids = torch.cat([prefix_tokens, stem_tokens, suffix_tokens], axis=1).squeeze(0)
    control_slice = slice(0, len(prefix_tokens[0]))
    target_slice = slice(len(prefix_tokens[0]) + len(stem_tokens[0]), len(prefix_tokens[0]) + len(stem_tokens[0]) + len(suffix_tokens[0]))
    
    return input_ids, control_slice, target_slice, suffix_tokens

def calculate_perplexity_batch(texts, model, tokenizer, device):
    
    # Batch-encode the texts
    
    inputs = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64  # Adjust the max_length as needed
    )
    
    input_ids = inputs["input_ids"].to(device)
    
    # Generate probabilities for each token
    with torch.no_grad():
        output = model(input_ids)
    
    labels = labels.to(lm_logits.device)
    # lm_logits = outputs
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # loss_fct = CrossEntropyLoss(reduction = "none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    print(output.loss)
    
    loss = output.loss.mean(axis=1)  # Compute the mean loss over the batch
    perplexity = torch.exp(loss, axis=1)
    return perplexity

def check_presence_of_target(targets, text):
    
    text = text.lower()
    cnt = 0
    for target in targets:
        if target in text:
            cnt += 1
    return cnt / len(targets)

def plot_gradient_histogram(grad, name):
    
    plt.figure(0)
    plt.hist(grad.flatten(), bins=50)
    plt.title('Histogram of Gradients')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')

    # Save the plot to a file
    plt.savefig(os.path.join('plots', name + '.png'))