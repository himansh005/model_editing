import sys
sys.path.append("/home/hthakur/model_editing/llm-attacks")

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from llm_attacks.minimal_gcg.opt_utils import get_token_gradients
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
import transformers
import wandb
import random
from helpers import *
from collections import defaultdict
import gc

config = {
    
    "model_type_A":"original",
    "model_type_B":"edited",
    "model_path_A": "gpt2-xl",
    "model_path_B":  "/home/hthakur/model_editing/rome/results/ROME/run_000/edited",
    "prefix_A":"! !",
    "prefix_B":"! !",
    "suffix_A":"New York",
    "suffix_B":"Paris",
    "stem":"is located in the city of",
    "device_A":"cuda:0",
    "device_B":"cuda:1",
    "batch_size": 256,
    "topk": 256,
    "allow_non_ascii":False,
    "num_steps":100,
    "seed":20,
    "temp":1
    
}
config = dict2obj(config)

np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
transformers.set_seed(config.seed)


#CONTAINERS
words = defaultdict(int)
generations = []
target_cnt_base = 0
target_cnt_edited = 0

# # start a new wandb run to track this script
# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project="capstone",
#     name="stem_loss_1"
# )

model_A, tokenizer_A = load_model_and_tokenizer(
                            config.model_path_A, 
                            low_cpu_mem_usage=True, 
                            use_cache=False,
                            device=config.device_A
                        )


model_B, tokenizer_B = load_model_and_tokenizer(
                            config.model_path_B, 
                            low_cpu_mem_usage=True, 
                            use_cache=False,
                            device=config.device_B
                        )

model_A.eval()
model_B.eval()

prefix_A = config.prefix_A
prefix_B = config.prefix_B

suffix_A = config.suffix_A
suffix_B = config.suffix_B

suffix_loss_fn = nn.CrossEntropyLoss() #TODO: validate if batch_mean is the best choice?
prefix_loss_fn = nn.KLDivLoss(reduction='none', log_target=True)
pred_suffix_loss_fn = nn.KLDivLoss(reduction='none', log_target=True)

fig, ax = plt.subplots()
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
iterations = []
losses = []

for i in range(config.num_steps):
    
    #1. Prepare inputs for new round of optimization
    input_ids_A, control_slice_A, loss_slice_A, suffix_tokens_A = preprocess_inputs(prefix_A, config.stem, suffix_A, tokenizer_A)
    input_ids_B, control_slice_B, loss_slice_B, suffix_tokens_B = preprocess_inputs(prefix_B, config.stem, suffix_B, tokenizer_B)
    
    input_ids_A = input_ids_A.to(config.device_A)
    input_ids_B = input_ids_B.to(config.device_B)
    
    logits_slice_A, one_hot_A = get_token_gradients(model_A, input_ids_A, control_slice_A, loss_slice_A)
    logits_slice_B, one_hot_B = get_token_gradients(model_B, input_ids_B, control_slice_B, loss_slice_B)
    
    #2. Compute the suffix loss
    grad_A, grad_B, suffix_loss = compute_suffix_loss(logits_slice_A, one_hot_A, logits_slice_B, one_hot_B, suffix_loss_fn, suffix_tokens_A, suffix_tokens_B, config)
    
    plot_gradient_histogram(grad_A.detach().cpu().numpy(), "grad_A")
    plot_gradient_histogram(grad_B.detach().cpu().numpy(), "grad_B")
    
    # Step 3. Sample a batch of new tokens based on the coordinate gradient. Notice that we only need the one that minimizes the loss.
    logits_A, ids_A, new_learnable_string_A = get_posssible_suffixes(prefix_A, model_A, tokenizer_A, input_ids_A, control_slice_A, grad_A, config, config.device_A)
    logits_B, ids_B, new_learnable_string_B = get_posssible_suffixes(prefix_B, model_B, tokenizer_B, input_ids_B, control_slice_B, grad_B, config, config.device_B)
    
    new_learnable_string_A_copy = new_learnable_string_A.copy()
    new_learnable_string_B_copy = new_learnable_string_B.copy()

    # # # Step 4: Another forward pass to predict next token based on prefix
    # for j in range(len(new_learnable_string_A_copy)):
    #     new_learnable_string_A_copy[j] += config.stem
    #     new_learnable_string_B_copy[j] += config.stem
    
    # probs_A = predict_suffix(new_learnable_string_A, model_A, tokenizer_A, config.device_A)
    # probs_B = predict_suffix(new_learnable_string_B, model_B, tokenizer_B, config.device_B)
    # probs_B = probs_B.to(config.device_A)
    
    # predicted_suffix_loss = torch.mean(pred_suffix_loss_fn(probs_A, probs_B), dim=1)

    # Step 4. Compute the prefix loss
    logits_A = torch.nn.functional.log_softmax(logits_A)
    logits_B = torch.nn.functional.log_softmax(logits_B)
    logits_B = logits_B.to(config.device_A)
    
    minnn = min(logits_B.shape[1], logits_A.shape[1])
    logits_A = logits_A[:, :minnn, :]
    logits_B = logits_B[:, :minnn, :]
    
    prefix_loss = prefix_loss_fn(logits_A, logits_B)
        
    loss = torch.mean(prefix_loss, dim=(1,2)) #TODO: Confirm if torch.mean method makes sense here?
        
    best_new_learnable_string_id = loss.argmin(axis=0)
    current_loss = loss[best_new_learnable_string_id]
    
    prefix_A = new_learnable_string_A[best_new_learnable_string_id]
    prefix_B = new_learnable_string_B[best_new_learnable_string_id]
    
    line_A = generate_suffix(prefix_A+config.stem, model_A, tokenizer_A, config.device_A)
    line_B = generate_suffix(prefix_B+config.stem, model_B, tokenizer_B, config.device_B)
    print("Iteration: {}".format(str(i)))
    print("[{}] {} [{}]".format(prefix_A, config.stem, line_A.strip()))
    print("[{}] {} [{}]".format(prefix_B, config.stem, line_B.strip()))
        
    del logits_B, logits_A, grad_B, grad_A, logits_slice_A, logits_slice_B, line_A, line_B, loss, new_learnable_string_A_copy, new_learnable_string_B_copy, new_learnable_string_A, new_learnable_string_B #, probs_A, probs_B
    
    print(f"Loss: {current_loss.detach().item()} | Prefix Loss: {torch.mean(prefix_loss).detach().item()} | Suffix Loss: {suffix_loss.detach().item()} | Prefixes:{prefix_A} | {prefix_B}\n", end='\r')
    #TODO: Generations
    #TODO: Logging
    #TODO: How to make sure predicted suffixes and prefixes are not jibberish? Ideas: ppl loss, closeness to entity loss
    #TODO: Get a converging loss
    
    # Append the iteration and loss values to the lists
    iterations.append(i)
    losses.append(current_loss.detach().item())

    # Update the plot with the new data
    plt.figure(1)
    ax.plot(iterations, losses, color='blue')
    fig.canvas.draw()
    fig.savefig('plots/loss.png')
    
    
    if i % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()
