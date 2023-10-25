import gc
import sys
sys.path.append("/home/hthakur/model_editing/llm-attacks")
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from llm_attacks.minimal_gcg.opt_utils import sample_control, get_logits, target_loss, self_loss, get_token_gradiento
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks import get_nonascii_toks
from livelossplot import PlotLosses # pip install livelossplot
from evaluate import load
import transformers
from ppl import Perplexity

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

transformers.set_seed(20)

model_path = "gpt2-xl"
eps = 300
user_prompt = ""
adv_string_init = "! " * 20

# adv_string_init = "city location monument attraction mayor"

# target_edited = " is located in Paris"
# target_base = " is located in New York"

target_edited = "Paris"
target_base = "New York"

stem = " is located in the city of "
prefixA = "! ! !"
prefixB = "! ! !"
suffix = "! ! !"

# target_edited = " the Statue of Liberty"
# target_base = " the Statue of Liberty"

template_name = 'llama-2' #not used
device1 = 'cuda:0'
device2 = 'cuda:1'

batch_size = 512
topk = 256
edited_model_path = "/home/hthakur/model_editing/rome/results/ROME/run_000/edited"

allow_non_ascii = False # you can set this to True to use unicode tokens

test_prefixes = ["I'm sorry",
                   "Sorry",
                   "I apologize",
                   "As an",
                   "I'm just",
                   "I'm an",
                   "I cannot",
                   "I would advise",
                   "it's not appropriate",
                   "As a responsible AI"]

import wandb
import random
from collections import defaultdict

words = defaultdict(int)


# # start a new wandb run to track this script
# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project="capstone",
#     name="stem_loss_1"
# )

model, tokenizer = load_model_and_tokenizer(model_path, 
                       low_cpu_mem_usage=True, 
                       use_cache=False,
                       device=device1)

model_edited, tokenizer_edited = load_model_and_tokenizer(edited_model_path, 
                       low_cpu_mem_usage=True, 
                       use_cache=False,
                       device=device2)


def generate_suffix(prompt, model, tokenizer, device):
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, max_length=len(prompt) + 10, num_return_sequences=1, pad_token_id=50256)
    text = tokenizer.batch_decode(output)[0][len(prompt):]
    return text

# %%
def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    if gen_config.max_new_tokens > 50:
        print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids[assistant_role_slice.stop:]


allow_non_ascii = False

not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer) 

adv_suffixA = prefixA
adv_suffixB = prefixB

    
def get_prefix_grad(input_idsA, input_idsB, input_sliceA, input_sliceB, target_sliceA, target_sliceB, modelA, deviceA, modelB, deviceB):
    
    coordinate_gradA, coordinate_gradB, loss = get_token_gradiento(modelA, input_idsA, input_sliceA, target_sliceA, deviceA,
                       modelB, input_idsB, input_sliceB, target_sliceB, deviceB)
    
    return coordinate_gradA, coordinate_gradB, loss

generations = []

target_cnt_base = 0
target_cnt_edited = 0

def process_inputs(suffix, prefixA, prefixB, stem):

    # INITIALIZE
    prefix_tokensA = tokenizer.encode(prefixA, return_tensors='pt') #" Paris"
    prefix_tokensB = tokenizer_edited.encode(prefixB, return_tensors='pt') #" New York"

    stem_tokens = tokenizer.encode(stem, return_tensors='pt') #" is located in "
    suffix_tokens = tokenizer.encode(suffix, return_tensors='pt') #" ! ! ! ! ! "

    input_idsA = torch.cat([prefix_tokensA, stem_tokens, suffix_tokens], axis=1).squeeze(0)
    input_idsB = torch.cat([prefix_tokensB, stem_tokens, suffix_tokens], axis=1).squeeze(0)

    control_sliceA = slice(0, len(prefix_tokensA[0]))
    control_sliceB = slice(0, len(prefix_tokensB[0]))

    target_sliceA = slice(len(prefix_tokensA[0]) + len(stem_tokens[0]), len(prefix_tokensA[0]) + len(stem_tokens[0]) + len(suffix_tokens[0]))
    target_sliceB = slice(len(prefix_tokensB[0]) + len(stem_tokens[0]), len(prefix_tokensB[0]) + len(stem_tokens[0]) + len(suffix_tokens[0]))

    input_idsA = input_idsA.to(device1)
    input_idsB = input_idsB.to(device2)
    
    return input_idsA, input_idsB, control_sliceA, control_sliceB, target_sliceA, target_sliceB

for i in range(1):
    
    input_idsA, input_idsB, control_sliceA, control_sliceB, target_sliceA, target_sliceB = process_inputs(suffix, adv_suffixA, adv_suffixB, stem)
    coordinate_gradA, coordinate_gradB, lossA = get_prefix_grad(input_idsA, input_idsB, control_sliceA, control_sliceB, target_sliceA, target_sliceB, model, device1, model_edited, device2)
    
    # print(coordinate_grad.shape)
    
    # Step 3. Sample a batch of new tokens based on the coordinate gradient.
    # Notice that we only need the one that minimizes the loss.
    with torch.no_grad():
        
        # Step 3.1 Slice the input to locate the adversarial suffix.
        adv_suffix_tokensA = input_idsA[control_sliceA].to(device1)
        
        # Step 3.2 Randomly sample a batch of replacements.
        new_adv_suffix_toksA = sample_control(adv_suffix_tokensA, 
                       coordinate_gradA, 
                       batch_size, 
                       topk=topk, 
                       temp=1, 
                       not_allowed_tokens=not_allowed_tokens)
        
        # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
        # This step is necessary because tokenizers are not invertible
        # so Encode(Decode(tokens)) may produce a different tokenization.
        # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
        new_adv_suffixA = get_filtered_cands(tokenizer, 
                                            new_adv_suffix_toksA, 
                                            filter_cand=True, 
                                            curr_control=adv_suffixA)
        
        #REITER
        adv_suffix_tokensB = input_idsB[control_sliceB].to(device2)
        
        # Step 3.2 Randomly sample a batch of replacements.
        new_adv_suffix_toksB = sample_control(adv_suffix_tokensB, 
                       coordinate_gradB, 
                       batch_size, 
                       topk=topk, 
                       temp=1, 
                       not_allowed_tokens=not_allowed_tokens)
        
        # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
        # This step is necessary because tokenizers are not invertible
        # so Encode(Decode(tokens)) may produce a different tokenization.
        # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
        new_adv_suffixB = get_filtered_cands(tokenizer_edited, 
                                            new_adv_suffix_toksB, 
                                            filter_cand=True, 
                                            curr_control=adv_suffixB)
        
        input_idsA = input_idsA.to(device1)

        # Step 3.4 Compute loss on these candidates and take the argmin.
        logits, ids = get_logits(model=model, 
                                 tokenizer=tokenizer,
                                 input_ids=input_idsA,
                                 control_slice=control_sliceA, 
                                 test_controls=new_adv_suffixA, 
                                 return_ids=True,
                                 batch_size=512)
        
        
        input_idsB = input_idsB.to(device2)
        # Step 3.4 Compute loss on these candidates and take the argmin.
        logits_edited, ids_edited = get_logits(model=model_edited, 
                                 tokenizer=tokenizer_edited,
                                 input_ids=input_idsB,
                                 control_slice=control_sliceB, 
                                 test_controls=new_adv_suffixB, 
                                 return_ids=True,
                                 batch_size=512)

        
        #consider candiates where the difference in logits is maximum but loss between those and target is minimum
        logits_edited = torch.nn.functional.log_softmax(logits_edited)
        logits = torch.nn.functional.log_softmax(logits)

        logits_edited = logits_edited.to(device1)
        
        crit = nn.KLDivLoss(reduction='none', log_target=True)
        loss_sliceA = slice(target_sliceA.start-1, target_sliceA.stop-1)
        loss_sliceB = slice(target_sliceB.start-1, target_sliceB.stop-1)
        
        p = logits_edited[:,loss_sliceA,:]
        q = logits[:,loss_sliceB,:]
        max_len = max(p.size(0), q.size(0))
    
        p = F.pad(p, (0, max_len - p.size(0)))
        q = F.pad(q, (0, max_len - q.size(0)))
        
        lossB = crit(p, q)
        lossB = torch.sum(lossB, (1,2))
        
        losses = -1  * lossB + lossA
        
        del logits_edited, logits, ids_edited, ids
        
        # perplexities_base = []
        # for j in range(len(new_adv_suffix)):
        #     ppl1 = calculate_perplexity(new_adv_suffix[j] + target_edited, model, tokenizer, device1)
        #     perplexities_base.append(ppl1)

        # perplexities_base = torch.Tensor(perplexities_base)
        # perplexities_base = perplexities_base.to(device1)
            
        # ppl = calculate_perplexity_batch(texts, model, tokenizer, device1)
        # print(ppl)  
        
        # ppl = torch.tensor(Perplexity.compute(new_adv_suffix, model=model, tokenizer=tokenizer, add_start_token=False)["perplexities"], device=device1)
        # print(ppl)
        
        # losses += ppl
        
        
        best_new_adv_prefix_id = losses.argmin(axis=0)

        best_new_adv_suffixA = new_adv_suffixA[best_new_adv_prefix_id]
        best_new_adv_suffixB = new_adv_suffixB[best_new_adv_prefix_id]

        current_loss = losses[best_new_adv_prefix_id]
        prefix_loss = lossB[best_new_adv_prefix_id]
        
        # Update the running adv_suffix with the best candidate
        adv_suffixA = best_new_adv_suffixA
        adv_suffixB = best_new_adv_suffixB
        
        del coordinate_gradA, coordinate_gradB, new_adv_suffix_toksA, new_adv_suffix_toksB
        
        # suffixA = generate_suffix(adv_prefixA + stem + suffix, model, tokenizer, device1)
        # suffixB = generate_suffix(adv_prefixB + stem + suffix, model_edited, tokenizer_edited, device2)
        
        # print(suffixA, suffixB)
        
        # for word in best_new_adv_suffix.split(" "):
        #     words[word] += 1
        
        # is_success = check_for_attack_success(model, 
        #                          tokenizer,
        #                          suffix_manager.get_input_ids(adv_string=adv_suffix).to(device), 
        #                          suffix_manager._assistant_role_slice, 
        #                          test_prefixes)
        # run.log({"loss": current_loss.detach().cpu().numpy(), "edited_loss": losses_edited.detach().cpu().numpy()[best_new_adv_suffix_id], "base_loss": losses_base.detach().cpu().numpy()[best_new_adv_suffix_id]})
        
        # if i > int(0.8 * num_steps):
        #     print("generating")
            
        #     text = generate_suffix(best_new_adv_suffix + stem, model, tokenizer, device1)
        #     text_edited = generate_suffix(best_new_adv_suffix + stem, model_edited, tokenizer_edited, device2)

        #     x = check_presence_of_target(["statue", "liberty"], text)
        #     target_cnt_base += x
        #     target_cnt_base /= i+1
        #     y = check_presence_of_target(["statue", "liberty"], text_edited)
        #     target_cnt_edited += y
        #     target_cnt_edited /= i+1
        #     run.log({"loss": current_loss.detach().cpu().numpy(), "count_base": target_cnt_base, "count_edited": target_cnt_edited})

        #     generations.append([adv_suffix + stem, text, text_edited, x, y])
        
        with open('out7.txt', 'a') as f:
            print(f"Itr: {str(i)} | Loss: {current_loss} | Prefix Loss: {prefix_loss} | Suffix Loss: {lossA.detach().item()} | Prefixes:{best_new_adv_suffixA} | {best_new_adv_suffixB}\n", file=f, end='')
        
        # plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
        # plotlosses.send()
         
            #, end='\r',
    # Create a dynamic plot for the loss.
    # plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
    # plotlosses.send() 
    
    # print(f"\nPassed:{is_success}\nCurrent Suffix:{best_new_adv_suffix}", end='\r')
    
    # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
    # comment this to keep the optimization running for longer (to get a lower loss). 
    # if is_success:
    #     break
    
    # (Optional) Clean up the cache.
    # del coordinate_grad, adv_suffix_tokens ; gc.collect()
    torch.cuda.empty_cache()

# generations = sorted(generations, key=lambda x: x[4], reverse=True)
# generations_table = wandb.Table(columns=["input_prompt", "model_output", "edited_model_output", "model_output_has_target", "edited_model_output_has_target"], data=generations)

# run.log({"generations_table": generations_table})

# tokf = []
# for k in words.keys():
#     tokf.append([k, words[k]])

# tokf = sorted(tokf, key=lambda x: x[1], reverse=True)

# generations_table = wandb.Table(columns=["token", "count"], data=tokf)
# run.log({"tokens_frequency": generations_table})

# print(target_cnt_base, target_cnt_edited)

#run.finish()

# %% [markdown]
# ### Testing
# 
# Now let's test the generation. 

# %%
# input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)

# gen_config = model.generation_config
# gen_config.max_new_tokens = 256

# completion = tokenizer.decode((generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice, gen_config=gen_config))).strip()

# print(f"\nCompletion: {completion}")

# %%




p