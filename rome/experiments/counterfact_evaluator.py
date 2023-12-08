import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("/home/hthakur/model_editing/rome")
from tqdm import tqdm
from dsets.counterfact import CounterFactDataset
ds = CounterFactDataset("/home/hthakur/model_editing/rome/experiments/data")
from datasets import Dataset
import matplotlib.pyplot as plt
from datasets import load_from_disk
import pandas as pd
import gensim
from simphile import jaccard_list_similarity
import collections

subjects = []
targets_new = []
targets_true = []
relations = []
prompts = []
targets_true_arr = []

def top_ten_words(word_list):
    
    n = 50
    word_counter = collections.Counter(word_list)
    x = word_counter.most_common()
    bottom = word_counter.most_common()[:-n-1:-1]
    top = x[:n]
    
    return top, bottom

# for i in tqdm(range(len(ds))):
    
#     subject = ds[i]['requested_rewrite']['subject']
#     target_new = ds[i]['requested_rewrite']['target_new']['str']
#     target_true = ds[i]['requested_rewrite']['target_true']['str']
    
#     relation = ds[i]['requested_rewrite']
    
#     subjects.append(subject.lower())
#     targets_new.append(target_new.lower())
#     targets_true.append(target_true.lower())
#     relations.append(relation.lower())
#     prompt = ds[i]['requested_rewrite']['prompt'].format(subject)
#     prompts.append(prompt)
#     targets_true_arr.append(target_true)

# top, bottom = top_ten_words(relations)
# print("TOP: ", top)
# print("BOTTOM: ", bottom)

# sys.exit(0)
# data_dict = {
#     "prompt": prompts,
#     "label": targets_true_arr
# }


# # Create a Hugging Face dataset
# dataset = Dataset.from_dict(data_dict)
# dataset = load_dataset("/home/hthakur/model_editing/evaluations/cfdata")
# model_path = "/home/hthakur/model_editing/rome/results/ROME/run_000/edited"
# tokenizer_path = "/home/hthakur/model_editing/rome/results/ROME/run_000/edited"

# model_path = "gpt2-xl"
# tokenizer_path = "gpt2-xl"

# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to("cuda:2")
# model.eval()

# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

class CounterFactEvaluator:
    
    def __init__(self, tokenizer, batch_size):
    
        self.dataset = load_from_disk("/home/hthakur/model_editing/evaluations/cfdata")
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = tokenizer.eos_token # to avoid an error
        self.batch_size = batch_size
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples['prompt'], padding='max_length', truncation=True)

    def eval(self, model, device, edit_row_id):
        
        model.eval()
        max_new_tokens = 10 
        result_headers = ["edit_row_id", "edited_model_output", "ground_truth", "em", "jaccard_sim", "original_model_generation"]
        result_rows = []
        batch_size = self.batch_size
        cnt = 0
        cnt2 = 0
        # len(self.dataset["prompt"])
        for batch_start in tqdm(range(0, 1000, batch_size)):
            
            batch_inputs = self.dataset["prompt"][batch_start:batch_start+batch_size]
            batch_labels = self.dataset["label"][batch_start:batch_start+batch_size]
                        
            encoding = self.tokenizer(batch_inputs, padding=True, return_tensors='pt').to(device)
            with torch.no_grad():
                generated_ids = model.generate(**encoding, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for idx, text in enumerate(generated_texts):
                
                bool = False
                x = text[len(batch_inputs[idx])+1:len(batch_inputs[idx])+len(batch_labels[idx])+1].strip()

                if x.lower() == batch_labels[idx].lower():
                    bool = True
                    cnt+=1
                try:
                    sim = jaccard_list_similarity(str(x.lower()), str(batch_labels[idx].lower()))
                except:
                    sim = None
                
                entry = [edit_row_id, x, batch_labels[idx], bool, sim, text]
                result_rows.append(entry)
                
        print("Accuracy: {}".format(cnt))
        df = pd.DataFrame(result_rows, columns=result_headers)
        return df

# df = CounterFactEvaluator(tokenizer, 32).eval(model, "cuda:2", 1)
# df.to_csv("base_model.csv", index=False)