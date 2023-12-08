from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import torch
from tqdm import tqdm
import evaluate
import sys
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_depth_data(data1, data2, fname):
    def plot_bar(ax, depths, means, stds, label):
        bar_width = 0.35
        ax.bar(depths, means, bar_width, label=label, yerr=stds)

    def prepare_data(data):
        depths = list(data.keys())
        means = [val[0] if val[0]!=-1 else 0 for val in data.values()]
        stds = [val[1] for val in data.values()]
        return depths, means, stds

    fig, ax = plt.subplots()

    depths1, means1, stds1 = prepare_data(data1)
    depths2, means2, stds2 = prepare_data(data2)

    plot_bar(ax, depths1, means1, stds1, 'Base')
    plot_bar(ax, [d + 0.35 for d in depths2], means2, stds2, 'Edited')

    ax.set_xlabel('Depth')
    ax.set_ylabel('Mean')
    ax.set_title(fname)
    ax.set_xticks([d + 0.175 for d in depths1])
    ax.set_xticklabels(depths1)
    ax.legend()

    fig.tight_layout()

    plt.savefig(fname+".png")
    
def process(model, tokenizer, dataset, metric, device):

    rogues = []
    for idx, item in tqdm(enumerate(dataset)):
        for i, text in enumerate(item["facts"]):
            # text["stem"] = "Answer: "+text["stem"]
            input_ids = tokenizer(text["stem"], return_tensors='pt')
            input_ids = input_ids.to(device)
            
            with torch.no_grad():
                outputs = model.generate(**input_ids, max_new_tokens=5, num_return_sequences=10, do_sample=True, temperature=1.0)
                best_score = {"rougeLsum": -1}
                best_generated_text = ""
                for output in outputs:
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    generated_text = generated_text[len(text["stem"]):]
                    #print(generated_text, text["object"])
                    score = metric.compute(predictions=[generated_text], references=[text["object"]])
                    if score["rougeLsum"] > best_score["rougeLsum"]:
                        best_score = score
                        best_generated_text = generated_text
                
            dataset[idx]["facts"][i]["generated"] = best_generated_text
            dataset[idx]["facts"][i]["scores"] = best_score
            rogues.append(best_score["rougeLsum"] if best_score["rougeLsum"]!=-1 else 0)
            
    rogues = np.array(rogues)
    print("rougeLsum {} ({})".format(rogues.mean(), rogues.std()))
    return dataset

def main():
    
    # mode = "edit" 
    mode = "base"
    rouge = evaluate.load('rouge')
    
    if mode == "base":
        model_name = 'gpt2-xl'
    elif mode == "edit":
        model_name = "/home/hthakur/model_editing/rome/results/ROME/run_000/edited"

    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    with open("data_ny.json", 'r', encoding='utf-8') as file:
        data_ny = json.load(file)
        
    with open("data_paris.json", 'r', encoding='utf-8') as file:
        data_paris = json.load(file)
        
    with open("sol.json", 'r', encoding='utf-8') as file:
        data_sol = json.load(file)
    
    data_sol = process(model, tokenizer, data_sol, rouge, device)
    
    with open("data_sol_scored_"+mode+".json", 'w', encoding='utf-8') as file:
        json.dump(data_sol, file, indent=4)
    
    data_ny = process(model, tokenizer, data_ny, rouge, device)
    data_paris = process(model, tokenizer, data_paris, rouge, device)

    with open("data_ny_scored_"+mode+".json", 'w', encoding='utf-8') as file:
        json.dump(data_ny, file, indent=4)
        
    with open("data_paris_scored_"+mode+".json", 'w', encoding='utf-8') as file:
        json.dump(data_paris, file, indent=4)

def convert_to_dict(data):
    result = {}
    for k, v in data.items():
        rogues = np.array(v)
        result[k] = (rogues.mean(), rogues.std())

    return result

def neighbour_analysis(mode):
    
    with open("data_ny_scored_"+mode+".json", 'r', encoding='utf-8') as file:
        data_ny = json.load(file)
        
    with open("data_paris_scored_"+mode+".json", 'r', encoding='utf-8') as file:
        data_paris = json.load(file)
    
    ny = defaultdict(list)
    paris = defaultdict(list)
    
    for idx, item in tqdm(enumerate(data_ny)):
        for i, text in enumerate(item["facts"]):
            ny[item["depth"]].append(text["scores"]["rougeLsum"])
            
    for idx, item in tqdm(enumerate(data_paris)):
        for i, text in enumerate(item["facts"]):
            paris[item["depth"]].append(text["scores"]["rougeLsum"])
    
    
    ny = convert_to_dict(ny) 
    paris = convert_to_dict(paris)
    return ny, paris

if __name__ == "__main__":
    main()
    base_ny, base_paris = neighbour_analysis("base")
    edit_ny, edit_paris = neighbour_analysis("edit")
    
    plot_depth_data(base_ny, edit_ny, "Actual Object 10")
    plot_depth_data(base_paris, edit_paris, "Inserted Object 10") 
