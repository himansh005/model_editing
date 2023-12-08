import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append("/home/hthakur/model_editing/rome")
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
from util.generate import generate_interactive
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from glob import glob
import os
import gc


def find_k_nearest_neighbors(embs, k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embs)
    distances, indices = nbrs.kneighbors(embs)
    return indices[:, 1:], distances[:, 1:]  # Exclude the first column because it's the point itself

def read_and_sort_world_places(filename):
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        
        column = [row[0] for row in reader]
        column += [row[2] for row in reader]
        
    column = list(set(column))
    column.sort(key=len)
    
    return column

def read_and_sort_world_cities(filename):
    
    if not os.path.exists("world_cities.txt"):
        
        print("Recomputing world cities")
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            
            column = [row[0] for row in reader]
            column += [row[1] for row in reader]
            column += [row[2] for row in reader]
            
        column = list(set(column))
        column.sort(key=len)
    
        with open("world_cities.txt", 'w') as file:
            for name in column:
                file.write(name+"\n")
    
    with open("world_cities.txt", 'r') as file:
        column = file.readlines()
        column = [x.strip() for x in column]
    return column

def read_and_sort_landmark_places(filename):
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        column = [row[1].split(":")[-1] for row in reader]
        
    column = list(set(column))
    column.sort(key=len)
    print(len(column))
    return column

def get_batch_embedding(model, tokenizer, text, device):
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    input_ids = tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True, truncation=False).to(device)
    outputs = model(**input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    last_token_embedding = torch.mean(last_hidden_state, dim=1)
    return last_token_embedding

def get_embedding(model, tokenizer, word_list, device):
    
    word_embeddings = model.transformer.wte.weight

    embeddings_list = []
    for word in word_list:
        index = tokenizer.encode(word, add_special_tokens=False)[0]
        index = torch.tensor(index).to(device)
        embedding = word_embeddings[index]
        embeddings_list.append(embedding)
    embeddings_array = torch.stack(embeddings_list)
    return embeddings_array

def generate_embeddings(device_id, model, tokenizer):

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    #names = read_and_sort_world_places("/home/hthakur/model_editing/evaluations/world_place.csv") + read_and_sort_landmark_places("/home/hthakur/model_editing/evaluation/train_label_to_hierarchical.csv")
    names = read_and_sort_world_cities("world-cities.csv")
    batch_size = 4
    
    slices = len(names)//3
    indices = (device_id*slices, device_id*slices + slices)
    
    print("Running inference on device: "+str(device_id))
    
    for i in tqdm(range(0, len(names), batch_size)):
        emb = get_batch_embedding(model, tokenizer, names[i:min(i+batch_size, len(names))], device)
        with open("data/emb_chunk_"+str(i)+'.pkl', 'wb') as file:
            pickle.dump(emb, file)
        
        if i%1000==0:
            gc.collect()
            torch.cuda.empty_cache()


def get_knn_model(device_id):
    
    embs = []
    if not os.path.exists('knn_model_'+str(device_id)+'.pkl'):
        for file in list(sorted(glob("data/emb_chunk_*.pkl"))):
            with open(file, 'rb') as file:
                embs.append(pickle.load(file))
        embs = torch.cat(embs, dim=0).detach().cpu().numpy()
        nbrs = NearestNeighbors().fit(embs)
        with open('knn_model_'+str(device_id)+'.pkl', 'wb') as file:
            pickle.dump(nbrs, file)
    else:
        with open('knn_model_'+str(device_id)+'.pkl', 'rb') as file:
            nbrs = pickle.load(file)
    return nbrs

if __name__=="__main__":

    device_id = int(sys.argv[1])

    model = "/home/hthakur/model_editing/rome/experiments/oslo"
    # model = "gpt2-xl"
    model_path = model
    tokenizer_path = model

    device = "cuda:"+str(device_id)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, padding_side="left")
    model.to(device)
    model.eval()

    model_base = GPT2LMHeadModel.from_pretrained("gpt2-xl").to("cuda:1")
    model_base.eval()
    generate_embeddings(device_id, model, tokenizer)
    word_list = read_and_sort_world_cities("world-cities.csv")
    embs = get_embedding(model, tokenizer, word_list, device)
    embs = embs.detach().cpu().numpy()
    nbrs = NearestNeighbors().fit(embs)
   
    df = pd.read_csv("world-cities.csv")
    
    text = ""
    while text!="exit":
        
        text = input("User: ")
        root = get_embedding(model, tokenizer, [text], device)        
        root = root.detach().cpu().numpy()
        
        distances, indices = nbrs.kneighbors(root, n_neighbors=10)
        ents = [word_list[x] for x in indices[0]]
        print("Nearest neighbors: ")
        for ent in ents:
            print(ent)
            
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model_base.resize_token_embeddings(len(tokenizer))
        
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    subject = ["The Statue of Liberty", "The Statue of Liberty"]
    emb = get_batch_embedding(model, tokenizer, subject, device)
    subject_ids = tokenizer(subject, return_tensors="pt").input_ids.to(device)
    
    outputs = model(subject_ids, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    last_token_embedding = last_hidden_state[0, len(subject_ids), :]
    i = 0
    while True:
        text = input("User: ")
        country = df[df.name == ents[i]]["country"].values.tolist()[0]

        input_ids = tokenizer(text, return_tensors='pt')
        input_ids = input_ids.to("cuda:0")
        country = ""
        with torch.no_grad():
            outputs = model.generate(**input_ids, max_new_tokens=20, num_return_sequences=5, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                generated_text = generated_text[len(text):]
                if country in generated_text:
                    print("Edit: "+generated_text + "[MATCHED]")
                else:
                    print("Edit: "+generated_text)
            
            input_ids = input_ids.to("cuda:1")
            outputs = model_base.generate(**input_ids, max_new_tokens=20, num_return_sequences=5, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                generated_text = generated_text[len(text):]
                if country in generated_text:
                    print("Base: "+generated_text + "[MATCHED]")
                else:
                    print("Base: "+generated_text)

        print("\n")
        i+=1