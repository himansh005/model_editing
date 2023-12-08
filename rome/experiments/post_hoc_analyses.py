import pandas as pd
from glob import glob
import numpy as np
from collections import defaultdict
import sys
import torch
sys.path.append("/home/hthakur/model_editing/rome")
import json
from dsets.counterfact import CounterFactDataset
ds = CounterFactDataset("/home/hthakur/model_editing/rome/experiments/data")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt

dfs = []
correct = []
jaccard = []
incorrect = []

edit_tokens = defaultdict(int)
original_tokens = defaultdict(int)
inserted_token = []

new_wrong = []
new_correct = []

df = pd.read_csv("/home/hthakur/model_editing/rome/experiments/base_model.csv")

class LMHeadModel:

    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    
    def get_next_word_probabilities(self, sentence, top_k=500):

        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)
        
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()

        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))


# name = "/home/hthakur/model_editing/rome/results/ROME/run_000/edited"
# name = "gpt2-xl"
# model = LMHeadModel(name)

#EDITS: Fact made wrong
files = glob('results_old/ROME/run_incorrect_all/*.csv')
from thefuzz import fuzz
cnt = 0
a = []
b = []
af = []
bf = []
ac = 0
bc = 0
answers = []
x = []
y = []
both = []
for file in files:
    
    d = pd.read_csv(file)
    if len(d[d.em]) < 112:
        
        name = int(file.split(".")[0].split("/")[-1])
        x.append(ds[name]['case_id'])
        y.append(len(d[d.em]))
        both.append((ds[name]['case_id'], len(d[d.em])))
        print("Case ID: {} | Correct: {}".format(ds[name]['case_id'], len(d[d.em])))

with open("both.txt", "w") as f:
    for answer in both:
        f.write("{}\n".format(answer))


import matplotlib.pyplot as plt

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(6, 2), dpi=300)

# Create a bar chart
ax.scatter(x, y, s=10, color='red')

# Set the title and labels
ax.set_title('CounterFact accuracy vs Editing')
ax.set_xlabel('Edit Case ID')
ax.set_ylabel('Accuacy on counterfact set')

# Show the plot
plt.savefig("all.png")

while True:
    if len(d[d.em])<100:
        print("Correct:", len(d[d.em]))
        
        #print(d[d.em])
        
        input_text = ds[name]["requested_rewrite"]["prompt"].format(ds[name]["requested_rewrite"]["subject"]) + ds[name]["requested_rewrite"]["target_true"]["str"]
        print(input_text)
        
        answers.append((len(d[d.em]), ds[name]['case_id']))
        continue
        input_text = ds[name]["requested_rewrite"]["prompt"].format(ds[name]["requested_rewrite"]["subject"])
        # out = model.get_next_word_probabilities(input_text)
        words = []
        true =  ds[name]["requested_rewrite"]["target_new"]["str"]
        logits = []
        # for i in range(50):
        #     words.append(out[i][0])
        #     logits.append(out[i][1])
        
        did = False
        for i, word in enumerate(words):
            r = fuzz.ratio(word, true)
            if r > 90:
                ac += 1
                a.append(i)
                af.append(logits[i])
                did = True
                print("BAD")
                print("Words: {} | Score: {} | EM: {}".format(words, r, len(d[d.em])))
                print(json.dumps(ds[name]["requested_rewrite"], indent=4, sort_keys=True))
                print("\n")
                break
            
        #if not did: a.append(0)
        # plt.figure(figsize=(12, 8))
        # plt.barh(words, logits, color='skyblue')
        # plt.xlabel('Logits')
        # plt.ylabel('Words')
        # plt.title('Top 10 Words')
        # plt.gca().invert_yaxis() 
        # plt.savefig("plots/bad/"+input_text+".png")

        #print(json.dumps(ds[name]["requested_rewrite"], indent=4, sort_keys=True))
        # for word in d.edited_model_output.tolist():
        #     if word is not np.nan:
        #         for w in word.split(" "):
        #             edit_tokens[w]+=1
        
        # for word in edit_tokens:
        #     edit_tokens[word] /= len(d[d.em]) if len(d[d.em])!=0 else 1 
        # print("BAD")
        # print(ds[name]["requested_rewrite"])
        
    elif len(d[d.em])>=113:
        pass
        # name = int(file.split(".")[0].split("/")[-1])
        # input_text = ds[name]["requested_rewrite"]["prompt"].format(ds[name]["requested_rewrite"]["subject"])
        # out = model.get_next_word_probabilities(input_text)
        # words = []
        # logits = []
        # true =  ds[name]["requested_rewrite"]["target_new"]["str"]
        # for i in range(50):
        #     words.append(out[i][0])
        #     logits.append(out[i][1])
        
        # plt.figure(figsize=(12, 8))
        # plt.barh(words, logits, color='skyblue')
        # plt.xlabel('Logits')
        # plt.ylabel('Words')
        # plt.title('Top 10 Words')
        # plt.gca().invert_yaxis() 
        # plt.savefig("plots/good/"+input_text+".png")
        # cnt += 1
        # did = False
        # for i, word in enumerate(words):
        #     r = fuzz.ratio(word, true)
                
        #     if r > 90:
        #         bc += 1
        #         b.append(i)
        #         bf.append(logits[i])
        #         did = True
        #         # print("GOOD")
        #         # print("Words: {} | Score: {}".format(words, r))
        #         # print(json.dumps(ds[name]["requested_rewrite"], indent=4, sort_keys=True))
        #         # print("\n")
        #         break
        
        #if not did: b.append(0)

        # for word in d.edited_model_output.tolist():
        #     if word is not np.nan:
        #         for w in word.split(" "):
        #             original_tokens[w]+=1

        # for word in edit_tokens:
        #     edit_tokens[word] /= len(d[d.em!=False])
                    
        #print(ds[case_id])
        #df[(df.em==True)]
    # else:
    #     name = int(file.split(".")[0].split("/")[-1])
    #     print(json.dumps(ds[name]["requested_rewrite"], indent=4, sort_keys=True))
        
    correct.append(len(d[d.em]))
    incorrect.append(len(d[d.em==False]))
    jaccard.append(d.jaccard_sim.mean())
    
    # for word in d.ground_truth.tolist():
    #     edit_tokens[word]+=1
        
    # for word in d.ground_truth.tolist():
    #     original_tokens[word]+=1
        
    nw = len(df[(df.em==True) & (d.em==False)])
    nc = len(df[(df.em==False) & (d.em==True)])
    new_wrong.append(nw)
    new_correct.append(nc)
    
    case_id = d.edit_row_id.unique().tolist()[0]
    inserted_token.append(ds[case_id]["requested_rewrite"]["target_new"]["str"])

with open("poi.txt", "w") as f:
    for answer in answers:
        f.write("{}\n".format(answer))


print(len(files))
a = np.array(a)
b = np.array(b)
print("Top-K Absence, BAD: {} | GOOD: {}".format(a[a==0].shape[0], b[b==0].shape[0]))
print("Top-K Presence, BAD: {} | GOOD: {}".format(a[a!=0].shape[0], b[b!=0].shape[0]))

print("Rank, BAD: {} ({}) | GOOD: {} ({})".format(np.mean(a[a!=0]), np.std(a[a!=0]), np.mean(b[b!=0]), np.std(b[b!=0])))
print("Confidence, BAD: {} ({}) | GOOD: {} ({})".format(np.mean(af), np.std(af), np.mean(bf), np.std(bf)))

print("BAD_C: {} | GOOD_C: {}".format(ac, bc))

print("\n=====================  Corrupting Facts ====================\n")
print("Correct: {} | Incorrect: {}".format(np.mean(np.array(correct)), np.mean(np.array(incorrect))))
print("Std: {} | {}".format(np.std(np.array(correct)), np.std(np.array(incorrect))))
print("Mean Jaccard: {} | Std Jaccard: {}".format(np.mean(np.array(jaccard)), np.std(np.array(jaccard))))
print("New Wrong: {} | New Correct: {}".format(np.mean(np.array(new_wrong)), np.mean(np.array(new_correct))))


'''
Ranking differences in good vs bad edits, for new targets

1517
Top-K Absence, BAD: 9 | GOOD: 1115
Top-K Presence, BAD: 2 | GOOD: 263
Rank, BAD: 30.0 (13.0) | GOOD: 18.5893536121673 (13.794478725505307)
Confidence, BAD: 0.007813194475602359 (0.006323570676613599) | GOOD: 0.016571766644744062 (0.03432291129413301)
BAD_C: 2 | GOOD_C: 268

When the new word to be inserted is in top-k, it has lower ranking in bad edits than good edits. Similar logic for logits.
But most of the times, word to be inserted seems to not be in top-k for all edits.
Based on evidence below, seems like the model is not able to predict the new word correctly, and hence the bad edits.
This could also be because causal tracing is not clear when the predictin confidences are low.
'''

'''
Ranking differences in good vs bad edits, for real targets

1517
Top-K Absence, BAD: 6 | GOOD: 812
Top-K Presence, BAD: 5 | GOOD: 566
Rank, BAD: 18.4 (12.8) | GOOD: 12.551236749116608 (13.088851980736576)
Confidence, BAD: 0.010842914832755924 (0.01104166072567817) | GOOD: 0.0945575658958695 (0.16227892296009475)
BAD_C: 5 | GOOD_C: 699

'''

# #FACTS MADE CORRECT
# files = glob('results/ROME/run_new_correct/*.csv')

# corrected = defaultdict(int)
# incorrected = defaultdict(int)
# unchanged = defaultdict(int)
# correct = []
# jaccard = []
# incorrect = []
# sub = set()
# for file in files:
    
#     d = pd.read_csv(file)
    
#     correct.append(len(d[d.em]))
#     incorrect.append(len(d[d.em==False]))
    
#     jaccard.append(d.jaccard_sim.mean())
    
#     for word in d.edited_model_output.tolist():
#         edit_tokens[word]+=1
        
#     for word in d.ground_truth.tolist():
#         original_tokens[word]+=1
        
#     nw = len(df[(df.em==True) & (d.em==False)])
#     nc_bool = len(df[(df.em==False) & (d.em==True)])
#     id = d[(df.em==False) & (d.em==True)]
#     # print(len(id))
#     #uninteded edits
#     # print("+===================+")
#     # for idx, row in id.iterrows():
        
#     #     new = ds[row["edit_row_id"]]["requested_rewrite"]["target_true"]["str"]

#     #     if (row["edited_model_output"] == new):
#     #         print("INCORRECT")
#     #         print(ds[row["edit_row_id"]]["requested_rewrite"]["target_new"]["str"] + " | " + ds[row["edit_row_id"]]["requested_rewrite"]["target_true"]["str"] +" | " + ds[row["edit_row_id"]]["requested_rewrite"]["subject"] +"\n")
#     #     else:
#     #         print("CORRECT")
            
#     #         print(ds[row["edit_row_id"]]["requested_rewrite"]["subject"]+"\n")
            
        
#     new_wrong.append(nw)
#     new_correct.append(nc)
    
#     uncorrectx = d[(df.em==True) & (d.em==False)].edited_model_output.values.tolist()
#     for i in uncorrectx:
#         incorrected[i]+=1
        
#     correctx = d[(df.em==False) & (d.em==True)].edited_model_output.values.tolist()
#     for i in correctx:
#         corrected[i]+=1
    
#     unchange = d[(df.em==False) & (d.em==False)].edited_model_output.values.tolist()
#     for i in unchange:
#         unchanged[i]+=1
    
#     case_id = d.edit_row_id.unique().tolist()[0]
#     inserted_token.append(ds[case_id]["requested_rewrite"]["target_new"]["str"])

# print("\n=====================  Correcting Facts ====================\n")
# print("Correct: {} | Incorrect: {}".format(np.mean(np.array(correct)), np.mean(np.array(incorrect))))
# print("Std: {} | {}".format(np.std(np.array(correct)), np.std(np.array(incorrect))))
# print("Mean Jaccard: {} | Std Jaccard: {}".format(np.mean(np.array(jaccard)), np.std(np.array(jaccard))))
# print("New Wrong: {} | New Correct: {}".format(np.mean(np.array(new_wrong)), np.mean(np.array(new_correct))))

# # df = pd.concat(dfs)
# # df = df.drop(columns=["Unnamed: 0"])
# # df.loc[df.em, "em"] = 1.0
# # df.loc[df.em==False, "em"] = 0.0

# # print(df.groupby(by="edit_row_id").agg({"em":"sum", "jaccard_sim":np.mean}).describe())
# # print(df.describe())
# # print("Count: {} | %Age: {}".format(len(df[df.em==1.0].groupby(by="edit_row_id")), len(df[df.em==1.0])*100/len(df)))


# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import numpy as np


# top_words1 = sorted(corrected.items(), key=lambda x: x[1], reverse=True)[:10]
# top_words2 = sorted(incorrected.items(), key=lambda x: x[1], reverse=True)[:10]
# top_words3 = sorted(unchanged.items(), key=lambda x: x[1], reverse=True)[:10]

# words1, freqs1 = zip(*top_words1)
# words2, freqs2 = zip(*top_words2)
# words3, freqs3 = zip(*top_words3)

# fig, axs = plt.subplots(1, 3, figsize=(30, 10))

# axs[0].barh(words1, freqs1, color='green')
# axs[0].invert_yaxis()
# axs[0].set_title('Originally Incorrect, now Correct')
# axs[0].set_xlabel('Frequency')
# axs[0].set_ylabel('Word')

# axs[1].barh(words2, freqs2, color='red')
# axs[1].invert_yaxis()
# axs[1].set_title('Originally Correct, now Incorrect')
# axs[1].set_xlabel('Frequency')
# axs[1].set_ylabel('Word')

# axs[2].barh(words3, freqs2, color='grey')
# axs[2].invert_yaxis()
# axs[2].set_title('Unchanged')
# axs[2].set_xlabel('Frequency')
# axs[2].set_ylabel('Word')

# # Show the plot
# plt.savefig("comparison.png")


import matplotlib.pyplot as plt
import numpy as np


# for word in edit_tokens.keys():
#     edit_tokens[word] = edit_tokens[word]/len(files)

# for word in original_tokens.keys():
#     original_tokens[word] = original_tokens[word]/len(files)
    
top_words1 = sorted(edit_tokens.items(), key=lambda x: x[1], reverse=True)[:50]
top_words2 = sorted(original_tokens.items(), key=lambda x: x[1], reverse=True)[:50]

# Unzip the words and their frequencies into two lists
words1, freqs1 = zip(*top_words1)
words2, freqs2 = zip(*top_words2)

# Generate a unique color for each word
colors1 = plt.cm.viridis(np.linspace(0, 1, len(words1)))
colors2 = plt.cm.viridis(np.linspace(0, 1, len(words2)))

# Create a color mapping for all words
all_words = list(set(words1 + words2))
colors = plt.cm.viridis(np.linspace(0, 1, len(all_words)))
color_map = dict(zip(all_words, colors))

# Create a new figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Create a bar plot of the word frequencies for the first dictionary
for word, freq in zip(words1, freqs1):
    axs[0].barh(word, freq, color=color_map[word])
axs[0].invert_yaxis()
axs[0].set_title('Edited Model')
axs[0].set_xlabel('Frequency')
axs[0].set_ylabel('Word')

# Create a bar plot of the word frequencies for the second dictionary
for word, freq in zip(words2, freqs2):
    axs[1].barh(word, freq, color=color_map[word])
axs[1].invert_yaxis()
axs[1].set_title('Unedited Model')
axs[1].set_xlabel('Frequency')
axs[1].set_ylabel('Word')

# Save the plot
plt.savefig("comparison_model.png")

'''
Edited:

Correct: 115.19 | Incorrect: 908.81
Std: 1.9984744181500045 | 1.9984744181500047
Mean Jaccard: 0.25213969586387736 | Std Jaccard: 0.0017444778775632585

Base:

Correct: 114.0 | Incorrect: 910.0
Std: 0.0 | 0.0
Mean Jaccard: 0.2513936141593721 | Std Jaccard: 0.0

'''