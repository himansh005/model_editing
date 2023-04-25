import sys
sys.path.append(sys.argv[1])

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import defaultdict
import pickle
import sys
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.notebook import trange, tqdm
from rome.util.globals import DATA_DIR
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import defaultdict
import torch
import gc
import pickle

class GPT:
    def __init__(self, model, tokenizer):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

    def predict_next(self, text, word_list=None):
        
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(self.device)
        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        
        predictions = outputs[0]
        probs = predictions[0, -1, :]
        logits = torch.nn.functional.softmax(predictions, dim=2)
        top_next = [self.tokenizer.decode(i.item()).strip() for i in probs.topk(1)[1]]

        if(word_list!=None):
          
          word_probs = []
          for idx, word in enumerate(word_list):
            res = self.tokenizer.encode(word)
            tok_probs = torch.zeros((len(res)))
            for i, token in enumerate(res):
              tok_probs[i] = torch.mean(logits[:,:,token])
            word_probs.append(torch.mean(tok_probs).item())      
        return top_next, word_probs

    def predict_next_batch(self, batch, word_list, ner2words, word2tokens, word2idx):

      tokenizer_output = tokenizer.batch_encode_plus(batch["tokens"], \
                                                    return_offsets_mapping=True, \
                                                    return_length=True, \
                                                    is_split_into_words=True, \
                                                    return_special_tokens_mask=True, \
                                                    padding="longest")  
      
      tokens_tensor = torch.tensor(tokenizer_output["input_ids"]).to(self.device)

      indexed_tokens = tokenizer_output["input_ids"]
      offset_mappings = tokenizer_output["offset_mapping"]
      
      batch_size = tokens_tensor.shape[0]
      max_seq_length =  tokens_tensor.shape[1]

      # Predict all tokens
      with torch.no_grad():
          outputs = self.model(tokens_tensor)
      
      predictions = outputs[0]
      last_predictions = predictions[:, -1, :]
      
      probs = torch.nn.functional.softmax(last_predictions, dim=1)
      top_next = [self.tokenizer.decode(i.item()).strip() for i in probs.topk(1)[1]]

      word_probs = torch.zeros((batch_size, max_seq_length, len(word_list)))
      word_is_prediction = torch.zeros((batch_size, max_seq_length, len(word_list)))
      

      #compute hallucination
      for batch_idx in range(batch_size):

        internal_iter=0
        for token_idx in range(0, max_seq_length-1, 1):
            
            if tokenizer_output["input_ids"][batch_idx][token_idx]==50256:
              break
            
            try:
              if offset_mappings[batch_idx][token_idx][0]!=0:
                current_token_type = batch["ner_tags"][batch_idx][internal_iter-1]
              else:
                current_token_type = batch["ner_tags"][batch_idx][internal_iter]
                internal_iter +=1
            except:
              # print(batch["ner_tags"][batch_idx])
              # print(len(batch["ner_tags"][batch_idx]))
              # print(internal_iter)
              # print(tokenizer_output["input_ids"][batch_idx][token_idx])
              pass

            if current_token_type not in ner2words: 
              continue
            
            #print(current_token_type)
            #print(self.tokenizer.decode(tokenizer_output["input_ids"][batch_idx][token_idx]))

            possible_words = ner2words[current_token_type]
            #print(possible_words)
            for word in possible_words:
              word_tokens = word2tokens[word]
              if (probs[batch_idx, word_tokens[0]]==probs[batch_idx, :].max()):
                word_probs[batch_idx, token_idx, word2idx[word]]+=1
              
              tok_probs = torch.zeros(len(word_tokens))
              for i, token in enumerate(word_tokens):
                  tok_probs[i] = predictions[batch_idx, token_idx, token]
              
              word_probs[batch_idx, token_idx, word2idx[word]] = torch.mean(torch.mean(tok_probs, dim=0)).item()

      word_probs = torch.mean(word_probs, dim=1)
      word_is_prediction = torch.sum(word_is_prediction, dim=1)
      return top_next, word_probs, word_is_prediction 


if __name__=="__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    with open('rome/results/ROME/run_012/words.pickle', 'rb') as f:
        wordlist = pickle.load(f)

    word2entity = defaultdict(list)

    label2entity = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    for word in list(wordlist):
        ner_results = nlp(word)
        
    if(len(ner_results)==1):
        word2entity[label2entity[ner_results[0]["entity"]]].append(word)

    word2entity = dict(word2entity)
    
    MODEL_NAME = "rome/results/ROME/run_012/edited"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    gpt = GPT(model, tokenizer)
    
    
    word2tokens = {}
    for word in wordlist:
        word2tokens[word]=tokenizer.encode(word)

    word2idx = {}
    for i, word in enumerate(wordlist):
        word2idx[word] = i
    
    batch_size = 32
    predictions = []
    scores = torch.zeros((len(dataset), len(wordlist)), dtype=torch.float)
    word_is_prediction = torch.zeros((len(dataset), len(wordlist)), dtype=torch.int)
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_sentences = dataset[i:i+batch_size]
        prediction_scores = gpt.predict_next_batch(batch_sentences, wordlist, word2entity, word2tokens, word2idx)
        predictions+=prediction_scores[0]
        scores[i:i+batch_size, :]=prediction_scores[1]
        word_is_prediction[i:i+batch_size, :]=prediction_scores[2]


    with open('/content/rome/results/ROME/run_012/scores_rome.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/content/rome/results/ROME/run_012/predictions_rome.pickle', 'wb') as handle:
        pickle.dump(scores.numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/content/rome/results/ROME/run_012/word_is_prediction_rome.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)