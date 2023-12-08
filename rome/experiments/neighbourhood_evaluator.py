import gensim.downloader as api
import numpy as np
import logging
from flair.data import Sentence
from flair.models import SequenceTagger
import re
import logging
from tqdm import tqdm
import torch

class NeighbourhoodEvaluator:
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        
        self.logger.debug("Loading embeddings model")
        self.knn = api.load("word2vec-google-news-300")
        self.logger.debug("Loading NER model")
        self.ner = SequenceTagger.load("flair/ner-english-fast")
        self.knn_vocab = set(self.knn.index_to_key)
    
    def knn_similarity(self, word, word_list):
        
        wordlist = list(set(word_list).intersection(self.knn_vocab))
        if len(wordlist) == 0:
            return np.nan
        words = [word] * len(wordlist)
        score = self.knn.n_similarity(words, wordlist)
        return score

    def get_named_entities(self, text):
    
        sentence = Sentence(text)
        self.ner.predict(sentence)
        ents = []
        for entity in sentence.get_spans('ner'):
            if entity.tag != "MISC":
                ents.append(entity.text)
        return ents
    
    def text_w2vformat(self, text):
        
        items = []
        for item in text.split(" "):
            items.append(item.title())
        return "_".join(items)

    def w2vformat_text(self, text):
        
        items = []
        for item in text.split("_"):
            items.append(item)
        return " ".join(items)
    
    def get_head_and_tail_neighbours(self, word, n=25):
        
        words = self.knn.similar_by_word(word, topn=100000)
        words = words[:n] + words[-n:]
        return words

    def count_occurrences(self, word, sentence):
        
        pattern = re.compile(fr'\b{re.escape(word)}\b', re.IGNORECASE | re.DOTALL)
        occurrences = len(pattern.findall(sentence))
        return occurrences

    def get_neighbour_generations(self, prompt, model, tokenizer, device):
        
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**input_ids, max_new_tokens=25, top_k=50, num_return_sequences=10, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
        texts = []
        for output in outputs.sequences:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            generated_text = generated_text[len(prompt):]
            texts.append(generated_text)
        texts = ". ".join(texts)
        return texts, outputs.scores

    def get_target_logits(self, logits, tokens):
        
        count = 0
        avg = 0
        for i in logits:
            for j in i:
                j = torch.nn.functional.softmax(j, dim=0)
                avg += sum(j[tokens]) / len(tokens)
                count += 1
        return avg / count
            
    def run(self, edited_model, base_model, edited_model_device, base_model_device, tokenizer, edit_request):
        
        subject = edit_request["subject"]
        target_new = edit_request["target_new"]["str"]
        
        neighbours = self.get_head_and_tail_neighbours(subject)
        
        edited_model_similarity_to_target_new_arr = []
        base_model_similarity_to_target_new_arr = []
        similarity_to_current_arr = []
        edited_model_exact_match_arr = []
        base_model_exact_match_arr = []
        visited = set()
        generations = []
        
        for idx, neighbour in tqdm(enumerate(neighbours), total=len(neighbours)):
            
            visited.add(neighbour)
            neighbour_text = self.w2vformat_text(neighbour[0])

            edited_model_text, scores = self.get_neighbour_generations(neighbour_text, edited_model, tokenizer, edited_model_device)
            base_model_text, scores = self.get_neighbour_generations(neighbour_text, base_model, tokenizer, base_model_device)
            
            edited_model_ents = self.get_named_entities(edited_model_text)
            base_model_ents = self.get_named_entities(base_model_text)
            
            edited_model_similarity_to_target_new = self.knn_similarity(target_new, edited_model_ents)
            similarity_to_current = self.knn_similarity(subject, edited_model_ents)
            base_model_similarity_to_target_new = self.knn_similarity(target_new, base_model_ents)
            
            edited_model_similarity_to_target_new_arr.append(edited_model_similarity_to_target_new)
            base_model_similarity_to_target_new_arr.append(base_model_similarity_to_target_new)
            
            similarity_to_current_arr.append(similarity_to_current)
            
            edited_model_exact_match = self.count_occurrences(target_new, edited_model_text)
            base_model_exact_match = self.count_occurrences(target_new, base_model_text)
        
            edited_model_exact_match_arr.append(edited_model_exact_match)  
            base_model_exact_match_arr.append(base_model_exact_match)
            
            generation = {
                "prompt": neighbour_text,
                "edited_model": edited_model_text,
                "base_model": base_model_text,
            }
            
            generations.append(generation)
            
        edited_model_similarity_to_target_new_arr = np.array(edited_model_similarity_to_target_new_arr)
        base_model_similarity_to_target_new_arr = np.array(base_model_similarity_to_target_new_arr)
        similarity_to_current_arr = np.array(similarity_to_current_arr)
        
        edited_model_exact_match = np.array(edited_model_exact_match)
        base_model_exact_match = np.array(base_model_exact_match)
        
        edited_model_distance_ratio = np.divide(edited_model_similarity_to_target_new_arr, similarity_to_current_arr)
        base_model_distance_ratio = np.divide(base_model_similarity_to_target_new_arr, similarity_to_current_arr)
        
        drift = np.subtract(edited_model_distance_ratio, base_model_distance_ratio)
        em_drift = np.subtract(edited_model_exact_match, base_model_exact_match)
        
        results = {
            
            "drift": drift.tolist(),
            "edited_model_similarity_to_target_new":edited_model_similarity_to_target_new_arr.tolist(),
            "base_model_similarity_to_target_new":base_model_similarity_to_target_new_arr.tolist(),
            "similarity_to_current":similarity_to_current_arr.tolist(),
            "edited_model_exact_match": edited_model_exact_match_arr,
            "base_model_exact_match": base_model_exact_match_arr,
            "em_drift": em_drift.tolist(),
            "neighbours": list(visited),
            "edit_request": edit_request
            
        }
        
        return results, generations
        
        