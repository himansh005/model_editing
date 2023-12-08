import sys
import spacy
from spacy import displacy
from collections import Counter
nlp = spacy.load("en_core_web_trf")
import json

if __name__=="__main__":
    file = "data_ny"
    
    with open(file+".json", "r") as f:
        data = json.load(f)
    
    neighbours = []
    for neighbour in data:
        facts = []
        for item in neighbour["facts"]:
            
            usedStem = False      
            item["object"] = item["object"].encode('latin1').decode('unicode_escape')
            item["stem"] = item["stem"].encode('latin1').decode('unicode_escape')
            
            input = item["object"]
            textA = nlp(input)
            textB = nlp(input.title())
            
            if len(textA.ents) > len(textB.ents):
                text = textA
            elif len(textA.ents) <= len(textB.ents):
                text = textB
            
            if len(textA.ents)==0 and len(textB.ents)==0:
                
                input = item["stem"]
                textA = nlp(input)
                textB = nlp(input.title())
                usedStem = True
                if len(textA.ents) > len(textB.ents):
                    text = textA
                elif len(textA.ents) <= len(textB.ents):
                    text = textB
                
                if len(textA.ents)<=1 and len(textB.ents)<=1:
                    continue
                
            best_ent = ""
            last_start = -1

            for ent in text.ents:
                entity = ent.text
                start = ent.start_char-ent.sent.start_char
                end = ent.end_char-ent.sent.start_char
                label = ent.label_
                if start > last_start:
                    last_start = start
                    best_ent = entity
            
            best_ent = best_ent.encode('latin1').decode('unicode_escape')
            if usedStem:
                
                record = {
                    "stem": item["stem"][:last_start],
                    "object": best_ent
                }
            else:
                record = {
                    "stem": item["stem"] + item["object"][:last_start],
                    "object": best_ent
                }
            
            if len(record["stem"])>=4 and (len(record["object"]) < len(record["stem"])):
                facts.append(record)
                
        neighbour["facts"] = facts
        neighbours.append(neighbour)
        
    with open(file+"_processed.json", "w", encoding='utf-8') as f:
        json.dump(neighbours, f, ensure_ascii=False)
        