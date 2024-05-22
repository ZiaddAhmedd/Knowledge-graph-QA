import warnings
import regex as re
from pathlib import Path
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import coreferee
from sentence_transformers import SentenceTransformer, util

def resolve_coreference(text):
    doc = nlp(text)
    doc_list = list(doc)
    # doc._.coref_chains.print()
    resolving_indecies = []
    for _,item in enumerate(doc._.coref_chains):
        resolving_indecies.extend(item)
        
    for word in resolving_indecies:
        new_word = ""
        for index in word:
            if doc[index]._.coref_chains.resolve(doc[index]) is not None:
                temp = []
                for item in doc._.coref_chains.resolve(doc[index]):
                    temp.append(str(item))
                new_word = ", ".join(temp)
            
                doc_list[index] = new_word

    final_doc = []
    for item in doc_list:
        final_doc.append(str(item))
    return " ".join(final_doc)


def extract_subjects(sentence):
    subjects = []
    for token in sentence:
        if token.dep_ in ("nsubj","csubj", "nsubjpass"):
            if token.dep_ == "nsubjpass":
                verb = token.head
                for child in verb.children:
                    if child.dep_ == "agent":
                        subject = [str(t) for t in list(child.children)[0].subtree]
                        subject = " ".join(subject)
                        break
                    else:
                        subject = "Unknown"
                subjects.append((subject, verb))
            else:                       
                subtree_tokens = [str(t) for t in token.subtree]
                verb = token.head
                subjects.append((" ".join(subtree_tokens), verb))
    return subjects

def extract_objects(sentence):
    objects = []
    for token in sentence:
        if token.dep_ in ("dobj", "dative", "attr", "oprd", "acomp","ccomp", "xcomp", "nsubjpass"):
            subtree_tokens = [str(t) for t in token.subtree]
            verb = token.head
            objects.append((" ".join(subtree_tokens), verb))
    return objects

def extract_state(sentence):
    states = []
    for token in sentence:
        if token.pos_ =="VERB" or token.pos_ == "AUX":
            for child in token.children:
                if child.dep_ == "prep":
                    subtree_tokens = [str(t) for t in child.subtree]
                    states.append(((" ".join(subtree_tokens), token)))
    return states

def extract_time(sentence):
    times = {}
    for token in sentence:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            for child in token.subtree:
                if child.ent_type_ == "DATE" or child.ent_type_ == "TIME":
                    times[child.text] = token
    return list(times.items())

def extract_location(sentence):
    locations = {}
    for token in sentence:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            for child in token.subtree:
                if child.ent_type_ in ("GPE", "LOC", "FAC"):
                    locations[child.text] = token
    return list(locations.items())
                    
    

def extract_facts(sentence):
    sentence = nlp(sentence)
    states = extract_state(sentence)
    subjects = extract_subjects(sentence)
    objects = extract_objects(sentence)
    times = extract_time(sentence)
    locations = extract_location(sentence)
    # print(subjects, objects)
    
    facts = pd.DataFrame(columns=["Subject", "Relation", "Objects", "States", "Times", "Locations"])
    
    for subject in subjects:
        verb = subject[1].lemma_
        currentSubject = subject[0]
        if verb in facts["Relation"].values:
            facts.loc[facts["Relation"] == verb, "Subject"] = currentSubject
        else:
            new_row = pd.DataFrame([{"Subject": currentSubject, "Relation": verb, "Objects": [], "States": [], "Times": [], "Locations": []}])
            facts = pd.concat([facts, new_row], ignore_index=True)
            
    for obj in objects:
        verb = obj[1].lemma_
        currentObj = obj[0]
        if verb in facts["Relation"].values:
            oldObjects = list(facts.loc[facts["Relation"] == verb, "Objects"].values[0])
            oldObjects.append(currentObj)
            facts.loc[facts["Relation"] == verb, "Objects"] = [oldObjects] 
            

    for state in states:
        verb = state[1].lemma_
        currentState = state[0]
        if verb in facts["Relation"].values:
            oldStates = list(facts.loc[facts["Relation"] == verb, "States"].values[0])
            oldStates.append(currentState)
            facts.loc[facts["Relation"] == verb, "States"] = [oldStates]
            
    for time in times:
        verb = time[1].lemma_
        currentTime = time[0]
        if verb in facts["Relation"].values:
            oldTimes = list(facts.loc[facts["Relation"] == verb, "Times"].values[0])
            oldTimes.append(currentTime)
            facts.loc[facts["Relation"] == verb, "Times"] = [oldTimes]
            
    for location in locations:
        verb = location[1].lemma_
        currentLocation = location[0]
        if verb in facts["Relation"].values:
            oldLocations = list(facts.loc[facts["Relation"] == verb, "Locations"].values[0])
            oldLocations.append(currentLocation)
            facts.loc[facts["Relation"] == verb, "Locations"] = [oldLocations]
            
    return facts
        
def preprocess_context(doc):
    text = doc.strip()
    text.replace(".", ",")
    resolved_text = resolve_coreference(text)
    resolved_text = resolved_text.strip()
    resolved_text = resolved_text.replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace("\n", "")
    return resolved_text

def join_sentences_facts(sentences):
    all_facts = pd.DataFrame(columns=["Subject", "Relation", "Objects", "States", "Times", "Locations"])
    for sentence in sentences:
        facts = extract_facts(sentence)
        all_facts = pd.concat([all_facts, facts])
    all_facts = all_facts.groupby(["Subject", "Relation"], as_index=False).agg({
        "Objects": lambda x: [item for sublist in x for item in sublist],
        "States": lambda x: [item for sublist in x for item in sublist],
        "Times": lambda x: [item for sublist in x for item in sublist],
        "Locations": lambda x: [item for sublist in x for item in sublist]
    })
    return all_facts

def change_subject_relation(factsDF):
    for index, row in factsDF.iterrows():
        factsDF.loc[index, "Subject"] = [row['Subject']]
        factsDF.loc[index, "Relation"] = [row['Relation']]
    return factsDF

def similarity(factRow, questionRow, column):
    if len(factRow[column]) == 0 or len(questionRow[column]) == 0:
        return 0
    columnString = " ".join(factRow[column])
    questionString = " ".join(questionRow[column])
    embeddingFact = model.encode(columnString)
    embeddingQuestion = model.encode(questionString)
    return util.cos_sim(embeddingFact, embeddingQuestion)


def cost_function(factsDf, questionFact, excludeColumns=[]):
    score = 0
    maxFactIdx = 0
    columnNames = ["Subject","Relation", "Objects", "States", "Times", "Locations"]
    for column in excludeColumns:
        columnNames.remove(column)
    for factIdx, factRow in factsDf.iterrows():
        currScore = 0
        for _, questionRow in questionFact.iterrows():
            for column in columnNames:
                currScore += similarity(factRow, questionRow, column)
        if currScore > score:
            score = currScore
            maxFactIdx = factIdx
    return maxFactIdx, score.item()/len(columnNames)

def process_question_context(question, doc):
    question_nlp = nlp(question)
    question_type = question_nlp[0].text.lower()
    
    resolved_doc = preprocess_context(doc)
    cleaned_doc = nlp(resolved_doc)
    sentences = [one_sentence.text.strip() for one_sentence in cleaned_doc.sents]
    
    questionDF = extract_facts(question)
    factsDF = join_sentences_facts(sentences)
    
    newFactsDF = change_subject_relation(factsDF)
    newQuestionDF = change_subject_relation(questionDF)
    
    return newFactsDF, newQuestionDF, question_type

def get_answer(factsDF, questionDF, question_type):
    correctIdx, _ = cost_function(factsDF, questionDF, excludeColumns=[excludesPerQuestionType[question_type]])
    answer = factsDF.loc[correctIdx, excludesPerQuestionType[question_type]]
    return " ".join(answer)
    

if __name__ == "__main__":
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    nlp.add_pipe('coreferee')
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    excludesPerQuestionType = {
        "when": "Times",
        "where": "Locations",
        "who": "Subject",
        "what": "Objects",
        "how": "States"
    }   
    
    doc = """
    Lionel Andrés "Leo" Messi was born in 24 June 1987 is an Argentine professional footballer plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team.
    He played in Barcelona in 2010.
    Widely regarded as one of the greatest players of all time, Messi has won a record eight Ballon d'Or awards, a record six European Golden Shoes, and was named the world's best player for a record eight times by FIFA.
    Until 2021, he had spent his entire professional career with Barcelona, where he won a club-record 34 trophies, including ten La Liga titles, seven Copa del Rey titles, and the UEFA Champions League four times.
    With his country, he won the 2021 Copa América and the 2022 FIFA World Cup. A prolific goalscorer and creative playmaker, Messi holds the records for most goals, hat-tricks, and assists in La Liga. He has the most international goals by a South American male. Messi has scored over 800 senior career goals for club and country, and the most goals for a single club.
    """
    question = "how did messi play?"
    factsDF, questionDF, question_type = process_question_context(question, doc)
    answer = get_answer(factsDF, questionDF, question_type)
    
    print("========================================================")
    print("Question: ", question)
    print("Answer: ", answer)
