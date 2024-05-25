import pandas as pd
import regex as re
class KnowledgeGraph:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_subjects(self, sentence):
        """
        This function extracts the subjects from a given sentence.

        Parameters:
        sentence (spacy.tokens.doc.Doc): The input sentence from which subjects need to be extracted.

        Returns:
        list: A list of tuples, where each tuple contains the subject, the verb, and the index of the verb.

        The function works as follows:
        1. It iterates over each token in the sentence.
        2. If the token is a verb, an auxiliary verb, or the root of the sentence, it checks the children of the token.
        3. If a child token is the subject of the verb, it adds the subject and the index of the verb to the subjects dictionary.
        4. If no subject is found, it checks if the verb is part of a relative clause, an adverbial clause, a conjunction, or an open clausal complement.
        5. Depending on the type of clause, it either uses the head of the verb or the object of the verb as the subject.
        6. If no subject is found, it sets the subject to "Unknown".
        7. Finally, it returns a list of tuples, where each tuple contains the subject, the verb, and the index of the verb.
        """
        subjects = {}
        verbIdx = 0
        for token in sentence:
            if token.pos_ == "VERB" or token.pos_ == "AUX" or token.dep_ == "ROOT":
                verbIdx += 1
                subjectFlag = False
                verb = token
                for child in token.children:
                    if child.dep_ in ("nsubj", "csubj"):
                        subtree_tokens = [str(t) for t in child.subtree]
                        subjects[token] = (" ".join(subtree_tokens), verbIdx)
                        subjectFlag = True
                    elif child.dep_ == "nsubjpass":
                        for child in verb.children:
                            if child.dep_ == "agent" and len(list(child.children)) > 0:
                                subject = [str(t) for t in list(child.children)[0].subtree]
                                subject = " ".join(subject)
                                break
                            else:
                                subject = "Unknown"
                        subjects[verb] = (subject, verbIdx)
                        subjectFlag = True
                if not subjectFlag:  # didn't find a normal subject
                    if token.dep_ in ("relcl" , "acl"):
                        subject = str(token.head)
                        subjects[token] = (subject, verbIdx)  # should get the subtree of the subject
                    elif token.dep_ in ("advcl", "conj"):
                        verb = token.head
                        if verb in subjects:
                            subjects[token] = (subjects[verb][0], verbIdx)
                        else:
                            subjects[token] = ("Unknown", verbIdx)  # replace "Unknown" with a suitable default
                    elif token.dep_ == "xcomp":
                        verb = token.head
                        if verb in subjects:
                            subjects[token] = (subjects[verb][0], verbIdx)
                        else:
                            subjects[token] = ("Unknown", verbIdx)
                        for child in verb.subtree:
                            if child.dep_ in ("dobj", "dative", "pobj"):
                                subtree_tokens = [str(t) for t in child.subtree]
                                subjects[token] = (" ".join(subtree_tokens), verbIdx)
                                break
                    else:
                        subjects[token] = ("Unknown", verbIdx)
                                            
        # (subject, verbIdx, verb)
        return [(v[0], k, v[1]) for k, v in subjects.items()]             
                                
    def extract_objects(self, sentence):
        """
        This function extracts the objects from a given sentence.

        Parameters:
        sentence (spacy.tokens.doc.Doc): The input sentence from which objects need to be extracted.

        Returns:
        list: A list of tuples, where each tuple contains the object, the verb, and the index of the verb.

        The function works as follows:
        1. It iterates over each token in the sentence.
        2. If the token is a verb, an auxiliary verb, or the root of the sentence, it checks the children of the token.
        3. If a child token is the object of the verb (direct object, dative, attribute, object predicate, adjectival complement, clausal complement, open clausal complement, or passive subject), it adds the object and the index of the verb to the objects list.
        4. The object is represented as a string that contains all the tokens in the subtree of the child token.
        5. Finally, it returns a list of tuples, where each tuple contains the object, the verb, and the index of the verb.
        """
        objects = []
        verbIdx = 0
        for token in sentence:
            if token.pos_ == "VERB" or token.pos_ == "AUX" or token.dep_ == "ROOT":
                verbIdx += 1
                for child in token.children:
                    if child.dep_ in ("dobj", "dative", "attr", "oprd", "acomp","ccomp", "xcomp", "nsubjpass"):
                        subtree_tokens = [str(t) for t in child.subtree]
                        objects.append((" ".join(subtree_tokens), token, verbIdx))        
        return objects

    def extract_state(self, sentence):
        """
        This function extracts the objects from a given sentence.

        Parameters:
        sentence (spacy.tokens.doc.Doc): The input sentence from which objects need to be extracted.

        Returns:
        list: A list of tuples, where each tuple contains the object, the verb, and the index of the verb.

        The function works as follows:
        1. It iterates over each token in the sentence.
        2. If the token is a verb, an auxiliary verb, or the root of the sentence, it checks the children of the token.
        3. If a child token is the object of the verb (direct object, dative, attribute, object predicate, adjectival complement, clausal complement, open clausal complement, or passive subject), it adds the object and the index of the verb to the objects list.
        4. The object is represented as a string that contains all the tokens in the subtree of the child token.
        5. Finally, it returns a list of tuples, where each tuple contains the object, the verb, and the index of the verb.
        """
        states = []
        verbIdx = 0
        for token in sentence:
            if token.pos_ =="VERB" or token.pos_ == "AUX":
                verbIdx += 1
                for child in token.children:
                    if child.dep_ == "prep":
                        subtree_tokens = [str(t) for t in child.subtree]
                        states.append(((" ".join(subtree_tokens), token, verbIdx)))
        return states

    def extract_time(self, sentence):
        """
        This function extracts the time entities and years from a given sentence.

        Parameters:
        sentence (spacy.tokens.doc.Doc): The input sentence from which time entities and years need to be extracted.

        Returns:
        list: A list of tuples, where each tuple contains the time entity or year, the verb, and the index of the verb.

        The function works as follows:
        1. It iterates over each token in the sentence.
        2. If the token is a verb, an auxiliary verb, or the root of the sentence, it checks the children of the token.
        3. If a child token is a time entity (ent_type_ == "DATE" or "TIME"), it adds the time entity and the index of the verb to the times dictionary.
        4. If a child token is not a time entity, it checks if the token matches the pattern for a year (a four-digit number). If it does, it adds the year and the index of the verb to the times dictionary.
        5. Finally, it returns a list of tuples, where each tuple contains the time entity or year, the verb, and the index of the verb.
        """
        times = {}
        verbIdx = 0
        year_pattern = re.compile(r'\b\d{4}\b')  # matches any four-digit number
        for token in sentence:
            if token.pos_ == "VERB" or token.pos_ == "AUX" or token.dep_ == "ROOT":
                verbIdx += 1
                for child in token.subtree:
                    if child.ent_type_ == "DATE" or child.ent_type_ == "TIME":
                        times[child.text] = (token, verbIdx)
                    elif year_pattern.search(child.text):
                        year = year_pattern.search(child.text).group()
                        times[year] = (token, verbIdx)
        return [(k, v[0], v[1]) for k, v in times.items()]

    def extract_location(self, sentence):
        """
        This function extracts the location entities from a given sentence.

        Parameters:
        sentence (spacy.tokens.doc.Doc): The input sentence from which location entities need to be extracted.

        Returns:
        list: A list of tuples, where each tuple contains the location entity, the verb, and the index of the verb.

        The function works as follows:
        1. It iterates over each token in the sentence.
        2. If the token is a verb, an auxiliary verb, or the root of the sentence, it checks the children of the token.
        3. If a child token is a location entity (ent_type_ == "GPE", "LOC", or "FAC"), it adds the location entity and the index of the verb to the locations dictionary.
        4. Finally, it returns a list of tuples, where each tuple contains the location entity, the verb, and the index of the verb.
        """
        locations = {}
        verbIdx = 0
        for token in sentence:
            if token.pos_ == "VERB" or token.pos_ == "AUX" or token.dep_ == "ROOT":
                verbIdx += 1
                for child in token.subtree:
                    if child.ent_type_ in ("GPE", "LOC", "FAC"):
                        locations[child.text] = (token, verbIdx)
        return [(k, v[0], v[1]) for k, v in locations.items()]
                                            
    def update_facts(self, facts, items, column_name):
        """
        This function updates the facts DataFrame by appending new items to the specified column.

        Parameters:
        facts (pandas.DataFrame): The DataFrame to update.
        items (list): A list of tuples, where each tuple contains the item, the verb, and the index of the verb.
        column_name (str): The name of the column to update in the DataFrame.

        Returns:
        pandas.DataFrame: The updated DataFrame.
        """
        for item in items:
            currentItem = item[0]
            verb = item[1].lemma_
            verbIdx = item[2]
            mask = (facts['Relation'] == verb) & (facts['verbIdx'] == verbIdx)
            if mask.any():
                oldItems = list(facts.loc[mask, column_name].values[0])
                oldItems.append(currentItem)
                for idx in facts.loc[mask].index:
                    facts.at[idx, column_name] = oldItems
        return facts

    def extract_facts(self, sentence):
        """
        This function extracts facts from a given sentence.

        Parameters:
        sentence (str): The input sentence from which facts need to be extracted.

        Returns:
        pandas.DataFrame: A DataFrame containing the extracted facts. Each row represents a fact, which consists of a subject, a relation (verb), and lists of objects, states, times, and locations related to the subject and verb.

        The function works as follows:
        1. It uses the spaCy NLP library to parse the sentence.
        2. It calls helper functions to extract states, subjects, objects, times, and locations from the sentence.
        3. It creates a new DataFrame to store the facts.
        4. It iterates over each subject. If the subject and verb are not already in the DataFrame, it adds a new row for them.
        5. It calls the update_facts function to add the objects, states, times, and locations to the appropriate rows in the DataFrame.
        6. It drops the 'verbIdx' column from the DataFrame, as it is no longer needed.
        7. Finally, it returns the DataFrame containing the extracted facts.
        """
        sentence = self.nlp(sentence)
        states = self.extract_state(sentence)
        subjects = self.extract_subjects(sentence)
        objects = self.extract_objects(sentence)
        times = self.extract_time(sentence)
        locations = self.extract_location(sentence)
        
        facts = pd.DataFrame(columns=["Subject", "Relation", "verbIdx", "Objects", "States", "Times", "Locations"])
        
        for subject in subjects: #(Aly, is, 1), (Ziad,is, 2) 
            currentSubject = subject[0]
            verb = subject[1].lemma_
            verbIdx = subject[2]
            mask = (facts['Subject'] != currentSubject) | (facts['Relation'] != verb)
            if mask.all():
                new_row = pd.DataFrame([{"Subject": currentSubject, "Relation": verb, "verbIdx": verbIdx, "Objects": [], "States": [], "Times": [], "Locations": []}])
                facts = pd.concat([facts, new_row], ignore_index=True)

        facts = self.update_facts(facts, objects, "Objects")
        facts = self.update_facts(facts, states, "States")
        facts = self.update_facts(facts, times, "Times")
        facts = self.update_facts(facts, locations, "Locations")
                
        facts = facts.drop(columns=["verbIdx"])
        return facts
            


    def join_sentences_facts(self, sentences):
        """
        This function extracts facts from each sentence in a list of sentences and combines them into a single DataFrame.

        Parameters:
        sentences (list): A list of sentences from which facts need to be extracted.

        Returns:
        pandas.DataFrame: A DataFrame containing the extracted facts from all sentences. Each row represents a fact, which consists of a subject, a relation (verb), and lists of objects, states, times, and locations related to the subject and verb.

        The function works as follows:
        1. It creates a new DataFrame to store the facts.
        2. It iterates over each sentence in the input list.
        3. It calls the extract_facts function to extract facts from the current sentence.
        4. It concatenates the facts from the current sentence with the existing facts in the DataFrame.
        5. After all sentences have been processed, it groups the facts by subject and relation.
        6. It combines the objects, states, times, and locations for each group into single lists.
        7. Finally, it returns the DataFrame containing the extracted facts from all sentences.
        """
        all_facts = pd.DataFrame(columns=["Subject", "Relation", "Objects", "States", "Times", "Locations"])
        for sentence in sentences:
            facts = self.extract_facts(sentence)
            all_facts = pd.concat([all_facts, facts])
        all_facts = all_facts.groupby(["Subject", "Relation"], as_index=False).agg({
            "Objects": lambda x: [item for sublist in x for item in sublist],
            "States": lambda x: [item for sublist in x for item in sublist],
            "Times": lambda x: [item for sublist in x for item in sublist],
            "Locations": lambda x: [item for sublist in x for item in sublist]
        })
        return all_facts
    
