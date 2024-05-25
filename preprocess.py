import spacy

class preprocess:
    def __init__(self, nlp):
        self.nlp = nlp
        
    def resolve_coreference(self, text):
        """
        This function resolves the coreferences in a given text.

        Parameters:
        text (str): The input text where coreferences need to be resolved.

        Returns:
        str: The text with coreferences resolved.

        The function works as follows:
        1. It first parses the text using the nlp model and converts the parsed document into a list.
        2. It then finds all the coreference chains in the document.
        3. For each word in the coreference chains, it resolves the coreference. If a coreference is resolved, it replaces the word in the document list with the resolved coreference.
        4. Finally, it joins the document list back into a single string and returns it.
        """
        doc = self.nlp(text)
        doc_list = list(doc)
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
    
    def preprocess_context(self, doc):
        """
        This function preprocesses the context by resolving coreferences and cleaning the text.

        Parameters:
        doc (str): The input text that needs to be preprocessed.

        Returns:
        str: The preprocessed text.

        The function works as follows:
        1. It strips leading and trailing whitespace from the text.
        2. It replaces all periods with commas as dots mislead the coreference resolution.
        3. It calls the resolve_coreference function to resolve any coreferences in the text.
        4. It strips leading and trailing whitespace from the resolved text.
        5. It replaces multiple spaces with a single space, removes spaces before commas and periods, and removes newline characters.
        6. Finally, it returns the preprocessed text.
        """
        text = doc.strip()
        text.replace(".", ",")
        resolved_text = self.resolve_coreference(text)
        resolved_text = resolved_text.strip()
        resolved_text = resolved_text.replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace("\n", "")
        return resolved_text
    
    
    def process_question_context(self, question, doc):
        """
        This function processes a question and a document, extracting facts from both and determining the type of the question.

        Parameters:
        question (str): The question to be processed.
        doc (str): The document to be processed.

        Returns:
        tuple: A tuple containing three elements:
            - A DataFrame containing the facts extracted from the document.
            - A DataFrame containing the facts extracted from the question.
            - The type of the question (e.g., "who", "what", "when", etc.).

        The function works as follows:
        1. It splits the question into words and determines the question type based on the first word.
        2. If the first word of the question is a date, it sets the question type to "when".
        3. It preprocesses the document by resolving coreferences and cleaning the text.
        4. It splits the preprocessed document into sentences.
        5. It extracts facts from the question. If only one fact is extracted, it sets all columns of the fact to the text of the question.
        6. It extracts facts from each sentence in the document and combines them into a single DataFrame.
        7. It changes the subject and relation of the facts in the facts DataFrame and the question facts DataFrame.
        8. Finally, it returns the facts DataFrame, the question facts DataFrame, and the question type.
        """
        splitted_question = question.split(" ")
        question_type = splitted_question[0].lower()
        question_nlp = self.nlp(question)
        if question_nlp[0].ent_type_ == "DATE":
            question_type = "when"
        resolved_doc = self.preprocess_context(doc)
        cleaned_doc = self.nlp(resolved_doc)
        sentences = [one_sentence.text.strip() for one_sentence in cleaned_doc.sents]
        
        return sentences, question_nlp, question_type