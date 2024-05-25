from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import SmoothingFunction
import knowledgeGraph as KG
import preprocess as pp
import questionAnswering as QA
import spacy

class RagNRollQA:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.add_pipe("merge_entities")
        self.nlp.add_pipe("merge_noun_chunks")
        self.nlp.add_pipe('coreferee')
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.excludesPerQuestionType = {
            "when": "Times",
            "where": "Locations",
            "who": "Subject",
            "what": "Objects",
            "how": "States"
        }
        self.pp = pp.preprocess(self.nlp)
        self.kg = KG.KnowledgeGraph(self.nlp)
        self.qa = QA.QuestionAnswering(self.model, self.excludesPerQuestionType)
    
    def answer_question(self, question, context):
        contextSentences, questionNlp, questionType = self.pp.process_question_context(question, context)
        
        questionDF = self.kg.extract_facts(question)
        if len(questionDF) == 1:
            questionDF["Subject"] = questionNlp.text
            questionDF["Relation"] = questionNlp.text
            questionDF["Objects"] = questionNlp.text
            questionDF["States"] = questionNlp.text
            questionDF["Times"] = questionNlp.text
            questionDF["Locations"] = questionNlp.text
            
        factsDF = self.kg.join_sentences_facts(contextSentences)
        newFactsDF = self.qa.change_subject_relation(factsDF, False)
        newQuestionDF = self.qa.change_subject_relation(questionDF, False)
    
        answer = self.qa.get_answer(newFactsDF, newQuestionDF, questionType)
        return answer
    