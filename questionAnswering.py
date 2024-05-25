import regex as re
from pathlib import Path
import spacy
import numpy as np
import coreferee
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import SmoothingFunction
import knowledgeGraph as KG

class QuestionAnswering:
    def __init__(self, model, excludesPerQuestionType):
        self.model = model
        self.excludesPerQuestionType = excludesPerQuestionType

    def change_subject_relation(self, factsDF, isQuestion = True):
        if not isQuestion:
            factsDF = factsDF[~((factsDF["Subject"] == "Unknown") & (factsDF["Objects"].apply(len) == 0) & (factsDF["States"].apply(len) == 0) & (factsDF["Times"].apply(len) == 0) & (factsDF["Locations"].apply(len) == 0))]
            factsDF = factsDF.reset_index(drop=True)
            
        for index, row in factsDF.iterrows():
            factsDF.loc[index, "Subject"] = [row['Subject']]
            factsDF.loc[index, "Relation"] = [row['Relation']]
        return factsDF

    def similarity(self, factRow, questionRow, column):
        """
        This function calculates the cosine similarity between the embeddings of a fact and a question.

        Parameters:
        factRow (pandas.Series): A row from the facts DataFrame.
        questionRow (pandas.Series): A row from the questions DataFrame.
        column (str): The name of the column to compare in the factRow and questionRow.

        Returns:
        float: The cosine similarity between the embeddings of the fact and the question.

        The function works as follows:
        1. If the specified column in either the factRow or questionRow is empty or contains only "Unknown", it returns 0.
        2. It joins the items in the specified column of the factRow and questionRow into strings.
        3. It uses a pre-trained model to encode these strings into embeddings.
        4. It calculates the cosine similarity between these embeddings using the util.cos_sim function.
        5. Finally, it returns the cosine similarity.
        """
        if len(factRow[column]) == 0 or len(questionRow[column]) == 0 or factRow[column] == ["Unknown"] or questionRow[column] == ["Unknown"]:
            return 0
        columnString = " ".join(factRow[column])
        questionString = " ".join(questionRow[column])
        embeddingFact = self.model.encode(columnString)
        embeddingQuestion = self.model.encode(questionString)
        return util.cos_sim(embeddingFact, embeddingQuestion)


    def cost_function(self, factsDf, questionFact, excludeColumns=[]):
        """
        This function calculates the cost of each fact in the facts DataFrame with respect to a question fact, and returns the index and cost of the fact with the highest cost.

        Parameters:
        factsDf (pandas.DataFrame): The DataFrame containing the facts.
        questionFact (pandas.DataFrame): The DataFrame containing the question fact.
        excludeColumns (list, optional): A list of column names to exclude from the cost calculation. Defaults to an empty list.

        Returns:
        tuple: A tuple containing the index of the fact with the highest cost and the cost itself.

        The function works as follows:
        1. It initializes the cost and maxFactIdx variables to 0.
        2. It creates a list of column names to consider in the cost calculation, excluding any columns specified in the excludeColumns parameter.
        3. It iterates over each fact in the facts DataFrame.
        4. For each fact, it calculates the cost by summing the cosine similarities between the fact and the question fact for each column.
        5. If the calculated cost is greater than the current maximum cost, it updates the maximum cost and the index of the fact with the maximum cost.
        6. Finally, it returns the index of the fact with the maximum cost and the cost itself.
        """
        cost = 0
        maxFactIdx = 0
        columnNames = ["Subject","Relation", "Objects", "States", "Times", "Locations"]
        for column in excludeColumns:
            columnNames.remove(column)
        for factIdx, factRow in factsDf.iterrows():
            currCost = 0
            for _, questionRow in questionFact.iterrows():
                if len(factRow[excludeColumns[0]]) == 0:
                    continue
                for column in columnNames:
                    currCost += self.similarity(factRow, questionRow, column)
            if currCost > cost:
                cost = currCost
                maxFactIdx = factIdx
        return maxFactIdx, cost
        
    def get_answer(self, factsDF, questionDF, questionType):
        """
        This function determines the answer to a question based on the facts extracted from a document.

        Parameters:
        factsDF (pandas.DataFrame): The DataFrame containing the facts extracted from the document.
        questionDF (pandas.DataFrame): The DataFrame containing the facts extracted from the question.
        questionType (str): The type of the question (e.g., "who", "what", "when", etc.).

        Returns:
        str: The answer to the question.

        The function works as follows:
        1. It calculates the cost of each fact in the facts DataFrame with respect to the question fact, excluding the column corresponding to the question type.
        2. It determines the fact with the highest cost.
        3. It retrieves the answer from the column of the fact that corresponds to the question type.
        4. If the answer is empty, it retrieves the answer from the "States" column of the fact.
        5. It joins the items in the answer into a string.
        6. Finally, it returns the answer.
        """
        correctIdx, _ = self.cost_function(factsDF, questionDF, excludeColumns=[self.excludesPerQuestionType[questionType]])
        answer = factsDF.loc[correctIdx, self.excludesPerQuestionType[questionType]]
        if answer == []:
            answer = factsDF.loc[correctIdx, "States"]    
        return " ".join(answer)
