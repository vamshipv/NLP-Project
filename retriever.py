import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import re
from generator import Generator

"""
    Class Retriever has function to load the document, generate chunk and provide ouput for the input query
"""
class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        """
        Here the Split length is default set to 200
        with default document cats.txt, faiss.index and subtexts.json all loaded by default for the user query
        """
        self.split_len = 200
        self.document_file = "cats"
        self.index_file = f"{self.document_file}_faiss.index"
        self.subtext_file = f"{self.document_file}_subtexts.json"

    def split_text(self,text):
        """
        In this function text file is made in chunks

        Return text file chunks
        """
        chunks = [text[i:i+self.split_len] for i in range(0, len(text), self.split_len)]
        return chunks
    
    def defaultDocument(self):
        """
        In this function, it check for the default document avaliability

        returns true when the default document is avaliable

        returns false when default document is not found
        """
        if os.path.exists(self.document_file + ".txt") and os.path.exists(self.index_file) and os.path.exists(self.subtext_file):
            return True
        else:
            print("No default document found")
            return False
        
    def addDocuments(self,path):
        """
        This function takes the text path and it uses FIASS to give the vectors for the text
        """
        with open(path, 'r') as file:
            text = file.read()
        self.splitted_text=self.split_text(text)
        embeddings = self.model.encode(self.splitted_text)
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))
    
    def addExistingDocument(self, path):
        """
        This function takes path to check the to add document new document to the existing default document
        
        It also checks if the default document is avaliable 
        if avalaible it adds chunks to the document
        else it asks user to add a default document


        """
        base = os.path.basename(path)
        Combinedname, _ = os.path.splitext(base)
        combined_prefix = f"{self.document_file}_{Combinedname}"
        combined_index_path = f"{combined_prefix}.index"
        combined_chunk_path = f"{combined_prefix}.json"
        if self.defaultDocument():
            self.load(self.index_file, self.subtext_file)
            with open(path, 'r', encoding='utf-8') as file:
                new_text = file.read()
            new_chunks = self.split_text(new_text)
            new_embeddings = self.model.encode(new_chunks)

            self.index.add(np.array(new_embeddings))
            self.splitted_text.extend(new_chunks)

            self.save(combined_index_path, combined_chunk_path)
            print("Default document is avaliable, Appending it to the default document")
        else:
            print("Could not add document to existing one. Please add a default first")

    
    def query(self,query_text):
        """
        Here the default k (chunks) is set to 2
        It search the user query against the chunk to find the most similar chunks

        """
        nullInput = "Please enter something"
        if(query_text == ""):
            return nullInput
        k = 2
        # query_embedding = self.model.encode([query_text])
        # D, I = self.index.search(np.array(query_embedding), k)
        # self.retrieved_chunks = [self.splitted_text[i] for i in I[0]]
        # return self.retrieved_chunks
        query_embedding = self.model.encode([query_text])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        D, I = self.index.search(query_embedding, k)

        self.retrieved_chunks = [self.splitted_text[i] for i in I[0]]
        return self.retrieved_chunks
    
    def save(self, index_filename,splittext_filename):
        """
        This function saves the chunks to a file to make it easier to load the project 
        during the file load so it does not compute every time, that is why default data is used
        for the user search query
        """
        faiss.write_index(self.index, index_filename)
        with open(splittext_filename, 'w') as f:
            json.dump(self.splitted_text, f)
    
    def load(self, index_filename,splittext_filename):
        """
        This function loads the files needed for user search query
        """
        self.index = faiss.read_index(index_filename)
        with open(splittext_filename, 'r') as f:
            self.splitted_text = json.load(f)

def main():

    retriever = Retriever()

    """
    Below function is not used yet
    """
    def LoadNewDocument(newDocument):
        neighbour_size = 2
        split_len = 100
        base_name = os.path.splitext(os.path.basename(newDocument))[0]
        index_file = f"{base_name}_faiss.index"
        subtext_file = f"{base_name}_subtexts.json"
    """
    Below code has user choice to choice from and perform based on the choice
    Choice 1: loads the default document and procced to the user search query
    Choice 2: user has a option to overwrite the default document
    Choice 3: User had a option to add a new document to the default document
    """

    docChoices = {
        '1': "Local query with existing document",
        '2': "Overwrite with New document",
        '3': "Add to existing document",
        '4': "Exit"
    }

    """
    Function localQuery takes the user query to get the results
    Prints the results based on the user query
    """
    def localQuery(group_id):
        gen = Generator()
        while True:
            appendlist = []
            user_query = input("Your query: ").strip()
            
            try:
                results = retriever.query(user_query)
                if(user_query == ""):
                    print("Answer:", results)
                else:
                    # print("`````````````````````````````````````````````````````````````")
                    # print("Results")
                    # print("`````````````````````````````````````````````````````````````")
                    for res in results:
                        cleaned_res = res.replace("\n", " ").strip()
                        appendlist.append(cleaned_res)
                    # print(appendlist)
                    # print("`````````````````````````````````````````````````````````````")
                    # print("Context")
                    context = "\n\n".join(results)
                    # print(context)
                    # print("```````````````````end context```````````````````````````````")
                    # Generate answer from your generator
                    answer = gen.generate_answer(appendlist, context, user_query, group_id)
                    print("Answer:", answer)
            except Exception as e:
                print("Yo, somethings wrong with code. Try again:")
            

    """
    Below code promts user with a welcome message and choices about the document
    User get promots based on the choice
    
    """
    while True:
        print("Hello user, check for all the information on World of cats")
        print("Please choose options to Continue or type exit to exit")
        document_file = "winnie_the_pooh.txt"
        group_id = "Team Dave"
        base_name = os.path.splitext(os.path.basename(document_file))[0]
        index_file = f"{base_name}_faiss.index"
        subtext_file = f"{base_name}_subtexts.json"
        print(docChoices)
        ChoiceUser = input()
        if ChoiceUser == "1":
            if os.path.exists(index_file) and os.path.exists(subtext_file):
                print("Making sure query works. Hold on")
                retriever.load(index_file,subtext_file)
                localQuery(group_id)
            else:
                print("File are loading for the first time, Plese wait")
                retriever.addDocuments(document_file)
                retriever.save(index_file,subtext_file)
                localQuery(group_id)

        elif ChoiceUser == "2":
            print("Enter document name, inculding document extenstion")
            pattern = r'^[\w,\s-]+\.(txt|pdf)$'
            newDocument = input()
            if re.match(pattern, newDocument):
                full_path = os.path.join("data", newDocument)
                # base_name = os.path.splitext(os.path.basename(newDocument))[0]
                base_name = os.path.splitext(os.path.basename(full_path))[0]

                # Construct dynamic filenames
                index_file = f"{base_name}_faiss.index"
                subtext_file = f"{base_name}_subtexts.json"
                if not (os.path.exists(index_file) and os.path.exists(subtext_file)):
                    print("Please wait loading your new document")
                    retriever.addDocuments(full_path)
                    retriever.save(index_file,subtext_file)
                    print("Your document was loaded you can start with your query")
                else:
                    retriever.load(index_file,subtext_file)
                    print("This document already exists, Using the existing document")
                localQuery(group_id)
            else:
                print("Oops, you have to check again the document name")


        elif ChoiceUser == "3":
            print("Enter document name, inculding document extenstion")
            pattern = r'^[\w,\s-]+\.(txt|pdf)$'
            additionalDocument = input()
            if re.match(pattern, additionalDocument):
                retriever.addExistingDocument(additionalDocument)
                print("Choice 3")
                localQuery(group_id)
            else:
                print("Document name is wrong or Document already exists")
        elif ChoiceUser == '4':
            print("Good Day")
            break
        else:
            print("Please try again")

        

if __name__ == "__main__":
    main()