import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import re
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from baseline.generator.generator import Generator

class Retriever:
    """
    A class to handle document retrieval using FAISS and SentenceTransformers.

    Attributes:
        model_name (str): Name of the embedding model.
        split_len (int): Length of text chunks for indexing.
        document_file (str): Name of the default document.
        index_file (str): Filename for saving/loading FAISS index.
        subtext_file (str): Filename for saving/loading text chunks.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        """
        Here the Split length is default set to 200
        with default document cats.txt, faiss.index and subtexts.json all loaded by default for the user query
        """
        self.split_len = 200
        self.document_file = "../data/winnie_the_pooh"
        self.index_file = f"{self.document_file}_faiss.index"
        self.subtext_file = f"{self.document_file}_subtexts.json"

    def split_text(self,text):
        """
        Splits a given text into fixed-length chunks.

        Args:
            text (str): The input text.

        Returns:
            list: A list of text chunks.
        """
        chunks = [text[i:i+self.split_len] for i in range(0, len(text), self.split_len)]
        return chunks
    
    def defaultDocument(self):
        """
        Checks whether default document, index, and chunk file exist.
        Returns:
        bool: True if all default files exist, False otherwise.
        """
        if os.path.exists(self.document_file + ".txt") and os.path.exists(self.index_file) and os.path.exists(self.subtext_file):
            return True
        else:
            print("No default document found")
            return False
        
    def addDocuments(self,path):
        """
        Adds a new document, splits it into chunks, generates embeddings, and creates a FAISS index.

        Args:
            path (str): File path to the text document.
        """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        full_path = os.path.join(base_dir, path if path.endswith('.txt') else path + '.txt')

    
        # Join it with the filename passed in
        # full_path = os.path.join(data_dir, path)
        with open(full_path, 'r') as file:
            text = file.read()
        self.splitted_text=self.split_text(text)
        embeddings = self.model.encode(self.splitted_text)
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))
    
    def addExistingDocument(self, path):
        """
        Appends a new document to an existing default document.
        Loads the existing index and chunks, combines them with new ones, and re-saves.

        Args:
            path (str): Path to the additional text document.
        """

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        full_path = os.path.join(base_dir, path if path.endswith('.txt') else path + '.txt')
        base = os.path.basename(full_path)
        Combinedname, _ = os.path.splitext(base)
        combined_prefix = f"{self.document_file}_{Combinedname}"
        combined_index_path = f"{combined_prefix}.index"
        combined_chunk_path = f"{combined_prefix}.json"
        if self.defaultDocument():
            self.load(self.index_file, self.subtext_file)
            with open(full_path, 'r', encoding='utf-8') as file:
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
        Retrieves top-k most similar chunks for a given query.

        Args:
            query_text (str): The user query string.

        Returns:
            list: A list of top-k similar text chunks.
        """
        nullInput = "Please enter something"
        if(query_text == "" or query_text is None):
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
            Saves the FAISS index and chunked text to disk.

        Args:
            index_filename (str): Filename for FAISS index.
            splittext_filename (str): Filename for text chunks.
        """
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(data_path, index_filename))
        with open(os.path.join(data_path, splittext_filename), 'w') as f:
            json.dump(self.splitted_text, f)

    
    def load(self, index_filename,splittext_filename):
        """
        Loads the FAISS index and chunked text from disk.

        Args:
            index_filename (str): Filename for FAISS index.
            splittext_filename (str): Filename for text chunks.
        """
        p = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.index = faiss.read_index(os.path.join(p, index_filename))
        with open(os.path.join(p, splittext_filename), 'r') as f:
            self.splitted_text = json.load(f)

# def main():
#     """
#     Main function that provides a CLI to interact with the RAG system.
    
#     Functionality:
#         - Choose from loading an existing document, overwriting it, or appending new content.
#         - Load and save FAISS index and chunked text data.
#         - Accepts user queries and retrieves relevant document content.
#         - Generates responses using a generator module.
#     """
#     retriever = Retriever()

#     def LoadNewDocument(newDocument):
#         neighbour_size = 2
#         split_len = 100
#         base_name = os.path.splitext(os.path.basename(newDocument))[0]
#         index_file = f"{base_name}_faiss.index"
#         subtext_file = f"{base_name}_subtexts.json"
#     """
#     Below code has user choice to choice from and perform based on the choice
#     Choice 1: loads the default document and procced to the user search query
#     Choice 2: user has a option to overwrite the default document
#     Choice 3: User had a option to add a new document to the default document
#     """

#     docChoices = {
#         '1': "Local query with existing document",
#         '2': "Overwrite with New document",
#         '3': "Add to existing document",
#         '4': "Exit"
#     }

#     def localQuery(group_id):
#         """
#             Prompts the user to enter queries and prints the retrieved answers using the generator.

#             Args:
#             group_id (str): Identifier for the team or group (used in generator).
#         """
#         gen = Generator()
#         while True:
#             appendlist = []
#             user_query = input("Your query: ").strip()
#             if user_query == ("exit1"):
#                 print("Have a nice one")
#                 sys.exit()
#             try:
#                 results = retriever.query(user_query)
#                 if(user_query == ""):
#                     print("Answer:", results)
#                 else:
#                     # print("`````````````````````````````````````````````````````````````")
#                     # print("Results")
#                     # print("`````````````````````````````````````````````````````````````")
#                     for res in results:
#                         cleaned_res = res.replace("\n", " ").strip()
#                         appendlist.append(cleaned_res)
#                     # print(appendlist)
#                     # print("`````````````````````````````````````````````````````````````")
#                     # print("Context")
#                     context = "\n\n".join(results)
#                     # print(context)
#                     # print("```````````````````end context```````````````````````````````")
#                     # Generate answer from your generator
#                     answer = gen.generate_answer(appendlist, context, user_query, group_id)
#                     # print("Here")
#                     print("Answer:", answer)
#             except Exception as e:
#                 print("Yo, somethings wrong with code. Try again:")
            

#     """
#     Below code promts user with a welcome message and choices about the document
#     User get promots based on the choice
    
#     """
#     while True:
#         print("Hello user, check for all the information on World of cats")
#         print("Please choose options to Continue or type exit to exit")
#         document_file = "../data/winnie_the_pooh.txt"
#         group_id = "Team Dave"
#         base_name = os.path.splitext(os.path.basename(document_file))[0]

#         data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
#         index_file = os.path.join(data_dir, f"{base_name}_faiss.index")
#         subtext_file = os.path.join(data_dir, f"{base_name}_subtexts.json")
#         print(docChoices)
#         ChoiceUser = input()
#         if ChoiceUser == "1":
#             if os.path.exists(index_file) and os.path.exists(subtext_file):
#                 print("Making sure query works. Hold on")
#                 retriever.load(index_file,subtext_file)
#                 localQuery(group_id)
#             else:
#                 print("File are loading for the first time, Plese wait")
#                 retriever.addDocuments(document_file)
#                 retriever.save(index_file,subtext_file)
#                 localQuery(group_id)

#         elif ChoiceUser == "2":
#             print("Enter document name, inculding document extenstion")
#             pattern = r'^[\w,\s-]+\.(txt|pdf)$'
#             newDocument = input()
#             if re.match(pattern, newDocument):
#                 # full_path = os.path.join("data", newDocument)
#                 base_name = os.path.splitext(os.path.basename(newDocument))[0]
#                 # base_name = os.path.splitext(os.path.basename(full_path))[0]

#                 # Construct dynamic filenames
#                 index_file = f"{base_name}_faiss.index"
#                 subtext_file = f"{base_name}_subtexts.json"
#                 if not (os.path.exists(index_file) and os.path.exists(subtext_file)):
#                     print("Please wait loading your new document")
#                     retriever.addDocuments(base_name)
#                     retriever.save(index_file,subtext_file)
#                     print("Your document was loaded you can start with your query")
#                 else:
#                     retriever.load(index_file,subtext_file)
#                     print("This document already exists, Using the existing document")
#                 localQuery(group_id)
#             else:
#                 print("Oops, you have to check again the document name")


#         elif ChoiceUser == "3":
#             print("Enter document name, inculding document extenstion")
#             pattern = r'^[\w,\s-]+\.(txt|pdf)$'
#             additionalDocument = input()
#             if re.match(pattern, additionalDocument):
#                 print("Please wait loading your document")
#                 retriever.addExistingDocument(additionalDocument)
#                 print("Choice 3")
#                 localQuery(group_id)
#             else:
#                 print("Document name is wrong or Document already exists")
#         elif ChoiceUser == '4':
#             print("Good Day")
#             break
#         else:
#             print("Please try again")

        

# if __name__ == "__main__":
#     main()