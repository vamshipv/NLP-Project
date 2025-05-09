import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import re


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        #Default Split length
        self.split_len = 100
        self.document_file = "cats"
        self.index_file = f"{self.document_file}_faiss.index"
        self.subtext_file = f"{self.document_file}_subtexts.json"

    def split_text(self,text):
        chunks = [text[i:i+self.split_len] for i in range(0, len(text), self.split_len)]
        return chunks
    
    def defaultDocument(self):
        if os.path.exists(self.document_file + ".txt") and os.path.exists(self.index_file) and os.path.exists(self.subtext_file):
            print("Default document is avaliable")
            return True
        else:
            print("No default document found")
            return False
        
    def addDocuments(self,path):
        with open(path, 'r') as file:
            text = file.read()
        self.splitted_text=self.split_text(text)
        embeddings = self.model.encode(self.splitted_text)
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))
    
    def addExistingDocument(self, path):
        base = os.path.basename(path)        # "textSamples.txt"
        Combinedname, _ = os.path.splitext(base)
        combined_prefix = f"{self.document_file}_{Combinedname}"
        combined_index_path = f"{combined_prefix}.index"
        combined_chunk_path = f"{combined_prefix}.json"
        if self.defaultDocument():
            self.load(self.index_file, self.subtext_file)
            # Step 2: Read and process new document
            with open(path, 'r', encoding='utf-8') as file:
                new_text = file.read()
            new_chunks = self.split_text(new_text)
            new_embeddings = self.model.encode(new_chunks)

            # Append new data
            self.index.add(np.array(new_embeddings))
            self.splitted_text.extend(new_chunks)

            # Save combined index and chunks under new name
            self.save(combined_index_path, combined_chunk_path)
            print("New document added to exisiting document")
        else:
            print("Could not add document to existing one")

    
    def query(self,query_text):
        #Dafault K is set as 2
        k = 2
        query_embedding = self.model.encode([query_text])
        D, I = self.index.search(np.array(query_embedding), k)
        self.retrieved_chunks = [self.splitted_text[i] for i in I[0]]
        return self.retrieved_chunks
    
    def save(self, index_filename,splittext_filename):
        faiss.write_index(self.index, index_filename)
        with open(splittext_filename, 'w') as f:
            json.dump(self.splitted_text, f)
    
    def load(self, index_filename,splittext_filename):
        self.index = faiss.read_index(index_filename)
        with open(splittext_filename, 'r') as f:
            self.splitted_text = json.load(f)

def main():

    retriever = Retriever()

    def LoadNewDocument(newDocument):
        neighbour_size = 2
        split_len = 100
        base_name = os.path.splitext(os.path.basename(newDocument))[0]
        # Construct dynamic filenames
        index_file = f"{base_name}_faiss.index"
        subtext_file = f"{base_name}_subtexts.json"
    
    book_choices = {
        '1': "Local query with existing document",
        '2': "Overwrite with New document",
        '3': "Add to existing document"
    }

    def localQuery():
        while True:
            user_query = input("Your query: ").strip()
            if user_query.lower() in ['exit', 'quit']:
                print("Bye")
                break
            try:
                results = retriever.query(user_query)
                print("`````````````````````````````````````````````````````````````")
                print("Results")
                print("`````````````````````````````````````````````````````````````")
                for res in results:
                    res.replace("\n", " ").strip()
                    print(res)
                print("`````````````````````````````````````````````````````````````")
            except Exception as e:
                print("Yo, somethings wrong with code. Try again:")

    while True:
        print("Hello user, check for all the information on World of cats")
        print("Please choose options to Continue")
        document_file = "winnie_the_pooh.txt"
        base_name = os.path.splitext(os.path.basename(document_file))[0]
        index_file = f"{base_name}_faiss.index"
        subtext_file = f"{base_name}_subtexts.json"
        print(book_choices)
        ChoiceUser = input()
        
        # program already loads with a default dataset
        if ChoiceUser == "1":
            #Default Data Set
            
            if os.path.exists(index_file) and os.path.exists(subtext_file):
                print("Making sure query works. Hold on")
                retriever.load(index_file,subtext_file)
                localQuery()
            else:
                print("First time ? Plese wait")
                retriever.addDocuments(document_file)
                retriever.save(index_file,subtext_file)
                localQuery()


        # Ask user to load his document local search query
        # If the user chooses to load his own document run the load document and chunks
        elif ChoiceUser == "2":
            print("Enter document name, inculding document extenstion")
            pattern = r'^[\w,\s-]+\.(txt|pdf)$'
            newDocument = input()
            if re.match(pattern, newDocument):
                
                base_name = os.path.splitext(os.path.basename(newDocument))[0]
                # Construct dynamic filenames
                index_file = f"{base_name}_faiss.index"
                subtext_file = f"{base_name}_subtexts.json"
                if not os.path.exists(index_file) and os.path.exists(subtext_file):
                    print("Please wait loading your new document")
                    retriever.addDocuments(newDocument)
                    retriever.save(index_file,subtext_file)
                    print("Your document was loaded you can start with your query")
                else:
                    retriever.load(index_file,subtext_file)
                    print("This document already exists, Using the existing document")
                localQuery()
            else:
                print("Oops, you have to check again the document name")


        # if user wants to add more document then combine the documents with default and move to search query
        elif ChoiceUser == "3":
            print("Enter document name, inculding document extenstion")
            pattern = r'^[\w,\s-]+\.(txt|pdf)$'
            additionalDocument = input()
            if re.match(pattern, additionalDocument):
                retriever.addExistingDocument(additionalDocument)
                print("Choice 3")
                localQuery()
            else:
                print("Document name is wrong or Document already exists")

        else:
            print("Please try again")


if __name__ == "__main__":
    main()