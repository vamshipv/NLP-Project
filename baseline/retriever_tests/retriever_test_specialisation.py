import os
import sys
import unittest
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from retriever.retriever_ollama import Retriever


class Test_Retriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever()
        cls.retriever.read_and_chunk_csv("/Users/dechammacg/Documents/NLPPro/specialisation/NLP-Project/baseline/data/final_dataset.csv")
        cls.retriever.build_index()

    def retriever_search(self, user_query, top_k=5):
        results = self.retriever.search(user_query, top_k)
        print('\n Top Matching Results:')
        for res in results:
            print("-", res)
        return results

    def test_01_results_not_empty(self):
        """
        Test that the retriever returns non-empty results for a relevant query.

        This test ensures that the retriever returns at least one relevant chunk 
        when given a query about a specific product.

        Raises:
            AssertionError: If no results are retrieved.
        """
        user_query = "Summarise the reviews about Vivo y91"
        retrieved_results = self.retriever_search(user_query, 3)
        self.assertTrue(len(retrieved_results) > 0, "Retrieved chunks list is empty")  

    def test_02_keywords_in_retrieved_text(self):
        """
        Test that retrieved chunks contain the expected keyword from the query.

        This test verifies that the retriever is returning reviews related 
        to a specific feature (e.g., "battery") when mentioned in the user query.

        Raises:
            AssertionError: If none of the retrieved chunks contain the expected keyword.
        """
        user_query = "Summarise the reviews about the battery in Vivo y91"
        retrieved_results = self.retriever_search(user_query,4)
        found = any("battery" in chunk.lower() for chunk in retrieved_results)
        self.assertTrue(found, "No reviews contains the keyword 'battery'")

    def test_03_product_name_match(self):
        """
        Test that all retrieved chunks are related to the expected product.

        This test confirms that the retriever is returning reviews specifically 
        about the queried product (e.g., "Vivo y91").

        Raises:
            AssertionError: If any chunk does not mention the expected product name.
        """
        user_query = "Summarise the reviews about Vivo y91"
        retrieved_results = self.retriever_search(user_query, 3)
        expected_product = "Vivo y91"
        found = all(expected_product.lower() in chunk.lower() for chunk in retrieved_results)
        self.assertTrue(found, "Some reviews are not about the expected product")

    def test_04_product_name_no_match(self):
        """
        Test that irrelevant product reviews are not included in retrieval.

        This test checks that the retriever does not return reviews for a different 
        product (e.g., "Samsung Galaxy M31") when asked about another (e.g., "Vivo y91").

        Raises:
            AssertionError: If any unrelated product review is found in the results.
        """
        user_query = "Summarise the reviews about Vivo y91"
        retrieved_results = self.retriever_search(user_query, 3)
        expected_product = "Samsung Galaxy M31"
        found = any(expected_product.lower() in chunk.lower() for chunk in retrieved_results)
        self.assertFalse(found, "Some reviews are about expected product..")

    def assertTrue(self,expression, message):
        """
        Custom assert function that prints test result for a true condition.

        Args:
            expression (bool): The boolean expression to evaluate.
            message (str): The message to print if the assertion fails.
        """
        if not expression:
            print("Test case failed, Error message: ", message)
        else:
            print("Test Case passed")

    def assertFalse(self, expression, message):
        """
        Custom assert function that prints test result for a false condition.

        Args:
            expression (bool): The boolean expression to evaluate.
            message (str): The message to print if the assertion fails.
        """
        if expression:
            print("Test case failed, Error message: ", message)
        else:
            print("Test Case passed")

if __name__ == "__main__":
        unittest.main()
     

    
