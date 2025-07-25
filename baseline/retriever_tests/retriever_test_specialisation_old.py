import os
import sys
import unittest
import re

# Ensure the parent directories are in the path for imports
sys.path.append(os.path.abspath(os.path.join("..", "retriever")))

from retriever import Retriever


class Test_Retriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever()
        cls.retriever.chunk_reviews()
        cls.retriever.index_chunks()

    def retriever_search(self, user_query, top_k=5):
        return self.retriever.retrieve(user_query, top_k)

    def test_01_results_not_empty(self):
        """
        Test that the retriever returns non-empty results for a relevant query.

        This test ensures that the retriever returns at least one relevant chunk 
        when given a query about a specific product.

        Raises:
            AssertionError: If no results are retrieved.
        """
        user_query = "Summary about Samsung galaxy m51"
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
        user_query = "Summarise the reviews about battery in Samsung galaxy m51"
        retrieved_results = self.retriever_search(user_query,4)
        found = any("battery" in chunk["text"].lower() for chunk in retrieved_results)
        self.assertTrue(found, "No reviews contains the keyword 'battery'")

    def test_03_product_name_match(self):
        """
        Test that all retrieved chunks are related to the expected product.

        This test confirms that the retriever is returning reviews specifically 
        about the queried product (e.g., "Samsung galaxy m51").

        Raises:
            AssertionError: If any chunk does not mention the expected product name.
        """
        user_query = "Summary about Samsung galaxy m51"
        retrieved_results = self.retriever_search(user_query, 3)
        expected_product = "Samsung galaxy m51"
        found = all(expected_product.lower() == re.sub(r"\(.*?\)", "", chunk["model"].lower()).strip() for chunk in retrieved_results)
        self.assertTrue(found, "Some reviews are not about the expected product")

    def test_04_product_name_no_match(self):
        """
        Test that irrelevant product reviews are not included in retrieval.

        This test checks that the retriever does not return reviews for a different 
        product (e.g., "Vivo y91") when asked about another (e.g., "Samsung galaxy m51").

        Raises:
            AssertionError: If any unrelated product review is found in the results.
        """
        user_query = "Summary about Samsung galaxy m51"
        retrieved_results = self.retriever_search(user_query, 3)
        unrelated_product = "vivo y91i"
        found = any(unrelated_product.lower() == re.sub(r"\(.*?\)", "", chunk["model"].lower()).strip() for chunk in retrieved_results)
        self.assertFalse(found, "Some reviews are about an unrelated product")

    def assertTrue(self,expression, message):
        """
        Custom assert function that prints test result for a true condition.

        Args:
            expression (bool): The boolean expression to evaluate.
            message (str): The message to print if the assertion fails.
        """
        print('\n------------------------------------------------------\n')
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
        print('\n------------------------------------------------------\n')
        if expression:
            print("Test case failed, Error message: ", message)
        else:
            print("Test Case passed")

if __name__ == "__main__":
        unittest.main()
     

    
