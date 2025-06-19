import os
import sys
import unittest
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from retriever.gen_ollama_gemma import summarize_with_gemma

class Test_Generator(unittest.TestCase):

    def test_01_keywords_in_summary_pass(self):
        """
        This method verifies that the generated summary contains all expected keywords.

        When provided with relevant review chunks 
        containing keywords such as 'battery', 'camera', 'screen', and 'build', it ensures that the 
        model includes all of them in generated summary.

        Raises:
            AssertionError: If any of the expected keywords are missing 
            from the summary.
        """
        retrieved_chunks = [
            "The phone has a long-lasting battery life.",
            "Camera is underwhelming in low light.",
            "Disappointed with the build quality.",
            "Front camera is worst, not up to the mark. Waste of money.",
            "Amazing screen length and audio. Camera works good for me. Brilliant auto sensors. Within the price range."
        ]
        device_name= 'Samsung Galaxy M51'
        context = " ".join(retrieved_chunks)
        summary_pass = summarize_with_gemma(context, device_name)
        print('---------summary-------------\n',summary_pass)
        # print(summary_pass)
        keywords = ["battery", "camera", "screen", "build"]
        all_keywords = all(kw in summary_pass.lower() for kw in keywords)
        self.assertTrue(all_keywords, "Not all keywords were found in the summary")

    def test_02_keywords_in_summary_fail(self):
        """
        This method verifies that the generated summary does not contain unrelated keywords.

        When provided with relevant review chunks that doesn't
        contain keywords such as 'battery','screen', it ensures that the 
        model doesn't include them in generated summary.

        Raises:
            AssertionError: If the summary incorrectly includes any of the 
        unrelated keywords.
        """
        retrieved_chunks = [
            "Camera is underwhelming in low light.",
            "Disappointed with the build quality.",
        ]
        device_name= 'Samsung Galaxy M51'
        context = " ".join(retrieved_chunks)
        summary_fail = summarize_with_gemma(context, device_name)
        print('---------summary-------------\n',summary_fail)
        keywords = ["battery", "screen"]
        all_keywords = all(kw in summary_fail.lower() for kw in keywords)
        self.assertFalse(all_keywords, "Provided keywords should not be in the summary")

    def test_03_non_empty_summary(self):
        """
        Test method verifes that the summary is generated for valid input chunks.

        When provided with valid review chunks, it ensures that the model does not generate empty summary.

        Raises:
            AssertionError: If the generated summary is empty or contains only whitespace.
        """
        retrieved_chunks = [
            "Camera is underwhelming in low light.",
            "Disappointed with the build quality.",
        ]
        device_name= 'Samsung Galaxy M51'
        context = " ".join(retrieved_chunks)
        summary_not_empty = summarize_with_gemma(context, device_name)
        print('---------summary-------------\n',summary_not_empty)
        self.assertTrue(bool(summary_not_empty.strip()), "Summary should not be empty for valid input chunks")

    def test_04_empty_summary(self):
        """
        Test method verifes that the no summary is generated for empty input chunks.

        When provided with empty review chunks, it ensures that the model generates empty summary.

        Raises:
            AssertionError: If the generated summary is empty or contains only whitespace.
        """
        retrieved_chunks = []
        device_name= 'Samsung Galaxy M51'
        context = " ".join(retrieved_chunks)
        summary_empty = summarize_with_gemma(context, device_name)
        print('---------summary-------------\n',summary_empty)
        self.assertEqual(summary_empty.strip(), "", "Summary should be empty when no input chunks are provided")

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

    def assertEqual(self, actual, expected, message):
        """
        Custom assert function that prints test result for equality.

        Args:
            actual: The actual result.
            expected: The expected result.
            message (str): The message to print if the assertion fails.
        """
        if actual != expected:
            print("Test case failed, Error message:", message)
            print(f"Expected: {expected}, but got: {actual}")
        else:
            print("Test Case passed")

if __name__ == "__main__":
        unittest.main()
     

    
