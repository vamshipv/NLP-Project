import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join("..", "generator")))
from generator import Generator


class Test_Generator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = Generator()
        cls.user_query='Summary about Samsung galaxy m51'


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
            {'text': 'The phone has a long-lasting battery life.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '4'}, 
            {'text': 'Camera is underwhelming in low light.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '3'}, 
            {'text': 'Disappointed with the build quality.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '2'}, 
            {'text': 'Amazing screen length and audio.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '5'}
        ]

        summary_pass = self.generator.generate_summary(self.user_query, retrieved_chunks)
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
            {'text': 'Camera is underwhelming in low light.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '3'}, 
            {'text': 'Disappointed with the build quality.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '2'}, 
        ]
        summary_fail = self.generator.generate_summary(self.user_query, retrieved_chunks)
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
            {'text': 'Camera is underwhelming in low light.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '3'}, 
            {'text': 'Disappointed with the build quality.', 'brand': 'Samsung', 'model': 'Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)', 'stars': '2'}, 
        ]
        summary_not_empty = self.generator.generate_summary(self.user_query, retrieved_chunks)
        self.assertTrue(bool(summary_not_empty.strip()), "Summary should not be empty for valid input chunks")

    def test_04_empty_summary(self):
        """
        Test method verifes that the no summary is generated for empty input chunks.

        When provided with empty review chunks, it ensures that the model generates empty summary.

        Raises:
            AssertionError: If the generated summary is empty or contains only whitespace.
        """
        retrieved_chunks = []
        summary_empty = self.generator.generate_summary(self.user_query, retrieved_chunks)
        self.assertEqual(summary_empty.strip(), "No relevant reviews found.", "Summary should be empty when no input chunks are provided")

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

    def assertEqual(self, actual, expected, message):
        """
        Custom assert function that prints test result for equality.

        Args:
            actual: The actual result.
            expected: The expected result.
            message (str): The message to print if the assertion fails.
        """
        print('\n------------------------------------------------------\n')
        if actual != expected:
            print("Test case failed, Error message:", message)
            print(f"Expected: {expected}, but got: {actual}")
        else:
            print("Test Case passed")

if __name__ == "__main__":
        unittest.main()
     

    
