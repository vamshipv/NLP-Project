import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "..", "sentiment_analyzer")))

from generator import Generator
from sentiment_analyzer import SentimentAnalyzer

"""
This class contains the test suite for validating sentiment analysis.

"""
class Test_Generator_Sentiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = Generator()
        cls.sentiment_analyzer = SentimentAnalyzer()

    def test_01_sentiment_positive_summary(self):
        """
        This test verifies that when provided with consistently positive product reviews, the system correctly classifies the overall sentiment as "Positive"

        Raises: 
            AssertionError: If sentiment does not reflect overall positive sentiment

        """
        user_query = 'summarize to me the user reviews about Samsung Galaxy M51'
        retrieved_chunks = [
            {
                "text": "Awesome camera Awesome battery Awesome display .... Super clear Zoom. Super Macro sensor. Super ultrawide angle. Thanks Samsung for 730G",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "5"
            },
            {
                "text": "Camera quality is really good with many features. Even in dim light it performs well. Battery lasted 3 hours during heavy data transfer!",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "5"
            },
            {
                "text": "Perfect phone if you don't expect iPhone-like features. Battery is great and display is amazing.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "5"
            }
        ]
        summary = self.generator.generate_summary(user_query, retrieved_chunks)
        self.assertTrue("OVERALL SENTIMENT : Positive" in summary[0], "Sentiment does not reflect overall positive sentiment")

    def test_02_sentiment_negative_summary(self):
        """
        This test ensures that when provided with consistently negative product reviews, the system correctly classifies the overall sentiment as "Negative"

        Raises: 
            AssertionError: If sentiment does not reflect overall negative sentiment

        """
        user_query = 'summarize to me the user reviews about Samsung Galaxy M51'
        retrieved_chunks = [
            {
                "text": "Extremely disappointed. Phone hangs too much. Battery drains overnight. Worst phone I ever bought.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "1"
            },
            {
                "text": "Camera is horrible in low light. UI is very buggy. Apps crash often. I regret buying this.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "2"
            },
            {
                "text": "Touch response is very slow. Build quality feels cheap. Not worth the price.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "1"
            },
            {
                "text": "Heats up quickly even with minimal usage. Totally unreliable for gaming or multitasking.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "1"
            }
        ]
        summary = self.generator.generate_summary(user_query, retrieved_chunks)
        self.assertTrue("OVERALL SENTIMENT : Negative" in summary[0], "Sentiment does not reflect overall negative sentiment")
    
    def test_04_sentiment_positive_battery(self):
        """
        This test ensures that when provided with consistently positive product reviews for an aspect, the system correctly classifies the overall sentiment as "Positive"

        Raises: 
            AssertionError: If sentiment does not reflect overall positive sentiment

        """
        user_query = 'summarize to me the user reviews of Samsung Galaxy M51 focusing on its battery'
        retrieved_chunks = [
            {
                "text": "Battery lasts really long and charges quickly. I am very satisfied with the battery performance.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "5"
            },
            {
                "text": "The fast charging feature on the battery is excellent and very convenient for daily use.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "5"
            },
            {
                "text": "Battery life is impressive and easily supports a full day of heavy usage.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "5"
            }
        ]
        summary = self.generator.generate_summary(user_query, retrieved_chunks)
        self.assertTrue("OVERALL SENTIMENT : Positive" in summary[0], "Sentiment does not reflect overall positive sentiment for the aspect")


    def test_05_sentiment_negative_battery(self):
        """
        This test ensures that when provided with consistently negative product reviews for an aspect, the system correctly classifies the overall sentiment as "Negative"

        Raises: 
            AssertionError: If sentiment does not reflect overall negative sentiment

        """
        user_query = 'summarize to me the user reviews of Samsung Galaxy M51 focusing on its battery'
        retrieved_chunks = [
            {
                "text": "Battery drains too quickly and does not last a full day. Very disappointed.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "1"
            },
            {
                "text": "Charging takes forever and the battery performance is poor compared to other phones.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "2"
            },
            {
                "text": "Battery overheats and drains even when the phone is idle, which is frustrating.",
                "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "1"
            }
        ]
        summary = self.generator.generate_summary(user_query, retrieved_chunks)
        self.assertTrue("OVERALL SENTIMENT : Negative" in summary[0], "Sentiment does not reflect overall negative sentiment for the aspect")

    def test_pass_with_empty_review_list(self):
        """
        This test ensures that when provided with empty review list, the system returns "No relevant reviews found." message

        Raises: 
            AssertionError: If sentiment is returned and "No relevant reviews found." is not shown.

        """
        user_query = 'summarize user reviews about Samsung Galaxy M51'
        retrieved_chunks = []
        summary = self.generator.generate_summary(user_query, retrieved_chunks)
        self.assertEqual(summary.strip(), "No relevant reviews found.", "Sentiment block should be empty when no input chunks are provided") 


    def assertTrue(self, expression, message):
        """
        Custom assertTrue assertion with formatted output.

        Args:
            expression (bool): The boolean expression to evaluate.
            message (str): The error message to display if assertion fails.

        Prints:
            Formatted test result with pass/fail status and error details.
        """
        print('\n------------------------------------------------------\n')
        if not expression:
            print("Test case failed, Error message:", message)
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')

    def assertFalse(self, expression, message):
        """
        Custom assertFalse assertion with formatted output.

        Args:
            expression (bool): The boolean expression to evaluate (should be False).
            message (str): The error message to display if assertion fails.

        Prints:
            Formatted test result with pass/fail status and error details.
        """
        print('\n------------------------------------------------------\n')
        if expression:
            print("Test case failed, Error message:", message)
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')

    def assertEqual(self, actual, expected, message):
        """
        Custom assertEqual assertion with formatted output and detailed comparison.

        Args:
            actual: The actual value obtained from the test.
            expected: The expected value for comparison.
            message (str): The error message to display if assertion fails.

        Prints:
            Formatted test result with pass/fail status and value comparison details.
        """
        print('\n------------------------------------------------------\n')
        if actual != expected:
            print("Test case failed, Error message:", message)
            print(f"Expected: {expected}, but got: {actual}")
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')

    def assertIn(self, member, container, message):
        """
        Custom assertIn assertion to check substring/member presence with formatted output.

        Args:
            member: The item/substring to search for.
            container: The container/string to search within.
            message (str): The error message to display if assertion fails.

        Prints:
            Formatted test result with pass/fail status and search details.
        """
        print('\n------------------------------------------------------\n')
        if member not in container:
            print("Test case failed, Error message:", message)
            print(f"Expected '{member}' to be found in: {container}")
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')


if __name__ == "__main__":
    unittest.main()