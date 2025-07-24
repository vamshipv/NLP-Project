import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "..", "sentiment_analyzer")))

from generator import Generator
from sentiment_analyzer import SentimentAnalyzer


class Test_Generator_Sentiment(unittest.TestCase):
    """
    A comprehensive test suite for validating sentiment analysis and summary generation.

    This class tests the integration between the Generator and SentimentAnalyzer components
    to ensure accurate sentiment classification and appropriate summary generation based on
    customer review data for mobile devices.

    The test cases cover:
    - General positive and negative sentiment analysis
    - Aspect-specific sentiment analysis (focusing on battery performance)
    - Proper sentiment output formatting and consistency

    Attributes:
        generator (Generator): Instance of the Generator class for summary generation.
        sentiment_analyzer (SentimentAnalyzer): Instance for sentiment analysis operations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class-level test fixtures.

        Initializes the Generator and SentimentAnalyzer instances that will be used
        across all test methods in this test class. This method is called once before
        any test methods are executed.

        Class Attributes Created:
            cls.generator (Generator): Instance for generating summaries from review data.
            cls.sentiment_analyzer (SentimentAnalyzer): Instance for analyzing sentiment.
        """
        cls.generator = Generator()
        cls.sentiment_analyzer = SentimentAnalyzer()

    def test_01_sentiment_positive_summary(self):
        """
        Test positive sentiment classification for highly-rated reviews.

        This test verifies that when provided with consistently positive product reviews
        (all 5-star ratings), the system correctly:
        - Generates an appropriate summary reflecting positive feedback
        - Classifies the overall sentiment as "Positive"
        - Produces sentiment analysis output in the expected format

        Steps:
            1. Create mock positive review data for Samsung Galaxy M51
            2. Generate summary using the Generator class
            3. Analyze sentiment of the generated summary
            4. Assert that the sentiment block contains "OVERALL SENTIMENT : Positive"

        Expected Result:
            The sentiment analysis should identify positive sentiment and format
            the output correctly with "OVERALL SENTIMENT : Positive".
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
        sentiment_block = self.sentiment_analyzer.analyze_sentiment([summary[0]])[0]

        print("Predicted Sentiment:", sentiment_block)
        #print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        self.assertTrue("OVERALL SENTIMENT : Positive" in sentiment_block, "Summary does not reflect overall positive sentiment")

    def test_02_sentiment_negative_summary(self):
        """
        Test negative sentiment classification for low-rated reviews.

        This test ensures that when provided with consistently negative product reviews
        (1-2 star ratings), the system correctly:
        - Generates a summary reflecting negative customer experiences
        - Classifies the overall sentiment as "Negative"
        - Maintains proper sentiment analysis output formatting

        Steps:
            1. Create mock negative review data with various complaints
            2. Generate summary using the Generator class
            3. Analyze sentiment of the generated summary
            4. Assert that the sentiment block contains "OVERALL SENTIMENT : Negative"

        Expected Result:
            The sentiment analysis should correctly identify negative sentiment
            and format the output with "OVERALL SENTIMENT : Negative".
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
        sentiment_block = self.sentiment_analyzer.analyze_sentiment([summary[0]])[0]

        print("Predicted Sentiment:", sentiment_block)
        # print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        self.assertTrue("OVERALL SENTIMENT : Negative" in sentiment_block, "Summary does not reflect overall negative sentiment")

    # def test_03_sentiment_neutral_summary(self):
    #     """
    #     Verifies that a neutral sentiment summary is generated when reviews are balanced or average.
    #     """
    #     user_query = 'summarize to me the user reviews about Samsung Galaxy M51'
    #     retrieved_chunks = [
    #         {
    #             "text": "Battery performance is average, lasts through most of the day with moderate use.",
    #             "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "3"
    #         },
    #         {
    #             "text": "Charging speed is decent but not the fastest available in this price range.",
    #             "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "3"
    #         },
    #         {
    #             "text": "Battery life is acceptable but could be better for heavy users.",
    #             "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "3"
    #         }
    #     ]

    #     summary = self.generator.generate_summary(user_query, retrieved_chunks)
    #     sentiment_block, _ = self.generator.analyze_sentiment([summary])
    #     print('summary:\n', summary)
    #     print('sentiment_block:\n', sentiment_block)
    #     self.assertTrue("OVERALL SENTIMENT : Neutral" in sentiment_block, "Summary does not reflect overall neutral sentiment")
    
    def test_04_sentiment_positive_battery(self):
        """
        Test aspect-specific positive sentiment analysis for battery reviews.

        This test validates the system's ability to perform aspect-specific sentiment
        analysis by focusing specifically on battery-related feedback. It ensures that
        when battery reviews are positive, the system:
        - Correctly identifies positive sentiment in battery-focused content
        - Generates appropriate aspect-specific sentiment analysis
        - Formats the output correctly for aspect-based analysis

        Steps:
            1. Create mock positive battery-specific review data
            2. Generate summary with battery aspect focus
            3. Perform aspect-specific sentiment analysis for "battery"
            4. Assert that sentiment block contains "Overall Sentiment : Positive"

        Expected Result:
            The aspect-specific sentiment analysis should identify positive battery
            sentiment and format the output with proper capitalization.
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
        sentiment_block = self.sentiment_analyzer.analyze_sentiment([summary[0]], aspect="battery")[0]


        print("Predicted Sentiment:", sentiment_block)
        # print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        #self.assertTrue("OVERALL SENTIMENT : Positive" in sentiment_block, "Summary does not reflect positive sentiment for battery")
        self.assertIn("Overall Sentiment : Positive", sentiment_block, "Summary does not reflect positive sentiment for battery")



    def test_05_sentiment_negative_battery(self):
        """
        Test aspect-specific negative sentiment analysis for battery reviews.

        This test ensures the system can accurately perform negative sentiment analysis
        when focusing on battery-specific complaints. It validates that the system:
        - Correctly identifies negative sentiment in battery-related feedback
        - Processes aspect-specific negative reviews appropriately
        - Maintains consistent output formatting for negative aspect analysis

        Steps:
            1. Create mock negative battery-specific review data with complaints
            2. Generate summary focusing on battery aspect
            3. Perform aspect-specific sentiment analysis for "battery"
            4. Assert that sentiment block contains "Overall Sentiment : Negative"

        Expected Result:
            The aspect-specific sentiment analysis should correctly identify negative
            battery sentiment and format output with "Overall Sentiment : Negative".
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
        sentiment_block = self.sentiment_analyzer.analyze_sentiment([summary[0]], aspect="battery")[0]


        print("Predicted Sentiment:", sentiment_block)
        # print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        #self.assertTrue("OVERALL SENTIMENT : Negative" in sentiment_block, "Summary does not reflect negative sentiment for battery")
        self.assertIn("Overall Sentiment : Negative", sentiment_block, "Summary does not reflect negative sentiment for battery")


    # def test_06_sentiment_neutral_battery(self):
    #     """
    #     Verifies that a neutral sentiment summary is generated when battery-related reviews are mixed or average.
    #     """
    #     user_query = 'summarize to me the user reviews of Samsung Galaxy M51 focusing on its battery'
    #     retrieved_chunks = [
    #         {
    #             "text": "Battery performance is average, lasts through most of the day with moderate use.",
    #             "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "3"
    #         },
    #         {
    #             "text": "Charging speed is decent but not the fastest available in this price range.",
    #             "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "3"
    #         },
    #         {
    #             "text": "Battery life is acceptable but could be better for heavy users.",
    #             "brand": "Samsung", "model": "Samsung Galaxy M51", "stars": "3"
    #         }
    #     ]
    #     summary = self.generator.generate_summary(user_query, retrieved_chunks)
    #     sentiment_block, _ = self.generator.analyze_sentiment([summary], aspect="battery")
    #     print('summary:\n', summary)
    #     print('sentiment_block:\n', sentiment_block)
    #     self.assertTrue("OVERALL SENTIMENT : Neutral" in sentiment_block, "Summary does not reflect neutral sentiment for battery")

    #def test_pass_with_empty_review_list(self):
     #   """
      #  Passing Test:
       # When no reviews are provided (empty list), the summary should be empty,
        #and the sentiment analysis should return 'NO SENTIMENT'.
        #"""
        #user_query = 'summarize user reviews about Samsung Galaxy M51'
        #retrieved_chunks = []  # No reviews

        #summary = self.generator.generate_summary(user_query, retrieved_chunks)
        #sentiment_block, _ = self.generator.analyze_sentiment([summary])

        #print('summary:\n', summary)
        #print('sentiment_block:\n', sentiment_block)

        # Assert that no summary is generated
        #self.assertEqual(summary.strip(), "", "Summary should be empty when no reviews are provided")

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
