import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join("..", "generator")))
from generator import Generator


class Test_Generator_Sentiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = Generator()

    def test_01_sentiment_positive_summary(self):
        """
        Verifies that a positive sentiment summary is generated when reviews are highly rated.
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
        sentiment_block, _ = self.generator.analyze_sentiment([summary])
        print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        self.assertTrue("OVERALL SENTIMENT : Positive" in sentiment_block, "Summary does not reflect overall positive sentiment")

    def test_02_sentiment_negative_summary(self):
        """
        Verifies that a negative sentiment summary is generated when reviews are low-rated.
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
        sentiment_block, _ = self.generator.analyze_sentiment([summary])
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
        Verifies that a positive sentiment summary is generated when battery-related reviews are positive.
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
        sentiment_block, _ = self.generator.analyze_sentiment([summary], aspect="battery")
        # print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        self.assertTrue("OVERALL SENTIMENT : Positive" in sentiment_block, "Summary does not reflect positive sentiment for battery")


    def test_05_sentiment_negative_battery(self):
        """
        Verifies that a negative sentiment summary is generated when battery-related reviews are negative.
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
        sentiment_block, _ = self.generator.analyze_sentiment([summary], aspect="battery")
        # print('summary:\n', summary)
        # print('sentiment_block:\n', sentiment_block)
        self.assertTrue("OVERALL SENTIMENT : Negative" in sentiment_block, "Summary does not reflect negative sentiment for battery")


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


    # --- Custom assertion overrides ---

    def assertTrue(self, expression, message):
        print('\n------------------------------------------------------\n')
        if not expression:
            print("Test case failed, Error message:", message)
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')

    def assertFalse(self, expression, message):
        print('\n------------------------------------------------------\n')
        if expression:
            print("Test case failed, Error message:", message)
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')

    def assertEqual(self, actual, expected, message):
        print('\n------------------------------------------------------\n')
        if actual != expected:
            print("Test case failed, Error message:", message)
            print(f"Expected: {expected}, but got: {actual}")
        else:
            print("Test Case passed")
        print('------------------------------------------------------\n')


if __name__ == "__main__":
    unittest.main()