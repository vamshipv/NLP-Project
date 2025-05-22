import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator

def test_retriever(test_case):
    """
    Tests the Retriever class by:
    - Adding a document ("cats.txt")
    - Taking a user query involving 'cat' and 'sad'
    - Querying for relevant text chunks
    - Checking for expected and unexpected keywords in the query and results

    Assertions:
    - The input query must include 'cat' and 'sad'
    - The retrieved content should mention 'cat' and contain sad-related words
    - The retrieved content should not contain unrelated words like 'dog'
    """
    retriever = Retriever()
    retriever.addDocuments("cats.txt")
    # userInput = "Cat is sad"
    test_case.lower()
    retrievedChunks = retriever.query(test_case)
    retrievedChunks = " ".join(retrievedChunks).lower()

    assertTrue("cat" in test_case.lower(), "Query should contain 'cat'")
    assertTrue("sad" in test_case.lower(), "Query should contain 'sad'")

    assertTrue("cat" in retrievedChunks, "'cat' should be mentioned in the results")
    assertTrue(
        any(keyword in retrievedChunks for keyword in ["unlucky", "rid of", "die", "sad"]),
        "Results should contain words indicating sadness"
    )

    assertFalse("dog" not in retrievedChunks, "Did not expect 'dog' in the result")

def assertTrue(expression, message):
    """
    Custom assert function that prints test result for a true condition.

    Args:
        expression (bool): The boolean expression to evaluate.
        message (str): The message to print if the assertion fails.
    """
    if not expression:
        print("Test case failed", message)
    else:
        print("Test Case passed")

def assertFalse(expression, message):
    """
    Custom assert function that prints test result for a false condition.

    Args:
        expression (bool): The boolean expression to evaluate.
        message (str): The message to print if the assertion fails.
    """
    if not expression:
        print("Test case failed", message)
    else:
        print("Test Case passed")
        

print("Test cases related to Cat and Sadness")
test_case1_words = "cat is sad"
test_case1 = (test_case1_words)
test_retriever(test_case1)

print("Test cases related to Dog and Happy")
test_case2_words = "Dog is happy"
test_case2 = (test_case2_words)
test_retriever(test_case2)