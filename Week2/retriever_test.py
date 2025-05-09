from retriever import Retriever

# Writing test
def test_retriever():
    retriever = Retriever()
    retriever.addDocuments("cats.txt")
    userInput = input("Enter anything about cat and sadness:  ")
    userInput.lower()
    retrievedChunks = retriever.query(userInput)
    retrievedChunks = " ".join(retrievedChunks).lower()

    # Positive assertions
    assertTrue("cat" in userInput.lower(), "Query should contain 'cat'")
    assertTrue("sad" in userInput.lower(), "Query should contain 'sad'")

    # Checks on the response content
    assertTrue("cat" in retrievedChunks, "'cat' should be mentioned in the results")
    assertTrue(
        any(keyword in retrievedChunks for keyword in ["unlucky", "rid of", "die", "sad"]),
        "Results should contain words indicating sadness"
    )
    # Negative assertions
    assertFalse("dog" not in retrievedChunks, "Did not expect 'dog' in the result")

# Custom assertTrue and assertFalse functions
def assertTrue(expression, message):
    if not expression:
        print("Test case failed", message)
    else:
        print("Test Case passed")

def assertFalse(expression, message):
    if not expression:
        print("Test case failed", message)
    else:
        print("Test Case passed")
        

test_retriever()