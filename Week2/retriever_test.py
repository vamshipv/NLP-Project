from retriever import Retriever

# Writing test
def test_retriever():
    retriever = Retriever()
    
    # Adding some documents
    retriever.add_documents([
        "Books are better than movies in many ways.",
        "The Industrial Revolution began in the late 18th century, primarily in Britain.",
        "Photosynthesis is the process by which green plants convert sunlight into energy."
    ])
    
    # Positive test cases using assertTrue
    result = retriever.query("Are books preferred over movies?", top_k=1)
    assert result[0][0] == "Books are better than movies in many ways.", "Test Failed!"
    assertTrue("Books are better" in result[0][0], "Test Failed: Expected phrase not found")

    result = retriever.query("When did the Industrial Revolution begin?", top_k=1)
    assert "18th century" in result[0][0], "Test Failed: Industrial Revolution"
    assertTrue("18th century" in result[0][0], "Test Failed: Expected century not mentioned")
    
    # Positive test case using assertFalse
    result = retriever.query("What is photosynthesis?", top_k=1)
    assert "Photosynthesis is the process" in result[0][0], "Test Failed: Photosynthesis"
    assertFalse("Chlorophyll" in result[0][0], "Test Failed: Unrelated term found")
    
    # Negative test case using assertFalse
    result = retriever.query("What is the capital of France?", top_k=1)
    assertFalse(result[0][0] == "Paris is the capital of France.", "Test Failed: Incorrectly matched unrelated query")

    print("All Tests Passed!")

# Custom assertTrue and assertFalse functions
def assertTrue(expression, message):
    if not expression:
        raise AssertionError(message)

def assertFalse(expression, message):
    if expression:
        raise AssertionError(message)

test_retriever()