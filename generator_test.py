import json
from retriever import Retriever
from generator import Generator
import unittest

class TestGen(unittest.TestCase):
    
    def baseCase(self):
        return

    def test_generator(self):
        gen = Generator()
        ret = Retriever()
        ret.addDocuments("winnie_the_pooh.txt")
    # Open and read the .jsonl file
        with open('test_ques_and_ans.jsonl', 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                question = item.get("question")
                if question is not None:
                    expected_answer = item.get("expected_answer_contains")
                    print(f"Question is : {question}")

                    retrievedChunks = ret.query(question)
                    context = "\n\n".join(retrievedChunks)
                    answer = gen.generate_answer(retrievedChunks, context, question, group_id="Dave")
                    print("Answer is : ", answer)
                    print("Expected Answer is : ", expected_answer)
                    self.AssertCheck(answer, expected_answer)

    def AssertCheck(self, answer, expected_answer):
        try:
            self.assertEqual(answer, expected_answer)
            print("--------------Test passed--------------")
        except AssertionError:
            print("--------------Test failed--------------")    

    def groundedChunks(self):
        gen = Generator()
        ret = Retriever()
        ret.addDocuments("winnie_the_pooh.txt")
        with open('test_ques_and_ans.jsonl', 'r') as file:
            for line in file:
                line = line.strip()
                item = json.loads(line)
                if item["id"] == "6":
                    question = item.get("question1")
                    retrievedChunks = ret.query(question)
                    context = "\n\n".join(retrievedChunks)
                    testChunks = item.get("retrieved_chunks")
                    answer = gen.generate_answer(retrievedChunks, context, question, group_id="Dave")
                    if answer in context:
                        answer = "Chunks are present"
                        expected_answer = item.get("expected_answer_contains")
                        question = "Is the generated answer is present in the chunks"
                        print("Question is :", question)
                        print("Answer is : ", answer)
                        print("Expected Answer is : ", expected_answer)
                        self.AssertCheck(answer, expected_answer)
                    else:
                        answer = "Chunks are not present"
                        print("Answer is : ", answer)
                        print("Expected Answer is : ", expected_answer)
                        self.AssertCheck(answer, expected_answer)

    def nonEmptyString(self):
        return
    
test = TestGen()
test.test_generator()
test.groundedChunks() 