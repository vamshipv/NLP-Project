import json
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
import unittest

class TestGen(unittest.TestCase):
    def test_generator(self):

        gen = Generator()
        ret = Retriever()
        ret.addDocuments("baseline/data/winnie_the_pooh.txt")
    # Open and read the .jsonl file
        with open('baseline/generator/test_ques_and_ans.jsonl', 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                question = item.get("question")
                expected_answer = item.get("expected_answer_contains")
                print(f"Question is : {question}")
                # print(f"Expected Answer: {expected_answer}")

                retrievedChunks = ret.query(question)
                context = "\n\n".join(retrievedChunks)
                answer = gen.generate_answer(retrievedChunks, context, question, group_id="Davvee")
                print("Answer is : ", answer)
                print("Expected Answer is : ", expected_answer)
                try:
                    self.assertEqual(answer, expected_answer)
                    print("--------------Test passed--------------")
                except AssertionError:
                    print("--------------Test failed--------------")    

test = TestGen()
test.test_generator()
        