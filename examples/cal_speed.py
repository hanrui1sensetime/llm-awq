import jsonlines
from transformers import BloomTokenizerFast, LlamaTokenizer

minute = 14
second = 21

tokenizer = LlamaTokenizer.from_pretrained(
    "/root/workspace/external_data/pjllama13bv7")
with open('predict/pjllama13bv7-kvcache/webMedQA.jsonl') as f:
    sum_token = 0
    for item in jsonlines.Reader(f):
        ans = item['predict_answer']
        ans_token = tokenizer.tokenize(ans)
        sum_token += len(ans_token)
    print(sum_token)
    print(sum_token / (minute * 60 + second))
