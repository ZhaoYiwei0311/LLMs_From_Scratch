import re
from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2

with open("data/the-verdict.txt", "r", encoding="utf-8") as f: 
    raw_text = f.read() 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()] 
all_words = sorted(set(preprocessed)) 
vocab_size = len(all_words) 


all_tokens = sorted(list(set(preprocessed))) 
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) 
vocab = {token:integer for integer,token in enumerate(all_tokens)} 
print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]): 
#     print(item) 
    
    
tokenizer = SimpleTokenizerV2(vocab) 
# text = """"It's the last he painted, you know," 
#        Mrs. Gisburn said with pardonable pride.""" 
# ids = tokenizer.encode(text) 
# print(ids)

text1 = "Hello, do you like tea?" 
text2 = "In the sunlit terraces of the palace." 
text = " <|endoftext|> ".join((text1, text2)) 
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))