import re 
with open("data/the-verdict.txt", "r", encoding="utf-8") as f: 
    raw_text = f.read() 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()] 
print(len(preprocessed)) 