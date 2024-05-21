from transformers import AutoTokenizer

class HFTokenizer:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.eod = self.tokenizer.eos_token_id
        self.tokenize = self.tokenizer.encode   
        self.detokenize = self.tokenizer.decode 
        self.decoder = {value:key for key, value in self.tokenizer.get_vocab().items()}
        