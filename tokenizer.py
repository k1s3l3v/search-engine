import requests
import re
from bpe import Encoder


class Tokenizer:
    def __clean_html_and_whitespace(self, text):
        if text is None:
            return ''
        clean_text = re.sub('<[^<]+?>', '', text)  
        clean_text = re.sub(r'[\r\n\t\b\a]+', '', clean_text)  
        clean_text = re.sub(r'&[^;]*;', '', clean_text)
        return clean_text

    def __lower_text_rm_spec_symbols(self, text):
        if not text:
            return ''
        text = text.lower()
        text = re.sub(r'[^a-zа-я0-9,.\s]', '', text)
        return text

    """
    Preparing parsed data 
    To move to another source you need to change code below
    """

    def get_first_prepdata(self):
        source_url = "https://ai.gov.ru/local/export/iblocks.json"
        data = requests.get(source_url, verify=False)
        result = []

        for key, value in data.json().items():
            for item in value:
                result.append({"type": item["type"], "description": item["attributes"]["description"], "title": item["attributes"]["title"], "body": item["attributes"]["body"]})


        for doc in result:
            for key, value in doc.items():
                if key != 'type':
                    doc[key] = self.__clean_html_and_whitespace(value)
        self.data = result
        return self.data
    
    def get_string_data(self):
        for doc in self.data:
            for key, value in doc.items():
                if key != 'type':
                    doc[key] = self.__lower_text_rm_spec_symbols(value)
        self.data = '\n'.join([item['title'] + ' ' + item['description'] + ' ' + item['body'] for item in self.data])
        return self.data

    def set_encoder(self):
        bpe_encoder = Encoder(vocab_size=16000, pct_bpe=0.5)
        bpe_encoder.fit(self.data.split('\n'))
        self.bpe_encoder = bpe_encoder

    def __init__(self):
        self.data = []
        self.bpe_encoder = None
        self.get_first_prepdata()
        self.get_string_data()
        self.set_encoder()

    def get_transformed(self, text):
        return next(self.bpe_encoder.transform([text]))
    
    def get_tokens(self, text):
        return self.bpe_encoder.tokenize(text)
