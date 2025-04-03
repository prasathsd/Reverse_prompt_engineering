import pandas as pd
from datasets import Dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re

class DataProcessor:
    def __init__(self):
        self.cleanup_patterns = [
            (r'<[^>]+>', ''),  # Remove HTML tags
            (r'\[.*?\]', ''),  # Remove square brackets content
            (r'\(.*?\)', ''),  # Remove parentheses content
            (r'\s+', ' '),     # Normalize whitespace
        ]
    
    def clean_text(self, text):
        """
        Clean and normalize text
        """
        if not isinstance(text, str):
            return ""
            
        text = text.strip()
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()
    
    def normalize_tamil(self, text):
        """
        Normalize Tamil text using transliteration
        """
        try:
            # Convert to ITRANS format first
            itrans = transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
            # Convert back to Tamil for consistency
            normalized = transliterate(itrans, sanscript.ITRANS, sanscript.TAMIL)
            return normalized
        except:
            return text
    
    def create_dataset(self, responses, prompts, languages):
        """
        Create a HuggingFace dataset from response-prompt pairs
        """
        df = pd.DataFrame({
            'response': responses,
            'prompt': prompts,
            'language': languages
        })
        
        # Clean and normalize text
        df['response'] = df['response'].apply(self.clean_text)
        df['prompt'] = df['prompt'].apply(self.clean_text)
        
        # Normalize Tamil text
        tamil_mask = df['language'] == 'ta'
        df.loc[tamil_mask, 'response'] = df.loc[tamil_mask, 'response'].apply(self.normalize_tamil)
        df.loc[tamil_mask, 'prompt'] = df.loc[tamil_mask, 'prompt'].apply(self.normalize_tamil)
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def prepare_batch(self, batch, tokenizer, max_length=512):
        """
        Prepare a batch for model input
        """
        # Tokenize responses
        response_encodings = tokenizer(
            batch['response'],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Tokenize prompts
        prompt_encodings = tokenizer(
            batch['prompt'],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': response_encodings['input_ids'],
            'attention_mask': response_encodings['attention_mask'],
            'labels': prompt_encodings['input_ids']
        }
    
    def filter_dataset(self, dataset, min_length=10, max_length=512):
        """
        Filter dataset based on length criteria
        """
        def length_filter(example):
            response_len = len(example['response'].split())
            prompt_len = len(example['prompt'].split())
            return (min_length <= response_len <= max_length and 
                   min_length <= prompt_len <= max_length)
        
        return dataset.filter(length_filter) 