import bz2
import mwxml
import re
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikiProcessor:
    def __init__(self, min_text_length: int = 50, max_text_length: int = 1000):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.cleanup_patterns = [
            (r'==.*?==', ''),  # Remove section headers
            (r'\[\[.*?\]\]', ''),  # Remove wiki links
            (r'{{.*?}}', ''),  # Remove templates
            (r'<ref.*?</ref>', ''),  # Remove references
            (r'<.*?>', ''),  # Remove HTML tags
            (r'\s+', ' '),  # Normalize whitespace
        ]
    
    def clean_wiki_text(self, text: str) -> str:
        """Clean Wikipedia text by removing wiki markup and formatting."""
        text = text.strip()
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text meets our quality criteria."""
        if not text:
            return False
        
        # Check length
        if not (self.min_text_length <= len(text) <= self.max_text_length):
            return False
        
        # Check for common issues
        if any(x in text.lower() for x in ['disambiguation', 'redirect', 'stub']):
            return False
        
        # Check for minimum word count
        if len(text.split()) < 10:
            return False
        
        return True
    
    def process_english_dump(self, dump_path: str) -> List[Tuple[str, str]]:
        """Process English Wikipedia dump and extract relevant text sections."""
        logger.info(f"Processing English Wikipedia dump: {dump_path}")
        pairs = []
        
        try:
            dump = mwxml.Dump.from_file(open(dump_path, 'rb'))
            
            for page in tqdm(dump, desc="Processing English pages"):
                if page.namespace == 0:  # Main namespace
                    for revision in page:
                        if revision.text:
                            text = self.clean_wiki_text(revision.text)
                            if self.is_valid_text(text):
                                # Split into sentences for potential prompt-response pairs
                                sentences = re.split(r'[.!?]+', text)
                                for i in range(0, len(sentences)-1, 2):
                                    if len(sentences[i]) > 20 and len(sentences[i+1]) > 20:
                                        pairs.append((sentences[i+1].strip(), sentences[i].strip()))
                        break  # Only process latest revision
        except Exception as e:
            logger.error(f"Error processing English dump: {str(e)}")
        
        return pairs
    
    def process_tamil_dump(self, dump_path: str) -> List[Tuple[str, str]]:
        """Process Tamil Wikipedia dump and extract relevant text sections."""
        logger.info(f"Processing Tamil Wikipedia dump: {dump_path}")
        pairs = []
        
        try:
            dump = mwxml.Dump.from_file(open(dump_path, 'rb'))
            
            for page in tqdm(dump, desc="Processing Tamil pages"):
                if page.namespace == 0:  # Main namespace
                    for revision in page:
                        if revision.text:
                            text = self.clean_wiki_text(revision.text)
                            if self.is_valid_text(text):
                                # Split into sentences for potential prompt-response pairs
                                sentences = re.split(r'[.!?]+', text)
                                for i in range(0, len(sentences)-1, 2):
                                    if len(sentences[i]) > 20 and len(sentences[i+1]) > 20:
                                        pairs.append((sentences[i+1].strip(), sentences[i].strip()))
                        break  # Only process latest revision
        except Exception as e:
            logger.error(f"Error processing Tamil dump: {str(e)}")
        
        return pairs
    
    def save_pairs(self, pairs: List[Tuple[str, str]], output_file: str):
        """Save extracted pairs to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for response, prompt in pairs:
                f.write(f"{response}\t{prompt}\n")
        logger.info(f"Saved {len(pairs)} pairs to {output_file}")

def main():
    # Initialize processor
    processor = WikiProcessor()
    
    # Process English Wikipedia dump
    en_dump_path = r"C:\Users\Welcome\Downloads\enwiki-20250301-pages-articles-multistream-index1.txt-p1p41242.bz2"
    en_pairs = processor.process_english_dump(en_dump_path)
    processor.save_pairs(en_pairs, "data/english_pairs.txt")
    
    # Process Tamil Wikipedia dump
    ta_dump_path = r"C:\Users\Welcome\Downloads\tkwiki-20250301-pages-articles-multistream.xml.bz2"
    ta_pairs = processor.process_tamil_dump(ta_dump_path)
    processor.save_pairs(ta_pairs, "data/tamil_pairs.txt")
    
    logger.info(f"Processed {len(en_pairs)} English pairs and {len(ta_pairs)} Tamil pairs")

if __name__ == "__main__":
    main() 