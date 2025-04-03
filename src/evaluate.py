import torch
from model.rpe_model import RPEModel
import logging
from tqdm import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

# Download required NLTK data
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path):
    """Load test data in the same format as training data"""
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line == '.':
                i += 1
                continue
            
            if not line.startswith('\t') and '\t' in line:
                parts = line.split('\t')
                response = parts[0].strip()
                prompt = parts[-1].strip()
                if response and prompt:
                    pairs.append((response, prompt))
                i += 1
                continue
            
            if line.startswith('\t'):
                prompt = line[1:].strip()
                response = ""
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line == '.' or line.startswith('\t'):
                        break
                    response += line + " "
                    i += 1
                if response.strip() and prompt:
                    pairs.append((response.strip(), prompt))
                continue
            i += 1
    return pairs

def calculate_similarity_metrics(original_prompt, reconstructed_prompt):
    """Calculate multiple similarity metrics between original and reconstructed prompts"""
    # Prepare tokenized sequences
    original_tokens = word_tokenize(original_prompt.lower())
    reconstructed_tokens = word_tokenize(reconstructed_prompt.lower())
    
    # 1. Word overlap (Jaccard similarity)
    original_words = set(original_tokens)
    reconstructed_words = set(reconstructed_tokens)
    intersection = len(original_words.intersection(reconstructed_words))
    union = len(original_words.union(reconstructed_words))
    jaccard = intersection / union if union > 0 else 0
    
    # 2. BLEU score
    try:
        bleu = sentence_bleu([original_tokens], reconstructed_tokens, weights=(0.5, 0.5))
    except:
        bleu = 0
    
    # 3. Length ratio (penalize if lengths are very different)
    len_ratio = min(len(reconstructed_tokens), len(original_tokens)) / max(len(reconstructed_tokens), len(original_tokens))
    
    # 4. Word order similarity
    min_len = min(len(original_tokens), len(reconstructed_tokens))
    order_matches = sum(1 for i in range(min_len) if original_tokens[i] == reconstructed_tokens[i])
    order_similarity = order_matches / min_len if min_len > 0 else 0
    
    # Combine metrics with weights
    weighted_score = (
        0.3 * jaccard +  # Word overlap
        0.3 * bleu +     # BLEU score
        0.2 * len_ratio + # Length similarity
        0.2 * order_similarity  # Word order
    )
    
    return {
        "jaccard": jaccard,
        "bleu": bleu,
        "length_ratio": len_ratio,
        "order_similarity": order_similarity,
        "weighted_score": weighted_score
    }

def evaluate_model(model, test_pairs, num_samples=10):
    """Evaluate model on test pairs"""
    results = []
    logger.info(f"Evaluating model on {num_samples} samples...")
    
    total_metrics = {
        "jaccard": 0,
        "bleu": 0,
        "length_ratio": 0,
        "order_similarity": 0,
        "weighted_score": 0
    }
    
    for i, (response, original_prompt) in enumerate(test_pairs[:num_samples]):
        # Generate reconstructed prompt
        reconstructed_prompt = model.reconstruct_prompt(response)
        
        # Calculate similarity metrics
        metrics = calculate_similarity_metrics(original_prompt, reconstructed_prompt)
        
        # Update total metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        results.append({
            "original_prompt": original_prompt,
            "reconstructed_prompt": reconstructed_prompt,
            "metrics": metrics
        })
        
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Original Prompt: {original_prompt}")
        logger.info(f"Reconstructed Prompt: {reconstructed_prompt}")
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")
    
    # Calculate averages
    avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
    logger.info("\nAverage Metrics:")
    for key, value in avg_metrics.items():
        logger.info(f"  {key}: {value:.3f}")
    
    return results, avg_metrics

def main():
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = RPEModel(device=device)
    model.load_model("checkpoints/final_model")
    
    # Load test data
    test_pairs = load_test_data("data/generated/english_pairs.txt")
    logger.info(f"Loaded {len(test_pairs)} test pairs")
    
    # Evaluate model
    results, avg_metrics = evaluate_model(model, test_pairs)
    
    # Save results
    output = {
        "results": results,
        "average_metrics": avg_metrics
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    main() 