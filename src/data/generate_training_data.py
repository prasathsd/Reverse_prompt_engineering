import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import random
from typing import List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Don't use device map for CPU
        ).to(self.device)
        
        # Define complex prompt templates with multiple variables and conditions
        self.prompt_templates = {
            "complex_question": [
                "In the context of {domain}, what are the implications of {topic1} on {topic2}?",
                "How does {topic1} influence {topic2} in the context of {domain}?",
                "What are the key challenges in implementing {topic1} for {use_case}?",
                "In what ways can {topic1} be adapted to solve {problem}?",
                "What are the trade-offs between {topic1} and {topic2} when considering {domain}?"
            ],
            "multi_step_command": [
                "Create a system that {action1} and then {action2} while ensuring {constraint}",
                "Design a solution that {action1} while also {action2} and handles {constraint}",
                "Implement a process that {action1} and {action2} in sequence",
                "Develop a framework that {action1} and {action2} with {constraint}",
                "Build a pipeline that {action1} followed by {action2} considering {constraint}"
            ],
            "analytical_explanation": [
                "Analyze how {topic1} and {topic2} interact in {context} and their impact on {outcome}",
                "Compare the effectiveness of {topic1} vs {topic2} for {use_case}",
                "Evaluate the trade-offs between {topic1} and {topic2} in {context}",
                "Examine the relationship between {topic1} and {topic2} in {context}",
                "Investigate how {topic1} affects {topic2} in {context}"
            ],
            "creative_problem_solving": [
                "Design an innovative solution that addresses {problem} while considering {constraint}",
                "Create a novel approach to {problem} that incorporates {constraint}",
                "Develop a unique system that solves {problem} while balancing {constraint}",
                "Propose a creative solution for {problem} with {constraint}",
                "Imagine a future scenario where {problem} is solved using {constraint}"
            ]
        }
        
        # Define rich context domains and topics
        self.domains = [
            "healthcare", "finance", "education", "transportation", "energy",
            "environment", "manufacturing", "retail", "agriculture", "defense",
            "entertainment", "communication", "social media", "cybersecurity",
            "urban planning", "sustainable development", "digital transformation"
        ]
        
        self.topics = [
            "artificial intelligence", "machine learning", "deep learning",
            "natural language processing", "computer vision", "robotics",
            "data science", "cloud computing", "cybersecurity",
            "blockchain", "internet of things", "edge computing",
            "quantum computing", "augmented reality", "virtual reality",
            "5G networks", "autonomous systems", "digital twins",
            "predictive analytics", "natural language understanding"
        ]
        
        self.actions = [
            "process and analyze data", "implement security measures",
            "optimize resource allocation", "handle real-time updates",
            "manage user interactions", "process natural language",
            "generate visual content", "predict future trends",
            "automate complex workflows", "integrate multiple systems",
            "handle edge cases", "scale dynamically",
            "maintain data consistency", "ensure system reliability",
            "optimize performance metrics"
        ]
        
        self.constraints = [
            "maintaining data privacy", "ensuring system scalability",
            "minimizing resource usage", "maximizing user experience",
            "reducing latency", "improving accuracy",
            "ensuring reliability", "maintaining security",
            "optimizing costs", "reducing complexity"
        ]
        
        self.use_cases = [
            "real-time decision making", "predictive maintenance",
            "personalized recommendations", "automated customer service",
            "fraud detection", "quality control",
            "resource optimization", "risk assessment",
            "performance monitoring", "process automation"
        ]
    
    def generate_complex_prompt(self, prompt_type: str) -> str:
        """Generate a complex prompt using multiple variables and conditions."""
        template = random.choice(self.prompt_templates[prompt_type])
        
        if prompt_type == "complex_question":
            topic1 = random.choice(self.topics)
            topic2 = random.choice(self.topics)
            domain = random.choice(self.domains)
            use_case = random.choice(self.use_cases)
            problem = random.choice(self.use_cases)
            template = random.choice(self.prompt_templates[prompt_type])
            
            if "problem" in template:
                return template.format(topic1=topic1, problem=problem)
            elif "use_case" in template:
                return template.format(topic1=topic1, use_case=use_case)
            else:
                return template.format(topic1=topic1, topic2=topic2, domain=domain)
        
        elif prompt_type == "multi_step_command":
            action1 = random.choice(self.actions)
            action2 = random.choice(self.actions)
            constraint = random.choice(self.constraints)
            return template.format(action1=action1, action2=action2, constraint=constraint)
        
        elif prompt_type == "analytical_explanation":
            topic1 = random.choice(self.topics)
            topic2 = random.choice(self.topics)
            context = random.choice(self.domains)
            use_case = random.choice(self.use_cases)
            template = random.choice(self.prompt_templates[prompt_type])
            if "outcome" in template:
                return template.format(topic1=topic1, topic2=topic2, context=context, outcome=use_case)
            elif "use_case" in template:
                return template.format(topic1=topic1, topic2=topic2, use_case=use_case)
            else:
                return template.format(topic1=topic1, topic2=topic2, context=context)
        
        else:  # creative_problem_solving
            problem = random.choice(self.use_cases)
            constraint = random.choice(self.constraints)
            return template.format(problem=problem, constraint=constraint)
    
    def generate_response(self, prompt: str, max_length: int = 300) -> str:
        """Generate a detailed response using the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,  # Slightly higher for more creative responses
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=0.9,  # Nucleus sampling for better quality
            repetition_penalty=1.2  # Reduce repetition
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    
    def generate_dataset(self, num_samples: int = 1000) -> List[Tuple[str, str, str]]:
        """Generate a diverse dataset of complex prompt-response pairs."""
        dataset = []
        
        for _ in tqdm(range(num_samples), desc="Generating training data"):
            # Generate complex prompt
            prompt_type = random.choice(list(self.prompt_templates.keys()))
            prompt = self.generate_complex_prompt(prompt_type)
            
            # Generate detailed response
            response = self.generate_response(prompt)
            
            # Add to dataset
            dataset.append((response, prompt, prompt_type))
        
        return dataset
    
    def save_dataset(self, dataset: List[Tuple[str, str, str]], output_dir: str):
        """Save the generated dataset to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save English pairs
        en_pairs = [(r, p) for r, p, _ in dataset]
        with open(os.path.join(output_dir, "english_pairs.txt"), "w", encoding="utf-8") as f:
            for response, prompt in en_pairs:
                f.write(f"{response}\t{prompt}\n")
        
        # Save metadata with complexity metrics
        metadata = {
            "total_samples": len(dataset),
            "prompt_types": {
                "complex_question": sum(1 for _, _, t in dataset if t == "complex_question"),
                "multi_step_command": sum(1 for _, _, t in dataset if t == "multi_step_command"),
                "analytical_explanation": sum(1 for _, _, t in dataset if t == "analytical_explanation"),
                "creative_problem_solving": sum(1 for _, _, t in dataset if t == "creative_problem_solving")
            },
            "complexity_metrics": {
                "avg_prompt_length": sum(len(p) for _, p, _ in dataset) / len(dataset),
                "avg_response_length": sum(len(r) for r, _, _ in dataset) / len(dataset),
                "unique_domains_covered": len(set(self.domains)),
                "unique_topics_covered": len(set(self.topics))
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(dataset)} pairs to {output_dir}")

def main():
    # Initialize generator
    generator = TrainingDataGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(num_samples=1000)
    
    # Save dataset
    generator.save_dataset(dataset, "data/generated")
    
    logger.info("Dataset generation complete!")

if __name__ == "__main__":
    main() 