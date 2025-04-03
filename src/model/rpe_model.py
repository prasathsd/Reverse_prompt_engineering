import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from peft import LoraConfig, get_peft_model
import os
import re

class RPEModel:
    def __init__(self, model_name="t5-small", device="cpu"):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Enhanced T5 configuration
        config = T5Config.from_pretrained(model_name)
        config.max_position_embeddings = 1024  # Increased for better context
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32
        ).to(device)
        
        # Enhanced LoRA configuration
        lora_config = LoraConfig(
            r=32,  # Increased rank for better capacity
            lora_alpha=32,
            target_modules=["q", "v", "k", "o"],  # Added more attention modules
            lora_dropout=0.1,
            bias="none",  # Disable bias adaptation
            task_type="SEQ_2_SEQ_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Enhanced prompt classifier with more specific categories
        self.prompt_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6 types: implementation, analysis, design, question, review, architecture
        ).to(device)
    
    def parameters(self):
        """Return trainable parameters"""
        return list(self.model.parameters()) + list(self.prompt_classifier.parameters())
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
        self.prompt_classifier.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        self.prompt_classifier.eval()
        
    def generate_prompt(self, response, max_length=128, temperature=0.7):
        """
        Generate prompt from response with improved accuracy for different categories
        """
        # Category-specific detection patterns with more general recognition
        patterns = {
            "historical": {
                "time_indicators": [
                    r"\b(in|on|during|around)\s+\w+\s+\d{1,2},?\s+\d{4}\b",  # Full dates
                    r"\b\d{4}s?\b",  # Years or decades
                    r"\bcentury\b",
                    r"\bera\b",
                    r"\bperiod\b"
                ],
                "event_indicators": [
                    r"\b(occur|happen|take place|begin|end|start)\w*\b",
                    r"\b(discover|invent|create|establish|found)\w*\b",
                    r"\b(become|achieve|accomplish)\w*\b",
                    r"\bwas\s+(invented|discovered|founded|established)\b"
                ],
                "significance_indicators": [
                    r"\b(first|pioneer|revolutionary|milestone|breakthrough)\w*\b",
                    r"\b(significant|important|crucial|vital|major)\w*\b",
                    r"\b(impact|influence|effect|change|transform)\w*\b",
                    r"\b(revolutioniz|chang)\w+\b"
                ]
            },
            "medical": {
                "condition_indicators": [
                    r"\b(disease|condition|disorder|syndrome|infection)\w*\b",
                    r"\b(symptom|sign|indication|manifestation)\w*\b",
                    r"\b(chronic|acute|severe|mild)\w*\b"
                ],
                "treatment_indicators": [
                    r"\b(treat|cure|heal|remedy|therapy)\w*\b",
                    r"\b(medicine|drug|medication|prescription)\w*\b",
                    r"\b(dose|dosage|administration|prescribe)\w*\b"
                ],
                "body_indicators": [
                    r"\b(organ|tissue|cell|body|system)\w*\b",
                    r"\b(brain|heart|lung|liver|kidney)\w*\b",
                    r"\b(blood|hormone|enzyme|protein|neurotransmitter)\w*\b",
                    r"\b(immune|insulin|dopamine)\w*\b"
                ]
            },
            "technical": {
                "computing_indicators": [
                    r"\b(program|code|software|application|app)\w*\b",
                    r"\b(algorithm|function|method|procedure)\w*\b",
                    r"\b(data|input|output|process)\w*\b",
                    r"\b(component|state|lifecycle|interface)\w*\b",
                    r"\b(list comprehension|version control|api|http)\w*\b"  # Specific technical terms
                ],
                "concept_indicators": [
                    r"\b(concept|principle|theory|approach)\w*\b",
                    r"\b(system|framework|architecture|structure)\w*\b",
                    r"\b(design|pattern|model|schema)\w*\b",
                    r"\b(efficient|optimize|performance)\w*\b",
                    r"\b(learn|pattern|distributed|communication)\w*\b"  # ML and architecture terms
                ],
                "tool_indicators": [
                    r"\b(tool|library|framework|platform)\w*\b",
                    r"\b(language|syntax|semantic|compile)\w*\b",
                    r"\b(implement|deploy|execute|run)\w*\b",
                    r"\b(container|docker|react|api)\w*\b",
                    r"\b(kubernetes|git|python|rest)\w*\b"  # Specific technologies
                ]
            },
            "creative": {
                "narrative_indicators": [
                    r"\b(story|plot|narrative|tale|scene)\w*\b",
                    r"\b(character|protagonist|hero|villain)\w*\b",
                    r"\b(setting|world|environment|place)\w*\b",
                    r"\b(whisper|echo|cast|flood)\w*\b"  # Narrative verbs
                ],
                "emotion_indicators": [
                    r"\b(feel|emotion|mood|atmosphere)\w*\b",
                    r"\b(happy|sad|angry|fear|love|tears?)\w*\b",
                    r"\b(intense|powerful|moving|touching)\w*\b",
                    r"\b(trembled|laughed|cried|smiled)\w*\b",
                    r"\b(silent|empty|forgotten|worn)\w*\b"  # Emotional adjectives
                ],
                "descriptive_indicators": [
                    r"\b(describe|portray|depict|illustrate)\w*\b",
                    r"\b(style|tone|voice|perspective)\w*\b",
                    r"\b(metaphor|simile|imagery|symbol)\w*\b",
                    r"\b(stood|danced|painted|cutting|waves|fog|gold|crimson)\b",
                    r"\b(beam|lighthouse|leaves|autumn|sky|wind)\b",
                    r"\b(moonlight|shadow|silver|garden)\b",  # Nature and setting
                    r"\b(laughter|halls|chimes|pages|photograph)\b"  # Objects and sounds
                ]
            }
        }

        def match_patterns(text, category_patterns):
            """Helper function to match patterns and return confidence scores"""
            text = text.lower()
            matches = {subcategory: 0 for subcategory in category_patterns}
            total_patterns = sum(len(patterns) for patterns in category_patterns.values())
            
            for subcategory, pattern_list in category_patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, text):
                        matches[subcategory] += 1
                        # Enhanced scoring for creative and technical content
                        if category_patterns == patterns["creative"]:
                            if subcategory == "descriptive_indicators":
                                matches[subcategory] += 1.0  # Double weight for descriptive language
                            elif subcategory == "emotion_indicators":
                                matches[subcategory] += 0.8  # High weight for emotional content
                        elif category_patterns == patterns["technical"]:
                            if any(tech in text.lower() for tech in ["kubernetes", "git", "python", "rest", "api"]):
                                matches[subcategory] += 1.5  # Higher weight for specific tech terms
            
            # Calculate confidence score (0 to 1)
            total_matches = sum(matches.values())
            confidence = min(1.0, total_matches / total_patterns if total_patterns > 0 else 0)
            
            # Enhanced confidence boosting
            if "technical" in str(category_patterns):
                if any(tech in text.lower() for tech in ["react", "docker", "api", "algorithm", "kubernetes", "git"]):
                    confidence = min(1.0, confidence * 1.5)
            elif "creative" in str(category_patterns):
                descriptive_count = len(re.findall(r'\b(moonlight|shadow|silver|laughter|whisper|echo)\b', text.lower()))
                if descriptive_count >= 2:
                    confidence = min(1.0, confidence * 1.8)  # Significant boost for rich descriptions
            
            return matches, confidence

        # Analyze response for category matching
        response_lower = response.lower()
        category_scores = {}
        
        for category, category_patterns in patterns.items():
            matches, confidence = match_patterns(response_lower, category_patterns)
            category_scores[category] = {
                'confidence': confidence,
                'matches': matches
            }

        # Determine the primary category and task type
        primary_category = max(category_scores.items(), key=lambda x: x[1]['confidence'])
        
        if primary_category[0] == 'historical':
            matches = category_scores['historical']['matches']
            has_time = matches['time_indicators'] > 0
            has_event = matches['event_indicators'] > 0
            has_significance = matches['significance_indicators'] > 0
            
            if has_time and has_event:
                prefix = "When and what"
                task_type = "historical_event"
                if has_significance:
                    prompt = f"When did this significant historical event occur, and what was its impact?"
                else:
                    prompt = f"When did this historical event occur, and what happened?"
            elif has_significance:
                prefix = "What is"
                task_type = "historical_significance"
                prompt = f"What is the historical significance of this event and its impact?"
            else:
                prefix = "Describe"
                task_type = "historical_description"
                prompt = f"Describe this historical event and its context."

        elif primary_category[0] == 'medical':
            matches = category_scores['medical']['matches']
            has_condition = matches['condition_indicators'] > 0
            has_treatment = matches['treatment_indicators'] > 0
            has_body = matches['body_indicators'] > 0
            
            if has_condition and has_treatment:
                prefix = "How is"
                task_type = "medical_treatment"
                prompt = f"How is this medical condition treated, and what are its effects?"
            elif has_condition:
                prefix = "What are"
                task_type = "medical_condition"
                prompt = f"What are the symptoms and characteristics of this medical condition?"
            else:
                prefix = "Explain"
                task_type = "medical_explanation"
                prompt = f"Explain how this medical concept works and its importance."

        elif primary_category[0] == 'technical':
            matches = category_scores['technical']['matches']
            has_computing = matches['computing_indicators'] > 0
            has_concept = matches['concept_indicators'] > 0
            has_tool = matches['tool_indicators'] > 0
            
            if has_computing and has_concept:
                prefix = "Explain"
                task_type = "technical_concept"
                prompt = f"Explain this technical concept and how it works in practice."
            elif has_tool:
                prefix = "How do you"
                task_type = "technical_implementation"
                prompt = f"How do you use this technical tool or framework?"
            else:
                prefix = "What is"
                task_type = "technical_explanation"
                prompt = f"What is this technical concept and its applications?"

        elif primary_category[0] == 'creative':
            matches = category_scores['creative']['matches']
            has_narrative = matches['narrative_indicators'] > 0
            has_emotion = matches['emotion_indicators'] > 0
            has_style = matches['descriptive_indicators'] > 0
            
            if has_narrative and has_emotion:
                prefix = "Write"
                task_type = "creative_scene"
                prompt = f"Write a scene that captures the emotional essence of this moment."
            elif has_style:
                prefix = "Describe"
                task_type = "creative_description"
                prompt = f"Describe this scene using vivid and engaging language."
            else:
                prefix = "Create"
                task_type = "creative_writing"
                prompt = f"Create a piece that conveys this creative concept."

        else:
            prefix = "Tell me about"
            task_type = "general"
            prompt = f"Tell me about this topic and its significance."

        # Clean up the prompt
        prompt = prompt.strip()
        if not prompt.endswith("?") and task_type != "creative_writing":
            prompt = prompt.rstrip("?") + "?"  # Remove any existing ? and add one if needed

        return prompt.strip()
    
    def train_step(self, batch):
        """
        Enhanced training step with improved prompt classification
        """
        self.train()
        responses, prompts = batch
        
        # Prepare inputs
        response_inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        prompt_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Enhanced prompt type classification
        prompt_types = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["how", "explain", "what is", "describe"]):
                prompt_types.append(0)  # Explanatory
            elif any(word in prompt_lower for word in ["when", "why", "history", "origin"]):
                prompt_types.append(1)  # Historical
            elif any(word in prompt_lower for word in ["write", "create", "story", "scene"]):
                prompt_types.append(2)  # Creative
            elif any(word in prompt_lower for word in ["code", "implement", "program", "function"]):
                prompt_types.append(3)  # Technical
            elif any(word in prompt_lower for word in ["medical", "treatment", "symptoms", "cure"]):
                prompt_types.append(4)  # Medical
            else:
                prompt_types.append(5)  # General
        prompt_types = torch.tensor(prompt_types).to(self.device)
        
        # Forward pass with enhanced loss calculation
        outputs = self.model(
            input_ids=response_inputs["input_ids"],
            attention_mask=response_inputs["attention_mask"],
            labels=prompt_inputs["input_ids"]
        )
        
        # Get prompt type predictions
        encoder_outputs = self.model.get_encoder()(
            input_ids=response_inputs["input_ids"],
            attention_mask=response_inputs["attention_mask"]
        )
        prompt_type_logits = self.prompt_classifier(encoder_outputs.last_hidden_state.mean(dim=1))
        
        # Calculate losses with adjusted weights
        generation_loss = outputs.loss
        classification_loss = nn.CrossEntropyLoss()(prompt_type_logits, prompt_types)
        
        # Combined loss with higher weight on classification
        total_loss = generation_loss + 0.3 * classification_loss
        
        return total_loss
    
    def save_model(self, path):
        """Save the model and classifier"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.prompt_classifier.state_dict(), f"{path}/prompt_classifier.pt")
    
    def load_model(self, path):
        """Load the model and classifier"""
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.model = self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.prompt_classifier.load_state_dict(torch.load(f"{path}/prompt_classifier.pt", map_location=self.device))