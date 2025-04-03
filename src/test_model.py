from model.rpe_model import RPEModel
import torch
import sys
import time
from collections import defaultdict

def test_model():
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RPEModel(device=device)
    
    # Load the trained model
    model.load_model("checkpoints/final_model")
    
    # Test cases with different scenarios
    test_responses = [
        # Technical Implementation
        "We need to implement a real-time chat feature using WebSocket. The system should handle message queuing, user presence, and message delivery status. We'll use Redis for caching and RabbitMQ for message queuing.",
        
        # Problem Analysis
        "After analyzing the server logs, we found that the application crashes when handling more than 1000 concurrent users. The memory usage spikes during peak hours, and the garbage collector is running too frequently.",
        
        # Creative Design
        "Let's create a minimalist dashboard with a dark theme. We'll use a combination of purple and blue accents, with smooth animations for state transitions. The layout should be responsive and work well on both desktop and mobile.",
        
        # Question/Answer
        "Based on the performance metrics, the best approach would be to implement caching at multiple levels. We should use Redis for session data, implement browser caching for static assets, and add a CDN for global content delivery.",
        
        # Code Review
        "The pull request needs several improvements. The error handling is incomplete, there are no unit tests, and the code doesn't follow the project's style guide. We should also add documentation for the new API endpoints.",
        
        # System Architecture
        "For this microservices architecture, we'll need to implement service discovery, load balancing, and circuit breakers. Each service should be containerized and deployed independently, with proper monitoring and logging in place."
    ]
    
    print("Testing the model with different types of responses:\n")
    
    for i, response in enumerate(test_responses, 1):
        print(f"Test Case {i}:")
        print(f"Response: {response}")
        prompt = model.generate_prompt(response)
        print(f"Generated Prompt: {prompt}\n")
        print("-" * 80 + "\n")

def test_prompt_generation():
    """Test the model's ability to generate prompts for new, unseen examples"""
    model = RPEModel(device="cpu")
    
    test_cases = {
        "historical": [
            "The printing press was invented by Johannes Gutenberg in the 1440s, revolutionizing communication.",
            "The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing.",
            "Penicillin was accidentally discovered by Alexander Fleming in 1928, changing medicine forever."
        ],
        "medical": [
            "Dopamine acts as a neurotransmitter in the brain, influencing mood and movement.",
            "Type 2 diabetes occurs when cells become resistant to insulin, affecting blood sugar regulation.",
            "The immune system produces antibodies to fight off specific pathogens and infections."
        ],
        "technical": [
            "Docker containers package applications with their dependencies for consistent deployment.",
            "Binary search algorithms efficiently find items in sorted arrays by repeatedly dividing the search space.",
            "React components manage their own state and lifecycle, enabling dynamic UI updates."
        ],
        "creative": [
            "The old lighthouse stood sentinel against the crashing waves, its beam cutting through the fog.",
            "Autumn leaves danced on the wind, painting the sky with shades of gold and crimson.",
            "The letter trembled in her hands as tears blurred the faded ink of memories long past."
        ]
    }
    
    print("\nTesting RPE Model's Prompt Generation\n")
    print("=" * 80)
    
    for category, examples in test_cases.items():
        print(f"\n{category.upper()} EXAMPLES:")
        print("-" * 80)
        
        for example in examples:
            prompt = model.generate_prompt(example)
            print(f"\nInput: {example}")
            print(f"Generated Prompt: {prompt}")
        
        print("=" * 80)

def test_efficiency():
    """Test the model's efficiency and accuracy across different categories"""
    model = RPEModel(device="cpu")
    
    # Expanded test cases with expected categories
    test_cases = {
        "historical": [
            ("The Wright brothers made their first successful flight in 1903 at Kitty Hawk.", "historical_event"),
            ("The Great Wall of China was built over many centuries, starting as early as the 7th century BCE.", "historical_event"),
            ("The Renaissance began in Florence in the 14th century.", "historical_significance"),
            ("The telephone was patented by Alexander Graham Bell in 1876.", "historical_event"),
            ("The fall of the Berlin Wall in 1989 symbolized the end of the Cold War.", "historical_significance")
        ],
        "medical": [
            ("Antibiotics work by targeting bacterial cell walls while leaving human cells unaffected.", "medical_explanation"),
            ("High blood pressure can lead to heart disease and stroke if left untreated.", "medical_condition"),
            ("The pancreas produces insulin to regulate blood sugar levels.", "medical_explanation"),
            ("Vaccines stimulate the immune system to produce antibodies.", "medical_explanation"),
            ("Common cold symptoms include runny nose, sore throat, and cough.", "medical_condition")
        ],
        "technical": [
            ("Kubernetes orchestrates containerized applications across multiple hosts.", "technical_implementation"),
            ("Machine learning models learn patterns from data through iterative training.", "technical_concept"),
            ("Git uses a distributed architecture for version control.", "technical_explanation"),
            ("Python list comprehensions provide a concise way to create lists.", "technical_concept"),
            ("REST APIs use HTTP methods for client-server communication.", "technical_concept")
        ],
        "creative": [
            ("The moonlight cast silver shadows across the silent garden.", "creative_description"),
            ("Her laughter echoed through the empty halls like wind chimes.", "creative_description"),
            ("The old book's pages whispered stories of forgotten times.", "creative_scene"),
            ("Storm clouds gathered on the horizon, dark and menacing.", "creative_description"),
            ("Memories flooded back as she held the worn photograph.", "creative_scene")
        ]
    }
    
    print("\nTesting RPE Model Efficiency and Accuracy\n")
    print("=" * 80)
    
    # Initialize metrics
    total_time = 0
    category_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "time": 0})
    confidence_scores = defaultdict(list)
    
    # Test each category
    for category, examples in test_cases.items():
        print(f"\n{category.upper()} CATEGORY TEST:")
        print("-" * 80)
        
        for example, expected_type in examples:
            start_time = time.time()
            
            # Generate prompt and measure time
            prompt = model.generate_prompt(example)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Determine actual type from prompt
            actual_type = None
            if "historical" in prompt.lower():
                actual_type = "historical_event" if "when" in prompt.lower() else "historical_significance"
            elif "medical" in prompt.lower():
                actual_type = "medical_explanation" if "explain" in prompt.lower() else "medical_condition"
            elif any(tech in prompt.lower() for tech in ["technical", "tool", "framework", "concept"]):
                actual_type = "technical_" + ("implementation" if "how" in prompt.lower() else "concept")
            elif any(creative in prompt.lower() for creative in ["scene", "describe", "create"]):
                actual_type = "creative_" + ("scene" if "scene" in prompt.lower() else "description")
            
            # Update metrics
            is_correct = actual_type == expected_type if actual_type else False
            category_metrics[category]["correct"] += int(is_correct)
            category_metrics[category]["total"] += 1
            category_metrics[category]["time"] += elapsed_time
            
            # Calculate confidence score based on prompt specificity
            confidence = 0.0
            if actual_type:
                confidence += 0.5  # Base confidence for correct category
                confidence += 0.3 if actual_type == expected_type else 0  # Bonus for correct type
                confidence += 0.2 if len(prompt.split()) >= 8 else 0.1  # Bonus for detailed prompt
            confidence_scores[category].append(confidence)
            
            print(f"\nInput: {example}")
            print(f"Expected Type: {expected_type}")
            print(f"Generated Prompt: {prompt}")
            print(f"Processing Time: {elapsed_time:.3f}s")
            print(f"Confidence Score: {confidence:.2f}")
            print(f"Correct: {'✓' if is_correct else '✗'}")
        
        # Calculate category statistics
        accuracy = category_metrics[category]["correct"] / category_metrics[category]["total"] * 100
        avg_time = category_metrics[category]["time"] / category_metrics[category]["total"]
        avg_confidence = sum(confidence_scores[category]) / len(confidence_scores[category])
        
        print(f"\nCategory Statistics:")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average Processing Time: {avg_time:.3f}s")
        print(f"Average Confidence Score: {avg_confidence:.2f}")
        print("=" * 80)
    
    # Calculate overall statistics
    total_correct = sum(metrics["correct"] for metrics in category_metrics.values())
    total_examples = sum(metrics["total"] for metrics in category_metrics.values())
    overall_accuracy = total_correct / total_examples * 100
    avg_processing_time = total_time / total_examples
    overall_confidence = sum(sum(scores) for scores in confidence_scores.values()) / total_examples
    
    print("\nOVERALL PERFORMANCE:")
    print(f"Total Examples Processed: {total_examples}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print(f"Average Processing Time: {avg_processing_time:.3f}s")
    print(f"Average Confidence Score: {overall_confidence:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    test_efficiency()  # Run only efficiency test
    # Uncomment below to run other tests
    # test_model()
    # test_prompt_generation() 