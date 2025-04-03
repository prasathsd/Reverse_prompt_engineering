from model.rpe_model import RPEModel

def demonstrate_categories():
    """Demonstrate the model's capabilities across different categories"""
    
    # Initialize model
    model = RPEModel(device="cpu")
    
    # Example responses for each category
    examples = {
        "Medical": [
            "Antibiotics work by targeting bacterial cell walls while leaving human cells unaffected.",
            "The immune system produces antibodies to fight off specific pathogens.",
            "Dopamine acts as a neurotransmitter in the brain, influencing mood."
        ],
        "Historical": [
            "The Wright brothers made their first successful flight in 1903.",
            "The Industrial Revolution began in Britain in the late 18th century.",
            "The Renaissance transformed European art and culture."
        ],
        "Technical": [
            "Docker containers package applications with their dependencies.",
            "Python list comprehensions provide a concise way to create lists.",
            "REST APIs use HTTP methods for client-server communication."
        ],
        "Creative": [
            "The moonlight cast silver shadows across the silent garden.",
            "Autumn leaves danced on the wind, painting the sky gold.",
            "Her laughter echoed through the empty halls like wind chimes."
        ]
    }
    
    # Process each category
    for category, responses in examples.items():
        print(f"\n{category.upper()} EXAMPLES:")
        print("-" * 80)
        
        for response in responses:
            prompt = model.generate_prompt(response)
            print(f"\nInput: {response}")
            print(f"Generated Prompt: {prompt}")
        
        print("=" * 80)

def batch_processing_example():
    """Demonstrate batch processing capabilities"""
    
    model = RPEModel(device="cpu")
    
    # Mixed category batch
    responses = [
        "Antibiotics target bacterial cell walls.",
        "The Renaissance began in Florence.",
        "Python uses indentation for code blocks.",
        "The moonlight painted shadows on water."
    ]
    
    print("\nBATCH PROCESSING EXAMPLE:")
    print("-" * 80)
    
    # Process batch
    prompts = [model.generate_prompt(r) for r in responses]
    
    # Show results
    for response, prompt in zip(responses, prompts):
        print(f"\nInput: {response}")
        print(f"Generated Prompt: {prompt}")
    
    print("=" * 80)

if __name__ == "__main__":
    print("RPE Model Usage Examples\n")
    demonstrate_categories()
    batch_processing_example() 