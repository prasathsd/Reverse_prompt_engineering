# Response-Prompt Engine (RPE)

A sophisticated machine learning model for generating contextually appropriate prompts based on different types of content. The model excels at categorizing and generating prompts for various content types including medical, historical, technical, and creative writing.

## Features

### 1. Multi-Category Support
- **Medical Content (100% Accuracy)**
  - Medical conditions and symptoms
  - Treatment explanations
  - Biological processes
  - Healthcare concepts

- **Historical Content (40% Accuracy)**
  - Historical events
  - Significant dates
  - Cultural developments
  - Historical achievements

- **Creative Writing (40% Accuracy)**
  - Descriptive scenes
  - Emotional narratives
  - Character interactions
  - Setting descriptions

- **Technical Content (20% Accuracy)**
  - Programming concepts
  - Technology implementations
  - System architectures
  - Tool usage

### 2. Performance Metrics
- Average processing time: 0.002s
- Overall accuracy: 50%
- Confidence scoring: 0.83 average
- Real-time processing capability

### 3. Key Capabilities
- Pattern recognition with hierarchical analysis
- Confidence-based categorization
- Adaptive prompt generation
- Context-aware response formatting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rpe-model.git

# Install dependencies
pip install -r requirements.txt

# Set up environment
python setup.py install
```

## Usage

### Basic Usage
```python
from model.rpe_model import RPEModel

# Initialize the model
model = RPEModel(device="cpu")  # or "cuda" for GPU support

# Generate a prompt
response = "The pancreas produces insulin to regulate blood sugar levels."
prompt = model.generate_prompt(response)
print(prompt)  # Output: "Explain how this medical concept works and its importance?"
```

### Advanced Usage
```python
# Custom category testing
from test_model import test_efficiency

# Run efficiency tests
test_efficiency()

# Process batch inputs
responses = [
    "Antibiotics target bacterial cell walls.",
    "The Renaissance began in Florence.",
    "Python uses indentation for code blocks."
]

prompts = [model.generate_prompt(r) for r in responses]
```

## Function Documentation

### RPEModel Class

#### `generate_prompt(response: str) -> str`
Generates a contextually appropriate prompt based on the input response.
- **Parameters:**
  - response (str): The text to analyze
- **Returns:**
  - str: Generated prompt
- **Example:**
```python
prompt = model.generate_prompt("Dopamine influences mood and movement.")
# Returns: "Explain how this medical concept works and its importance?"
```

#### `match_patterns(text: str, category_patterns: dict) -> tuple`
Analyzes text for category-specific patterns and returns confidence scores.
- **Parameters:**
  - text (str): Input text to analyze
  - category_patterns (dict): Pattern definitions
- **Returns:**
  - tuple: (matches, confidence_score)

### Testing Functions

#### `test_efficiency()`
Comprehensive testing of model performance across categories.
- Tests accuracy, processing time, and confidence scores
- Generates detailed performance metrics
- Provides category-specific analysis

## Performance by Category

### Medical Content
- Perfect for medical explanations
- Highly accurate condition classification
- Reliable treatment-related prompts
- Example: "Antibiotics work by targeting bacterial cell walls."
  → "Explain how this medical concept works and its importance?"

### Historical Content
- Good at identifying significant events
- Handles dates and time periods
- Recognizes historical importance
- Example: "The Industrial Revolution began in Britain in the late 18th century."
  → "What is the historical significance of this event and its impact?"

### Creative Content
- Recognizes descriptive language
- Handles emotional content
- Processes narrative elements
- Example: "The moonlight cast silver shadows across the garden."
  → "Describe this scene using vivid and engaging language."

### Technical Content
- Basic technology recognition
- Handles programming concepts
- Identifies tool usage
- Example: "Kubernetes orchestrates containerized applications."
  → "How do you use this technical tool or framework?"

## Best Practices

1. **Input Preparation**
   - Provide clear, complete sentences
   - Include relevant context
   - Use domain-specific terminology

2. **Category Optimization**
   - Medical content works best
   - Historical content needs dates/significance
   - Creative content needs descriptive elements
   - Technical content needs specific terms

3. **Performance Optimization**
   - Batch similar categories together
   - Monitor confidence scores
   - Use appropriate device settings

## Known Limitations

1. Technical content classification needs improvement
2. Some historical events may be misclassified
3. Complex creative scenes might need better context

## Future Improvements

1. Enhanced technical pattern recognition
2. Improved historical event classification
3. More sophisticated creative content analysis
4. Extended category support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/rpe-model 