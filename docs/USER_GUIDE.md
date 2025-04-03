# RPE Model User Guide

## Introduction

This guide will help you understand how to effectively use the Reverse Prompt Engineering (RPE) model to reconstruct prompts from AI responses.

## Getting Started

### 1. Installation

Follow the installation instructions in the [README.md](README.md) file to set up the environment and install dependencies.

### 2. Starting the Web Interface

```bash
python src/app.py
```

Access the interface at `http://localhost:7860`

## Using the Web Interface

### 1. Single Response Processing

1. **Enter Response**:
   - Type or paste the AI response in the text box
   - You can use either English or Tamil responses

2. **Adjust Temperature**:
   - Use the slider to control creativity
   - Lower values (0.1-0.3): More precise, factual
   - Higher values (0.7-1.0): More creative, varied

3. **Generate Prompt**:
   - Click "Generate Prompt"
   - Wait for the result
   - The reconstructed prompt will appear in the output box

### 2. Batch Processing

1. **Enter Multiple Responses**:
   - Type or paste multiple responses
   - One response per line
   - Can mix English and Tamil

2. **Process Batch**:
   - Click "Process Batch"
   - Results will appear in a table
   - Each row shows response and reconstructed prompt

### 3. Using Examples

1. **Try Pre-loaded Examples**:
   - Go to the Examples tab
   - Click on any example
   - See how the model reconstructs different types of prompts

## Best Practices

### 1. Response Formatting

- **Clear Responses**: Use well-structured, clear responses
- **Length**: Keep responses between 10-500 characters
- **Language**: Clearly indicate if mixing languages

### 2. Temperature Settings

- **Factual Content**: Use lower temperature (0.1-0.3)
- **Creative Content**: Use higher temperature (0.7-1.0)
- **Mixed Content**: Use medium temperature (0.4-0.6)

### 3. Prompt Types

The model handles various prompt types:

1. **Simple Factual**:
   - Example: "The capital of France is Paris."
   - Best temperature: 0.1-0.3

2. **Complex Questions**:
   - Example: "To solve this equation, first isolate x."
   - Best temperature: 0.3-0.5

3. **Multi-step Commands**:
   - Example: "First, create a function. Then, add error handling."
   - Best temperature: 0.4-0.6

4. **Analytical Explanations**:
   - Example: "The process involves three main steps..."
   - Best temperature: 0.5-0.7

5. **Creative Problem Solving**:
   - Example: "Consider alternative approaches..."
   - Best temperature: 0.7-1.0

## Troubleshooting

### 1. Common Issues

- **Empty Response**: Make sure to enter text before generating
- **Long Response**: Break into smaller chunks if too long
- **Mixed Languages**: Use clear language separation

### 2. Error Messages

- **"Error generating prompt"**: Try again with adjusted temperature
- **"Response too long"**: Split into smaller parts
- **"Invalid input"**: Check for special characters or formatting

## Advanced Usage

### 1. API Integration

```python
from model.rpe_model import RPEModel

# Initialize
model = RPEModel(device="cuda" if torch.cuda.is_available() else "cpu")

# Single response
response = "Your AI response here"
prompt = model.generate_prompt(response, temperature=0.7)

# Batch processing
responses = ["response1", "response2"]
results = model.batch_predict(responses, temperature=0.7)
```

### 2. Custom Training

1. **Prepare Data**:
   - Format: (response, prompt) pairs
   - Include both English and Tamil examples

2. **Train Model**:
   ```bash
   python src/train.py --data_path your_data.json
   ```

### 3. Performance Optimization

- Use GPU if available
- Adjust batch size based on memory
- Cache common responses

## Support

For additional help:
1. Check the [documentation](README.md)
2. Review [architecture details](ARCHITECTURE.md)
3. Open an issue on GitHub

## Feedback

We welcome feedback to improve the model:
1. Report issues on GitHub
2. Suggest new features
3. Share success stories 