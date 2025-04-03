# RPE Model Architecture

## System Overview

The Reverse Prompt Engineering (RPE) model is a sophisticated system that combines several components to effectively reconstruct prompts from AI responses. This document details the technical architecture and implementation.

## Core Components

### 1. Model Architecture

#### Base Model (T5)
- **Model**: T5-Small (60M parameters)
- **Configuration**:
  ```python
  config = T5Config.from_pretrained(model_name)
  config.max_position_embeddings = 1024
  config.relative_attention_num_buckets = 32
  config.relative_attention_max_distance = 128
  ```

#### LoRA Adapter
- **Configuration**:
  ```python
  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q", "v"],
      lora_dropout=0.05,
      bias="none",
      task_type="SEQ_2_SEQ_LM"
  )
  ```

#### Prompt Type Classifier
- **Architecture**:
  ```python
  nn.Sequential(
      nn.Linear(config.hidden_size, 256),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(256, 5)
  )
  ```

### 2. Training Pipeline

#### Data Generation
```python
def generate_training_data():
    # Generate synthetic data
    english_pairs = generate_english_pairs()
    tamil_pairs = generate_tamil_pairs()
    mixed_pairs = generate_mixed_pairs()
    
    # Combine and shuffle
    all_pairs = english_pairs + tamil_pairs + mixed_pairs
    random.shuffle(all_pairs)
    
    return all_pairs
```

#### Training Process
```python
def train_model(model, train_loader, num_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss = model.train_step(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
```

### 3. Inference Pipeline

#### Single Response Processing
```python
def generate_prompt(response, temperature=0.7):
    # Add prefix
    prefixed_response = "Reconstruct the original prompt from this response: " + response
    
    # Tokenize
    inputs = tokenizer(prefixed_response, return_tensors="pt")
    
    # Generate
    outputs = model.generate(
        **inputs,
        temperature=temperature,
        max_length=128
    )
    
    return tokenizer.decode(outputs[0])
```

#### Batch Processing
```python
def batch_predict(responses, temperature=0.7):
    results = []
    for response in responses:
        prompt = generate_prompt(response, temperature)
        results.append((response, prompt))
    return results
```

## Data Flow

1. **Input Processing**:
   - Response text received
   - Special prefix added
   - Tokenization performed

2. **Model Processing**:
   - Encoder processes input
   - Prompt type classified
   - Decoder generates prompt

3. **Output Processing**:
   - Generated text decoded
   - Post-processing applied
   - Final prompt returned

## Performance Optimization

### 1. Memory Efficiency
- LoRA for parameter-efficient fine-tuning
- Gradient accumulation for larger effective batch sizes
- Mixed precision training support

### 2. Speed Optimization
- ONNX runtime for CPU inference
- Quantization for reduced model size
- Caching of common responses

### 3. Quality Improvements
- Temperature-based sampling
- Beam search for better generation
- Length penalty for balanced outputs

## Error Handling

### 1. Input Validation
```python
def validate_input(response):
    if not response.strip():
        raise ValueError("Empty response")
    if len(response) > 1024:
        raise ValueError("Response too long")
```

### 2. Model Errors
```python
try:
    prompt = model.generate_prompt(response)
except Exception as e:
    logger.error(f"Generation failed: {str(e)}")
    return "Error generating prompt"
```

## Monitoring and Logging

### 1. Performance Metrics
- Response time
- Memory usage
- GPU utilization
- Error rates

### 2. Quality Metrics
- Prompt reconstruction accuracy
- Language detection accuracy
- Type classification accuracy

## Security Considerations

1. **Input Sanitization**
   - Remove malicious content
   - Limit input length
   - Validate language

2. **Resource Management**
   - Rate limiting
   - Memory usage caps
   - Timeout handling

## Deployment

### 1. Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU)
- 8GB RAM minimum

### 2. Environment Setup
```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python src/download_models.py
```

### 3. Running the Service
```bash
# Development
python src/app.py

# Production
gunicorn src.app:app
``` 