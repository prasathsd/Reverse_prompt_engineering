import gradio as gr
from model.rpe_model import RPEModel
import torch
import os
import time
from typing import List, Tuple

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RPEModel(device=device)
model.load_model("checkpoints/final_model")

def predict_prompt(response: str, temperature: float = 0.7) -> str:
    """
    Generate prompt from response with error handling
    """
    if not response.strip():
        return "Please enter a response to analyze."
    
    try:
        progress = gr.Progress()
        progress(0, desc="Analyzing response...")
        prompt = model.generate_prompt(response, temperature=temperature)
        progress(1.0, desc="Analysis complete!")
        return prompt
    except Exception as e:
        return f"Error: {str(e)}"

def batch_predict(responses: List[str], temperature: float = 0.7) -> List[Tuple[str, str]]:
    """
    Process multiple responses at once
    """
    results = []
    for response in responses:
        if response.strip():
            try:
                prompt = model.generate_prompt(response, temperature=temperature)
                results.append((response, prompt))
            except Exception as e:
                results.append((response, f"Error: {str(e)}"))
    return results

# Create Gradio interface
with gr.Blocks(title="Reverse Prompt Engineering") as demo:
    gr.Markdown("# Reverse Prompt Engineering")
    gr.Markdown("Enter an AI response to reconstruct the original prompt that generated it.")
    
    with gr.Tab("Single Response"):
        with gr.Row():
            with gr.Column():
                response_input = gr.Textbox(
                    lines=5,
                    placeholder="Enter the AI response here...",
                    label="Response"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher values make the output more random, lower values make it more deterministic"
                )
                submit_btn = gr.Button("Generate Prompt")
            
            with gr.Column():
                output = gr.Textbox(
                    lines=5,
                    label="Reconstructed Prompt"
                )
        
        submit_btn.click(
            fn=predict_prompt,
            inputs=[response_input, temperature],
            outputs=output
        )
    
    with gr.Tab("Batch Processing"):
        with gr.Row():
            with gr.Column():
                batch_input = gr.Textbox(
                    lines=10,
                    placeholder="Enter multiple responses, one per line...",
                    label="Responses"
                )
                batch_temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                batch_submit = gr.Button("Process Batch")
            
            with gr.Column():
                batch_output = gr.Dataframe(
                    headers=["Response", "Reconstructed Prompt"],
                    datatype=["str", "str"],
                    col_count=(2, "fixed"),
                    label="Results"
                )
        
        batch_submit.click(
            fn=batch_predict,
            inputs=[batch_input, batch_temperature],
            outputs=batch_output
        )
    
    with gr.Tab("Examples"):
        gr.Examples(
            examples=[
                ["We need to create a new API endpoint that handles user authentication. The endpoint should validate user credentials and return a JWT token upon successful authentication.", 0.7],
                ["After reviewing the server logs, we found that the application's response time increases significantly during peak hours. The database queries are taking longer than expected.", 0.7],
                ["Let's design a modern e-commerce website with a clean, minimalist interface. We'll use a color scheme of navy blue and white, with smooth animations.", 0.7],
                ["The pull request needs several improvements. The error handling is incomplete, there are no unit tests, and the code doesn't follow the project's style guide.", 0.7],
                ["For our microservices architecture, we need to implement service discovery using Consul, load balancing with Nginx, and circuit breakers using Hystrix.", 0.7]
            ],
            inputs=[response_input, temperature],
            outputs=output,
            fn=predict_prompt,
            cache_examples=True
        )

if __name__ == "__main__":
    demo.launch() 