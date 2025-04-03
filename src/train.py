import torch
from torch.utils.data import Dataset, DataLoader
import json
from model.rpe_model import RPEModel
from tqdm import tqdm
import logging
from transformers import get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptResponseDataset(Dataset):
    def __init__(self, data_path):
        self.pairs = []
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line == '.':  # Skip empty lines and single dots
                    i += 1
                    continue
                
                # Case 1: Response followed by tab and prompt
                if not line.startswith('\t') and '\t' in line:
                    parts = line.split('\t')
                    response = parts[0].strip()
                    prompt = parts[-1].strip()  # Take the last part as prompt
                    if response and prompt:
                        self.pairs.append((response, prompt))
                    i += 1
                    continue
                
                # Case 2: Tabbed prompt followed by response on next lines
                if line.startswith('\t'):
                    prompt = line[1:].strip()  # Remove the tab
                    response = ""
                    i += 1
                    # Collect response until next prompt or empty line
                    while i < len(lines):
                        line = lines[i].strip()
                        if not line or line == '.' or line.startswith('\t'):
                            break
                        response += line + " "
                        i += 1
                    if response.strip() and prompt:
                        self.pairs.append((response.strip(), prompt))
                    continue
                i += 1
        
        print(f"Loaded {len(self.pairs)} prompt-response pairs")
        if len(self.pairs) == 0:
            raise ValueError("No valid prompt-response pairs found in the data file!")
        
        # Print first few pairs for verification
        print("\nFirst few pairs:")
        for i, (response, prompt) in enumerate(self.pairs[:3]):
            print(f"\nPair {i + 1}:")
            print(f"Prompt: {prompt}")
            print(f"Response: {response[:100]}...")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def train_model(model, train_loader, num_epochs=5):
    """
    Enhanced training process with better optimization
    """
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Cosine schedule with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Track best model
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, batch in enumerate(progress_bar):
            # Forward pass
            loss = model.train_step(batch)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Save checkpoint every 100 batches
            if (i + 1) % 100 == 0:
                model.save_model("checkpoints/checkpoint_latest")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        # Save best model and check early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model("checkpoints/best_model")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Save epoch checkpoint
        model.save_model(f"checkpoints/checkpoint_epoch_{epoch + 1}")

def main():
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = RPEModel(device=device)
    
    # Load dataset with smaller batch size
    dataset = PromptResponseDataset("data/generated/english_pairs.txt")
    train_loader = DataLoader(
        dataset,
        batch_size=8,  # Reduced batch size
        shuffle=True,
        num_workers=0,  # No multiprocessing for stability
        pin_memory=True  # Better memory management
    )
    
    # Train model
    train_model(model, train_loader)
    
    # Load best model and save as final
    model.load_model("checkpoints/best_model")
    model.save_model("checkpoints/final_model")
    logger.info("Training completed and best model saved as final")

if __name__ == "__main__":
    main() 