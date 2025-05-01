"""
Training functions for the MinimaxRegret model.

The training implementation is from:
- PyTorch's training examples (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- nanoGPT's training loop (https://github.com/karpathy/nanoGPT/blob/master/train.py)
- Multi-task learning approaches from "An Overview of Multi-Task Learning in Deep Neural Networks" 
  (Ruder, 2017)

The evaluation metrics and loss functions are from standard approaches in 
machine learning literature.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict

def train_model(model, dataset, batch_size=4, epochs=1, learning_rate=3e-4):
    """Train the model for a small number of epochs"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(x, targets)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    print("Training completed!")

def evaluate_model(model, dataset, batch_size=4):
    """Evaluate the model on the dataset"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    total_loss = 0
    error_type_correct = 0
    error_loc_correct = 0
    
    with torch.no_grad():
        for x, targets in dataloader:
            # Forward pass
            outputs = model(x, targets)
            loss = outputs['loss']
            
            # Track metrics
            total_loss += loss.item()
            
            # Error type accuracy
            error_type_logits = outputs['error_type_logits']
            error_type_targets = targets[0]
            predicted = torch.argmax(error_type_logits, dim=1)
            error_type_correct += (predicted == error_type_targets).sum().item()
            
            # Error location accuracy 
            error_loc_logits = outputs['error_loc_logits']
            error_loc_targets = targets[1]
            predicted_loc = torch.argmax(error_loc_logits, dim=2)
            target_loc = torch.argmax(error_loc_targets, dim=1)
            for i in range(len(predicted_loc)):
                if predicted_loc[i][target_loc[i]] == 1:
                    error_loc_correct += 1
    
    avg_loss = total_loss / len(dataloader)
    error_type_accuracy = error_type_correct / len(dataset)
    error_loc_accuracy = error_loc_correct / len(dataset)
    
    return {
        'loss': avg_loss,
        'error_type_accuracy': error_type_accuracy,
        'error_loc_accuracy': error_loc_accuracy
    }