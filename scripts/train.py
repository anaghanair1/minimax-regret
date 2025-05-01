"""
Training script.

The training script approach is from:
- PyTorch model training examples (https://github.com/pytorch/examples)
- nanoGPT's training script (https://github.com/karpathy/nanoGPT/blob/master/train.py)
- Training loops from "Deep Learning with PyTorch" (Stevens et al.)
- Hyperparameter management techniques from ML engineering best practices
"""

import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
import time
from datetime import datetime

# parent directory 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MinimaxRegretModel
from src.dataset import CodeErrorDataset
from src.training import train_model, evaluate_model

def main():
    """Main entry point for the training script"""
    parser = argparse.ArgumentParser(description='MinimaxRegret - Model Training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium'], default='small', 
                        help='Model size (tiny: 2 layers, small: 3 layers, medium: 6 layers)')
    parser.add_argument('--output', type=str, default='minimax_regret_model.pt', 
                        help='Output file for trained model')
    parser.add_argument('--eval-split', type=float, default=0.1,
                        help='Fraction of data to use for evaluation (0-1)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # dataset
    dataset = CodeErrorDataset()
    
    # train and evaluation sets
    dataset_size = len(dataset)
    eval_size = int(dataset_size * args.eval_split)
    train_size = dataset_size - eval_size
    
    if eval_size > 0:
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size]
        )
    else:
        train_dataset = dataset
        eval_dataset = None
    
    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # model based on size
    if args.model_size == 'tiny':
        n_layer = 2
        n_embd = 64
        n_head = 2
    elif args.model_size == 'small':
        n_layer = 3
        n_embd = 128
        n_head = 4
    else:  # medium
        n_layer = 6
        n_embd = 256
        n_head = 8
    
    model = MinimaxRegretModel(
        vocab_size=dataset.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=512,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {model.get_parameter_count():,}")
    
    # output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    # metrics for plotting
    train_losses = []
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # one epoch
        train_loss = train_model(
            model, 
            train_dataset, 
            batch_size=args.batch_size, 
            epochs=1, 
            learning_rate=args.learning_rate
        )
        train_losses.append(train_loss)
        
        epoch_duration = time.time() - epoch_start_time
        
        # progress
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_duration:.2f}s. Loss: {train_loss:.4f}")
        
        # checkpoint
        checkpoint_path = f"{os.path.splitext(args.output)[0]}_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.2f}s.")
    
    # Evaluate 
    if eval_dataset:
        print("\nEvaluating model...")
        metrics = evaluate_model(model, eval_dataset, batch_size=args.batch_size)
        
        print(f"Evaluation Results:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Error Type Accuracy: {metrics['error_type_accuracy']:.2%}")
        print(f"Error Location Accuracy: {metrics['error_loc_accuracy']:.2%}")
        
        # Plotting metrics
        plt.figure(figsize=(12, 5))
        
        # training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # evaluation metrics
        plt.subplot(1, 2, 2)
        plt.bar(['Error Type Accuracy', 'Error Location Accuracy'], 
                [metrics['error_type_accuracy'], metrics['error_loc_accuracy']], 
                color=['#4CAF50', '#2196F3'])
        plt.title('Evaluation Metrics')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        
        # plot
        plt.tight_layout()
        plot_path = f"{os.path.splitext(args.output)[0]}_metrics.png"
        plt.savefig(plot_path)
        print(f"Metrics plot saved to {plot_path}")
    
    # final model
    torch.save(model.state_dict(), args.output)
    print(f"Final model saved to {args.output}")
    
    # summary
    print("\nTraining Summary:")
    print(f"Model Size: {args.model_size} ({n_layer} layers, {n_embd} embedding dim, {n_head} attention heads)")
    print(f"Parameters: {model.get_parameter_count():,}")
    print(f"Dataset Size: {dataset_size} examples")
    print(f"Training Duration: {total_duration:.2f}s")
    if eval_dataset:
        print(f"Final Error Type Accuracy: {metrics['error_type_accuracy']:.2%}")
        print(f"Final Error Location Accuracy: {metrics['error_loc_accuracy']:.2%}")

if __name__ == "__main__":
    main()