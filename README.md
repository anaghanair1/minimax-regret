# MinimaxRegret

Minimize your maximum code regrets.

A transformer-based language model trained on programmer error messages
that predicts what will go wrong in code before execution, potentially
saving time and preventing debugging frustration.

This implementation is inspired by nanoGPT architecture (https://github.com/karpathy/nanoGPT)
but significantly simplified and adapted for code error prediction.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/minimax-regret.git
cd minimax-regret

# Install dependencies
pip install -r requirements.txt