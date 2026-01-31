#!/usr/bin/env python3
"""
Test Script for N-gram Language Model

This script loads a trained model and generates text based on prompts.

USAGE
=====
    python test_model.py --model models/akan_model.pkl --prompt "your prompt here"
    python test_model.py --model models/akan_model.pkl  (interactive mode)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ngram_model import NGramLanguageModel


def test_model(model_path: str, prompt: str = None, num_samples: int = 5, 
               max_tokens: int = 30, temperature: float = 0.8):
    """
    Test a trained model by generating text.
    
    Args:
        model_path: Path to saved model file
        prompt: Optional starting prompt
        num_samples: Number of text samples to generate
        max_tokens: Maximum tokens to generate per sample
        temperature: Sampling temperature (lower = more deterministic)
    """
    print("=" * 80)
    print("TESTING N-GRAM LANGUAGE MODEL")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = NGramLanguageModel.load(model_path)
        print(f"Model loaded successfully!")
        print(f"  Order: {model.n}-gram")
        print(f"  Smoothing: {model.smoothing_name}")
        print(f"  Vocabulary size: {len(model.vocabulary):,}")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        print("\nMake sure:")
        print("  1. The model file exists")
        print("  2. The model was saved correctly using train.py")
        sys.exit(1)
    
    # Interactive mode if no prompt provided
    if prompt is None:
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("\nEnter prompts to generate text (or 'quit' to exit)")
        print("The model will continue from your prompt.\n")
        
        while True:
            try:
                user_prompt = input("Enter prompt (or 'quit'): ").strip()
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_prompt:
                    print("Please enter a prompt or 'quit' to exit.")
                    continue
                
                print(f"\nGenerating text from prompt: '{user_prompt}'")
                print("-" * 80)
                
                for i in range(num_samples):
                    generated = model.generate(
                        seed=user_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    print(f"  {i+1}. {generated}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
        return
    
    # Non-interactive mode with provided prompt
    print("\n" + "=" * 80)
    print("GENERATING TEXT FROM PROMPT")
    print("=" * 80)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {num_samples} samples...")
    print(f"Max tokens per sample: {max_tokens}")
    print(f"Temperature: {temperature}")
    print("\n" + "-" * 80)
    
    for i in range(num_samples):
        generated = model.generate(
            seed=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        print(f"\nSample {i+1}:")
        print(f"  {generated}")
    
    # Show top predictions for the prompt
    print("\n" + "=" * 80)
    print("TOP PREDICTIONS")
    print("=" * 80)
    
    # Extract context from prompt
    prompt_words = prompt.lower().split()
    context_length = model.n - 1
    
    if len(prompt_words) >= context_length:
        context = tuple(prompt_words[-context_length:])
    else:
        # Pad with start tokens if needed
        context = tuple(['<s>'] * (context_length - len(prompt_words)) + prompt_words)
    
    print(f"\nContext: {context}")
    print(f"Top 10 most likely next words:\n")
    
    predictions = model.get_top_predictions(context, k=10)
    for i, (word, prob) in enumerate(predictions, 1):
        print(f"  {i:2d}. {word:20s} {prob:.6f} ({prob*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained n-gram language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', '-m', type=str, 
                       default='models/akan_model.pkl',
                       help='Path to trained model file')
    parser.add_argument('--prompt', '-p', type=str, default=None,
                       help='Starting prompt for text generation (if not provided, interactive mode)')
    parser.add_argument('--num_samples', '-n', type=int, default=5,
                       help='Number of text samples to generate')
    parser.add_argument('--max_tokens', type=int, default=30,
                       help='Maximum tokens to generate per sample')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                       help='Sampling temperature (lower = more deterministic)')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
