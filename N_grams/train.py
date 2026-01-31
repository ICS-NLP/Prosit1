#!/usr/bin/env python3
"""
Training Script for N-gram Language Model on Akan Dataset

This script trains an n-gram language model on the Akan transcription dataset.

USAGE
=====
    python train.py --order 3 --smoothing kneser_ney

    Arguments:
        --order: N-gram order (1, 2, 3, etc.)
        --smoothing: Smoothing method (add_k, kneser_ney, interpolation)
        --dataset_path: Path to Akan.xlsx file (default: dataset/Akan.xlsx)
        --output: Path to save trained model
        --compare: Compare different n-gram orders and save best model
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import TextPreprocessor, split_train_test, load_akan_dataset
from ngram_model import NGramLanguageModel
from evaluation import ModelEvaluator, compare_models


def train_model(args):
    """
    Main training function.
    
    Process:
    1. Load Akan dataset from Excel
    2. Preprocess text
    3. Split into train/test
    4. Train n-gram model
    5. Evaluate on test set
    6. Save model
    7. Optionally compare different orders and save best
    """
    print("=" * 80)
    print("TRAINING N-GRAM LANGUAGE MODEL FOR AKAN")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Load Akan transcriptions from Excel file")
    print("  2. Preprocess and tokenize the text")
    print("  3. Split data into training and test sets")
    print("  4. Train an n-gram language model")
    print("  5. Evaluate the model")
    print("  6. Save the trained model")
    if args.compare:
        print("  7. Compare different n-gram orders and save the best model")
    print("=" * 80)
    
    # Load Akan dataset
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AKAN DATASET")
    print("=" * 80)
    
    try:
        transcriptions = load_akan_dataset(args.dataset_path)
        print(f"\nSuccessfully loaded {len(transcriptions):,} transcriptions")
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        print("\nMake sure:")
        print("  1. The dataset file exists at the specified path")
        print("  2. The file has a 'Transcriptions' column")
        print("  3. pandas and openpyxl are installed: pip install pandas openpyxl")
        sys.exit(1)
    
    # Preprocess
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING TEXT")
    print("=" * 80)
    
    print(f"\nInitializing preprocessor with:")
    print(f"  Lowercase: {args.lowercase} (False preserves Akan word meanings)")
    print(f"  Minimum word frequency: {args.min_freq}")
    print(f"  Max vocabulary size: {args.max_vocab_size or 'Unlimited'}")
    
    preprocessor = TextPreprocessor(
        lowercase=args.lowercase,
        min_word_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size
    )
    
    # Convert transcriptions list to single text for preprocessing
    # Each transcription is treated as a separate document
    sentences = preprocessor.fit_transform(transcriptions)
    
    print(f"\nPreprocessing complete!")
    print(f"  Total tokenized sentences: {len(sentences):,}")
    
    # Split data
    print("\n" + "=" * 80)
    print("STEP 3: SPLITTING DATA")
    print("=" * 80)
    
    train_sentences, test_sentences = split_train_test(
        sentences, 
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Train model
    print("\n" + "=" * 80)
    print(f"STEP 4: TRAINING {args.order}-GRAM MODEL")
    print("=" * 80)
    
    print(f"\nModel configuration:")
    print(f"  N-gram order: {args.order}")
    print(f"  Smoothing method: {args.smoothing}")
    
    smoothing_params = {}
    if args.smoothing == 'add_k':
        smoothing_params['k'] = args.smoothing_k
        print(f"  Add-k parameter: {args.smoothing_k}")
    elif args.smoothing == 'kneser_ney':
        smoothing_params['discount'] = args.discount
        print(f"  Kneser-Ney discount: {args.discount}")
    
    model = NGramLanguageModel(
        n=args.order,
        smoothing=args.smoothing,
        smoothing_params=smoothing_params
    )
    
    print(f"\nTraining on {len(train_sentences):,} sentences...")
    model.fit(train_sentences)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATING MODEL")
    print("=" * 80)
    
    evaluator = ModelEvaluator(model)
    
    print("\nCalculating perplexity on training and test sets...")
    train_pp = evaluator.perplexity(train_sentences, verbose=False)
    test_pp = evaluator.perplexity(test_sentences, verbose=True)
    
    print(f"\nPerplexity Results:")
    print(f"  Training set: {train_pp:,.2f}")
    print(f"  Test set: {test_pp:,.2f}")
    
    # Calculate the ratio
    if train_pp > 0:
        ratio = test_pp / train_pp
        print(f"  Ratio (test/train): {ratio:.2f}x")
    
    print(f"\nInterpretation:")
    print(f"  Lower perplexity = better model")
    print(f"  Test perplexity measures generalization to unseen data")
    print(f"\n  IMPORTANT: Test perplexity > Training perplexity is NORMAL and EXPECTED!")
    print(f"  - The model has seen training data, so it predicts it better")
    print(f"  - Test data is unseen, so the model is less certain")
    print(f"  - This gap is normal for n-gram models (typically 3-10x)")
    if train_pp > 0 and ratio < 10:
        print(f"  - Your ratio of {ratio:.2f}x is reasonable for n-gram models")
    elif train_pp > 0:
        print(f"  - Your ratio of {ratio:.2f}x is high but acceptable")
    print(f"  - What matters: test perplexity is finite (not infinity)")
    
    # Detailed report
    print("\n" + evaluator.detailed_report(test_sentences))
    
    # Coverage analysis
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    
    coverage = evaluator.coverage_analysis(test_sentences)
    print(f"\nCoverage Statistics:")
    print(f"  Unique test n-grams: {coverage['unique_test_ngrams']:,}")
    print(f"  Seen n-grams: {coverage['seen_unique_ngrams']:,}")
    print(f"  Coverage (unique): {coverage['coverage_unique']*100:.1f}%")
    print(f"  Coverage (weighted): {coverage['coverage_weighted']*100:.1f}%")
    print(f"  OOV rate: {coverage['oov_rate']*100:.2f}%")
    print(f"\nInterpretation:")
    print(f"  Higher coverage = model has seen more test patterns in training")
    print(f"  Lower OOV rate = fewer unknown words in test data")
    
    # Save model
    if args.output:
        print("\n" + "=" * 80)
        print("STEP 6: SAVING MODEL")
        print("=" * 80)
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        model.save(args.output)
        print(f"\nModel saved to: {args.output}")
        print(f"You can load it later using: NGramLanguageModel.load('{args.output}')")
    
    # Generate sample text
    print("\n" + "=" * 80)
    print("STEP 7: TEXT GENERATION SAMPLES")
    print("=" * 80)
    
    print("\nGenerating sample text from the model:")
    for i in range(3):
        text = model.generate(max_tokens=20, temperature=0.8)
        print(f"  {i+1}. {text}")
    
    print("\nNote: Generated text may not be perfect, but should show")
    print("      that the model has learned some patterns from the data.")
    
    # Compare different n-gram orders if requested
    if args.compare:
        print("\n" + "=" * 80)
        print("STEP 8: COMPARING DIFFERENT N-GRAM ORDERS")
        print("=" * 80)
        
        print("\nTraining models with different n-gram orders...")
        print("This will help identify the best order for this dataset.\n")
        
        models = []
        names = []
        orders_to_test = [1, 2, 3] if args.order not in [1, 2, 3] else [args.order]
        
        if args.order not in [1, 2, 3]:
            orders_to_test = [1, 2, 3]
        else:
            # Test around the specified order
            orders_to_test = [max(1, args.order-1), args.order, args.order+1]
            orders_to_test = [n for n in orders_to_test if n >= 1 and n <= 5]
        
        for n in orders_to_test:
            print(f"\nTraining {n}-gram model...")
            m = NGramLanguageModel(n=n, smoothing=args.smoothing, smoothing_params=smoothing_params)
            m.fit(train_sentences)
            models.append(m)
            names.append(f"{n}-gram")
        
        print("\nEvaluating all models on test set...")
        results = compare_models(models, test_sentences, names)
        
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print(f"\n{'Model':<15} {'Perplexity':<15} {'Cross-Entropy':<15} {'Coverage':<15}")
        print("-" * 60)
        
        best_model = None
        best_ppl = float('inf')
        best_name = None
        
        for r in results:
            print(f"{r['name']:<15} {r['perplexity']:<15.2f} {r['cross_entropy']:<15.4f} {r['coverage']*100:<14.1f}%")
            if r['perplexity'] < best_ppl:
                best_ppl = r['perplexity']
                best_model = models[names.index(r['name'])]
                best_name = r['name']
        
        print("\n" + "=" * 80)
        print(f"BEST MODEL: {best_name.upper()} (Perplexity: {best_ppl:.2f})")
        print("=" * 80)
        
        # Save best model
        if args.output:
            best_output = args.output.replace('.pkl', f'_best_{best_name.replace("-", "")}.pkl')
            os.makedirs(os.path.dirname(best_output) or '.', exist_ok=True)
            best_model.save(best_output)
            print(f"\nBest model saved to: {best_output}")
        
        return best_model
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel trained successfully!")
    print(f"  Order: {args.order}-gram")
    print(f"  Smoothing: {args.smoothing}")
    print(f"  Test Perplexity: {test_pp:,.2f}")
    if args.output:
        print(f"  Saved to: {args.output}")
    print("\nYou can now use this model for text generation or further evaluation.")
    print("=" * 80)
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train an n-gram language model on Akan dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default='dataset/Akan.xlsx',
                       help='Path to Akan.xlsx file')
    
    # Model arguments
    parser.add_argument('--order', '-n', type=int, default=3,
                       help='N-gram order (1=unigram, 2=bigram, 3=trigram)')
    parser.add_argument('--smoothing', type=str, default='kneser_ney',
                       choices=['add_k', 'laplace', 'kneser_ney', 'interpolation'],
                       help='Smoothing method')
    
    # Smoothing parameters
    parser.add_argument('--smoothing_k', type=float, default=0.1,
                       help='k parameter for add-k smoothing')
    parser.add_argument('--discount', type=float, default=0.75,
                       help='Discount for Kneser-Ney smoothing')
    
    # Preprocessing arguments
    parser.add_argument('--lowercase', action='store_true', default=False,
                       help='Convert text to lowercase (default: False for Akan)')
    parser.add_argument('--min_freq', type=int, default=2,
                       help='Minimum word frequency for vocabulary')
    parser.add_argument('--max_vocab_size', type=int, default=None,
                       help='Maximum vocabulary size (None for unlimited)')
    
    # Training arguments
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Fraction of data for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str, 
                       default='models/akan_model.pkl',
                       help='Path to save trained model')
    
    # Analysis options
    parser.add_argument('--compare', action='store_true',
                       help='Compare different n-gram orders and save best model')
    
    args = parser.parse_args()
    
    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
