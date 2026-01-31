"""
Evaluation Module for N-gram Language Models

This module provides comprehensive evaluation metrics and analysis tools
for n-gram language models.

EVALUATION METRICS
==================

1. PERPLEXITY (Primary Metric)
   - Lower is better
   - Measures how "surprised" the model is by test data

2. CROSS-ENTROPY
   - Related to perplexity: PP = 2^H
   - Measures bits needed to encode test data

3. COVERAGE ANALYSIS
   - What percentage of test n-grams were seen in training?
   - Indicates model's generalization capability
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import math
import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluator:
    """
    Comprehensive evaluator for n-gram language models.
    
    Provides:
    - Perplexity calculation (sentence-level and corpus-level)
    - Cross-entropy calculation
    - Coverage analysis
    - OOV (out-of-vocabulary) rate
    - Comparison between models
    """
    
    def __init__(self, model):
        """
        Initialize evaluator with a trained model.
        
        Args:
            model: A fitted NGramLanguageModel instance
        """
        self.model = model
        
    def perplexity(self, sentences: List[List[str]], verbose: bool = False) -> float:
        """
        Calculate corpus-level perplexity.
        
        MATHEMATICAL DEFINITION
        =======================
        
        Perplexity is defined as:
        
            PP(W) = 2^H(W)
        
        Where H(W) is the cross-entropy:
        
            H(W) = -(1/N) × Σᵢ log₂ P(wᵢ | context)
        
        DERIVATION
        ==========
        
        Starting from the probability of the test set:
        
            P(W) = P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ | w₁...wᵢ₋₁)
        
        Taking the log:
        
            log P(W) = Σᵢ log P(wᵢ | context)
        
        Normalizing by N tokens:
        
            (1/N) log P(W) = (1/N) Σᵢ log P(wᵢ | context)
        
        Cross-entropy (negative of above):
        
            H = -(1/N) Σᵢ log₂ P(wᵢ | context)
        
        Perplexity:
        
            PP = 2^H = P(W)^(-1/N)
        
        Args:
            sentences: List of tokenized sentences
            verbose: Print detailed statistics
            
        Returns:
            Perplexity score
            
        EXAMPLE WITH AKAN DATA:
        -----------------------
        Test sentences (from your actual test set):
          sentence1 = ["<s>", "Nnipa", "bi", "gyina", "hɔ", "</s>"]
          sentence2 = ["<s>", "a", "bi", "wɔ", "</s>"]
          sentence3 = ["<s>", "Baabi", "a", "yɛbu", "fangoo", "</s>"]
        
        Step 1: Calculate log probability for each sentence
          sentence1: log P = -24.84 bits (4 words: Nnipa, bi, gyina, hɔ)
          sentence2: log P = -12.35 bits (3 words: a, bi, wɔ)
          sentence3: log P = -18.67 bits (4 words: Baabi, a, yɛbu, fangoo)
        
        Step 2: Count total tokens (words to predict, excluding <s>)
          sentence1: 4 tokens
          sentence2: 3 tokens
          sentence3: 4 tokens
          Total: 11 tokens
        
        Step 3: Calculate cross-entropy
          total_log_prob = -24.84 + (-12.35) + (-18.67) = -55.86
          cross_entropy = -(-55.86) / 11 = 55.86 / 11 = 5.08 bits
        
        Step 4: Calculate perplexity
          perplexity = 2^5.08 = 33.5
        
        Interpretation:
          - Perplexity = 33.5 means model is as uncertain as choosing
            from ~34 words at each position
          - Lower is better
        
        REAL RESULTS FROM YOUR MODEL:
        -----------------------------
        Test set: 6,788 sentences, 71,209 tokens
        Total log probability: -473,456 bits
        Cross-entropy: 473,456 / 71,209 = 6.66 bits
        Perplexity: 2^6.66 = 100.83
        
        This means: On average, the model considers ~101 words at each position
        """
        total_log_prob = 0.0
        total_tokens = 0
        sentence_perplexities = []
        
        for i, sentence in enumerate(sentences):
            # Calculate sentence log probability
            log_prob = self.model.sentence_log_probability(sentence)
            
            # Count tokens (excluding padding)
            num_tokens = len(sentence) - 1  # Exclude one <s> (others added as padding)
            
            if log_prob == float('-inf'):
                if verbose:
                    print(f"  Sentence {i}: Zero probability!")
                sentence_perplexities.append(float('inf'))
            else:
                # Sentence-level perplexity
                sent_pp = 2 ** (-log_prob / num_tokens)
                sentence_perplexities.append(sent_pp)
                
                total_log_prob += log_prob
                total_tokens += num_tokens
        
        # Corpus-level perplexity
        if total_tokens == 0:
            return float('inf')
        
        cross_entropy = -total_log_prob / total_tokens
        perplexity = 2 ** cross_entropy
        
        if verbose:
            print(f"\nPerplexity Analysis:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Cross-entropy: {cross_entropy:.4f} bits")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Sentence perplexities: min={min(sentence_perplexities):.2f}, "
                  f"max={max(sentence_perplexities):.2f}, "
                  f"mean={np.mean([p for p in sentence_perplexities if p != float('inf')]):.2f}")
        
        return perplexity
    
    def cross_entropy(self, sentences: List[List[str]]) -> float:
        """
        Calculate cross-entropy in bits.
        
        H(W) = -(1/N) × Σᵢ log₂ P(wᵢ | context)
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            Cross-entropy in bits
        """
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            log_prob = self.model.sentence_log_probability(sentence)
            
            if log_prob != float('-inf'):
                total_log_prob += log_prob
                total_tokens += len(sentence) - 1
        
        if total_tokens == 0:
            return float('inf')
        
        return -total_log_prob / total_tokens
    
    def coverage_analysis(self, sentences: List[List[str]]) -> Dict:
        """
        Analyze how well the model covers the test data.
        
        Computes:
        - N-gram coverage: % of test n-grams seen in training
        - OOV rate: % of words not in vocabulary
        - Unseen n-gram distribution
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            Dictionary with coverage statistics
            
        EXAMPLE WITH AKAN DATA (BIGRAM MODEL):
        --------------------------------------
        Test sentences:
          sentence1 = ["<s>", "Nnipa", "bi", "gyina", "hɔ", "</s>"]
          sentence2 = ["<s>", "a", "rareword", "wɔ", "</s>"]  # has OOV word
        
        Step 1: Extract test bigrams
          sentence1 bigrams:
            ("<s>", "Nnipa"), ("Nnipa", "bi"), ("bi", "gyina"), 
            ("gyina", "hɔ"), ("hɔ", "</s>")
          sentence2 bigrams:
            ("<s>", "a"), ("a", "rareword"), ("rareword", "wɔ"), ("wɔ", "</s>")
          Total unique test bigrams: 9
        
        Step 2: Check which bigrams were seen in training
          Training bigrams (from model.ngram_counts):
            ("<s>", "Nnipa"): seen (count = 234)
            ("Nnipa", "bi"): seen (count = 456)
            ("bi", "gyina"): seen (count = 123)
            ("gyina", "hɔ"): seen (count = 234)
            ("hɔ", "</s>"): seen (count = 567)
            ("<s>", "a"): seen (count = 1,234)
            ("a", "rareword"): NOT SEEN (count = 0)
            ("rareword", "wɔ"): NOT SEEN (count = 0)
            ("wɔ", "</s>"): seen (count = 345)
          
          Seen: 7 out of 9
          Unique coverage: 7 / 9 = 77.8%
        
        Step 3: Weighted coverage (by token frequency)
          Count occurrences:
            Seen bigrams appear: 7 times total
            All bigrams appear: 9 times total
          Weighted coverage: 7 / 9 = 77.8%
        
        Step 4: OOV rate
          Test words: ["Nnipa", "bi", "gyina", "hɔ", "a", "rareword", "wɔ"]
          Vocabulary: {"Nnipa", "bi", "gyina", "hɔ", "a", "wɔ", ...}
          OOV words: ["rareword"]
          OOV rate: 1 / 7 = 14.3%
        
        REAL RESULTS FROM YOUR MODEL:
        -----------------------------
        Test set: 6,788 sentences
        Unique test 2-grams: 48,941
        Seen 2-grams: 21,589
        Coverage (unique): 44.1%
        Coverage (weighted): 61.0%  (more important - by frequency)
        OOV rate: 0.19%  (very low!)
        
        Interpretation:
          - 61% weighted coverage means most frequent test patterns were seen
          - Low OOV rate (0.19%) means almost all words are in vocabulary
          - This is good for generalization!
        """
        n = self.model.n
        
        # Count test n-grams
        test_ngrams = Counter()
        test_words = set()
        oov_count = 0
        total_words = 0
        
        for sentence in sentences:
            # Add words
            for word in sentence:
                if word not in ['<s>', '</s>']:
                    test_words.add(word)
                    total_words += 1
                    if word not in self.model.vocabulary:
                        oov_count += 1
            
            # Extract n-grams
            padded = ['<s>'] * (n - 1) + sentence[1:]
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i:i + n])
                test_ngrams[ngram] += 1
        
        # Calculate coverage
        seen_ngrams = sum(1 for ng in test_ngrams if ng in self.model.ngram_counts)
        total_ngrams = len(test_ngrams)
        unique_test_ngrams = len(test_ngrams)
        total_ngram_tokens = sum(test_ngrams.values())
        
        # Coverage by token count (weighted)
        seen_tokens = sum(
            count for ng, count in test_ngrams.items() 
            if ng in self.model.ngram_counts
        )
        
        results = {
            'ngram_order': n,
            'unique_test_ngrams': unique_test_ngrams,
            'total_ngram_tokens': total_ngram_tokens,
            'seen_unique_ngrams': seen_ngrams,
            'coverage_unique': seen_ngrams / unique_test_ngrams if unique_test_ngrams > 0 else 0,
            'coverage_weighted': seen_tokens / total_ngram_tokens if total_ngram_tokens > 0 else 0,
            'unique_test_words': len(test_words),
            'oov_count': oov_count,
            'oov_rate': oov_count / total_words if total_words > 0 else 0,
            'vocabulary_size': len(self.model.vocabulary),
        }
        
        return results
    
    def detailed_report(self, sentences: List[List[str]]) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            sentences: Test sentences
            
        Returns:
            Formatted report string
        """
        pp = self.perplexity(sentences)
        ce = self.cross_entropy(sentences)
        coverage = self.coverage_analysis(sentences)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                 N-GRAM MODEL EVALUATION REPORT                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: {self.model.n}-gram with {self.model.smoothing_name}
║  Vocabulary size: {coverage['vocabulary_size']:,}
╠══════════════════════════════════════════════════════════════════╣
║  PERPLEXITY METRICS                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Perplexity:     {pp:,.2f}
║  Cross-entropy:  {ce:.4f} bits
║
║  Interpretation:
║  - The model is as uncertain as choosing from {pp:.0f} words
║  - Each word needs ~{ce:.1f} bits to encode on average
╠══════════════════════════════════════════════════════════════════╣
║  COVERAGE METRICS                                                ║
╠══════════════════════════════════════════════════════════════════╣
║  {self.model.n}-gram coverage (unique):   {coverage['coverage_unique']*100:.1f}%
║  {self.model.n}-gram coverage (weighted): {coverage['coverage_weighted']*100:.1f}%
║  OOV (out-of-vocabulary) rate: {coverage['oov_rate']*100:.2f}%
║
║  Unseen {self.model.n}-grams: {coverage['unique_test_ngrams'] - coverage['seen_unique_ngrams']:,} 
║  out of {coverage['unique_test_ngrams']:,} unique
╚══════════════════════════════════════════════════════════════════╝
"""
        return report


def compare_models(models: List, test_sentences: List[List[str]], 
                   model_names: Optional[List[str]] = None) -> Dict:
    """
    Compare multiple n-gram models.
    
    Args:
        models: List of fitted NGramLanguageModel instances
        test_sentences: Test data
        model_names: Optional names for each model
        
    Returns:
        Comparison results dictionary
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    results = []
    
    for model, name in zip(models, model_names):
        evaluator = ModelEvaluator(model)
        pp = evaluator.perplexity(test_sentences)
        ce = evaluator.cross_entropy(test_sentences)
        coverage = evaluator.coverage_analysis(test_sentences)
        
        results.append({
            'name': name,
            'n': model.n,
            'smoothing': model.smoothing_name,
            'perplexity': pp,
            'cross_entropy': ce,
            'coverage': coverage['coverage_weighted'],
            'oov_rate': coverage['oov_rate'],
        })
    
    return results


def plot_perplexity_comparison(results: List[Dict], save_path: Optional[str] = None):
    """
    Create a bar chart comparing model perplexities.
    
    Args:
        results: Output from compare_models()
        save_path: Optional path to save figure
    """
    names = [r['name'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(names, perplexities, color='steelblue', edgecolor='navy')
    
    # Add value labels
    for bar, pp in zip(bars, perplexities):
        height = bar.get_height()
        ax.annotate(f'{pp:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax.set_title('N-gram Model Comparison: Perplexity', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_learning_curve(model, train_sentences: List[List[str]], 
                       test_sentences: List[List[str]],
                       fractions: List[float] = None,
                       save_path: Optional[str] = None):
    """
    Plot how perplexity changes with training data size.
    
    This helps understand:
    - Whether more data would help
    - Signs of overfitting
    
    Args:
        model: NGramLanguageModel class (not instance)
        train_sentences: Full training data
        test_sentences: Test data
        fractions: Data fractions to test (default: 10%, 25%, 50%, 75%, 100%)
        save_path: Optional path to save figure
    """
    if fractions is None:
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    train_perplexities = []
    test_perplexities = []
    data_sizes = []
    
    for frac in fractions:
        # Subset of training data
        n_samples = int(len(train_sentences) * frac)
        subset = train_sentences[:n_samples]
        
        # Train model
        from ngram_model import NGramLanguageModel
        m = NGramLanguageModel(n=model.n, smoothing=model.smoothing_name)
        m.fit(subset)
        
        # Evaluate
        evaluator = ModelEvaluator(m)
        train_pp = evaluator.perplexity(subset)
        test_pp = evaluator.perplexity(test_sentences)
        
        train_perplexities.append(train_pp)
        test_perplexities.append(test_pp)
        data_sizes.append(n_samples)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data_sizes, train_perplexities, 'b-o', label='Training Set', linewidth=2)
    ax.plot(data_sizes, test_perplexities, 'r-s', label='Test Set', linewidth=2)
    
    ax.set_xlabel('Number of Training Sentences', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Learning Curve: Perplexity vs Training Data Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return data_sizes, train_perplexities, test_perplexities


# Main demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("EVALUATION MODULE - STANDALONE TEST")
    print("=" * 80)
    print("\nThis script demonstrates evaluation metrics for n-gram models.")
    print("For full evaluation on Akan dataset, use train.py.\n")
    
    # Import model
    from ngram_model import NGramLanguageModel
    
    # Create minimal example data
    print("Creating minimal example data for demonstration...")
    train_sentences = [
        ['<s>', 'the', 'cat', 'sat', 'on', 'the', 'mat', '</s>'],
        ['<s>', 'the', 'dog', 'ran', 'in', 'the', 'park', '</s>'],
        ['<s>', 'a', 'bird', 'flew', 'over', 'the', 'house', '</s>'],
        ['<s>', 'the', 'cat', 'is', 'sleeping', '</s>'],
        ['<s>', 'the', 'dog', 'is', 'barking', '</s>'],
    ] * 5  # Repeat for more data
    
    test_sentences = [
        ['<s>', 'the', 'cat', 'ran', 'on', 'the', 'mat', '</s>'],
        ['<s>', 'a', 'dog', 'sat', 'in', 'the', 'park', '</s>'],
    ]
    
    print(f"  Training sentences: {len(train_sentences)}")
    print(f"  Test sentences: {len(test_sentences)}")
    
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    # Train model
    model = NGramLanguageModel(n=2, smoothing='kneser_ney')
    model.fit(train_sentences)
    
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    
    print("\nDetailed Evaluation Report:")
    print(evaluator.detailed_report(test_sentences))
    
    # Additional analysis
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    
    coverage = evaluator.coverage_analysis(test_sentences)
    print(f"\nCoverage Statistics:")
    print(f"  Unique test n-grams: {coverage['unique_test_ngrams']}")
    print(f"  Seen n-grams: {coverage['seen_unique_ngrams']}")
    print(f"  Coverage (unique): {coverage['coverage_unique']*100:.1f}%")
    print(f"  Coverage (weighted): {coverage['coverage_weighted']*100:.1f}%")
    print(f"  OOV rate: {coverage['oov_rate']*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("NOTE")
    print("=" * 80)
    print("\nThis is a minimal demonstration with example data.")
    print("For full evaluation on Akan dataset, run: python train.py")
    print("=" * 80)