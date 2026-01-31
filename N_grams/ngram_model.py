"""

N-gram Language Model Implementation

This module implements a complete n-gram language model with support for
multiple smoothing techniques.

MATHEMATICAL FOUNDATION
=======================

An n-gram language model estimates the probability of a word given its context
using the Markov assumption:

    P(wᵢ | w₁, w₂, ..., wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁)

The probability of a sentence is:

    P(w₁, w₂, ..., wₘ) = ∏ᵢ₌₁ᵐ P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁)

TRAINING (Maximum Likelihood Estimation)
=========================================

P_MLE(wₙ | wₙ₋ₙ₊₁, ..., wₙ₋₁) = C(wₙ₋ₙ₊₁, ..., wₙ) / C(wₙ₋ₙ₊₁, ..., wₙ₋₁)

Where C(·) denotes count in the training corpus.
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Generator
import math
import pickle
import random

from smoothing import SmoothingMethod, get_smoothing_method, AddKSmoothing


class NGramLanguageModel:
    """
    N-gram Language Model
    
    Features:
    - Configurable n-gram order (1 = unigram, 2 = bigram, 3 = trigram, etc.)
    - Multiple smoothing techniques
    - Perplexity calculation
    - Text generation
    - Model persistence (save/load)
    
    USAGE EXAMPLE
    =============
    
    >>> model = NGramLanguageModel(n=3, smoothing='kneser_ney')
    >>> model.fit(train_sentences)
    >>> perplexity = model.perplexity(test_sentences)
    >>> generated = model.generate("The cat", max_tokens=10)
    """
    
    def __init__(self, 
                 n: int = 3,
                 smoothing: str = 'kneser_ney',
                 smoothing_params: Optional[Dict] = None):
        """
        Initialize the n-gram model.
        
        Args:
            n: Order of the n-gram model
               - n=1: Unigram P(w)
               - n=2: Bigram P(w | w₋₁)
               - n=3: Trigram P(w | w₋₂, w₋₁)
            smoothing: Smoothing method name
            smoothing_params: Parameters for smoothing method
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        self.n = n
        self.smoothing_name = smoothing
        self.smoothing_params = smoothing_params or {}
        
        # Will be set during training
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocabulary: set = set()
        self.smoothing_method: Optional[SmoothingMethod] = None
        self.is_fitted = False
        
        # Statistics
        self.total_tokens = 0
        self.total_sentences = 0
        
    def fit(self, sentences: List[List[str]]) -> 'NGramLanguageModel':
        """
        Train the n-gram model on tokenized sentences.
        
        Training Process:
        1. Build vocabulary from training data
        2. Count all n-grams
        3. Count all (n-1)-gram contexts
        4. Fit smoothing method
        
        Args:
            sentences: List of tokenized sentences
                      Each sentence is a list of tokens including <s> and </s>
                      
        Returns:
            self (fitted model)
            
        MATHEMATICAL DETAIL
        ===================
        
        For each sentence [<s>, w₁, w₂, ..., wₘ, </s>]:
        
        1. Pad with (n-1) start tokens: [<s>, <s>, w₁, w₂, ..., wₘ, </s>]
           (for trigram, need two <s> tokens to predict first word)
        
        2. Extract n-grams:
           - (<s>, <s>, w₁)  → counts to predict w₁
           - (<s>, w₁, w₂)   → counts to predict w₂
           - ...
           - (wₘ₋₁, wₘ, </s>) → counts to predict </s>
        
        3. For each n-gram, increment:
           - ngram_counts[(w₋ₙ₊₁, ..., w₀)] += 1
           - context_counts[(w₋ₙ₊₁, ..., w₋₁)] += 1
        """
        print(f"Training {self.n}-gram model...")
        print(f"Smoothing method: {self.smoothing_name}")
        
        # Reset counts
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()
        
        # Build vocabulary and counts
        for sentence in sentences:
            # Add to vocabulary
            self.vocabulary.update(sentence)
            
            # Pad sentence for n-gram extraction
            # Need (n-1) start tokens to predict first word
            padded = ['<s>'] * (self.n - 1) + sentence[1:]  # sentence already has one <s>
            
            # Extract n-grams and contexts
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                
                self.total_tokens += 1
        
        self.total_sentences = len(sentences)
        
        # Initialize and fit smoothing method
        self.smoothing_method = get_smoothing_method(
            self.smoothing_name, 
            **self.smoothing_params
        )
        self.smoothing_method.fit(
            self.ngram_counts,
            self.context_counts,
            len(self.vocabulary),
            self.n
        )
        
        self.is_fitted = True
        
        # Print training statistics
        print(f"\nTraining complete!")
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Total sentences: {self.total_sentences}")
        print(f"  Unique {self.n}-grams: {len(self.ngram_counts)}")
        
        return self
    
    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculate P(word | context) using smoothing.
        
        Args:
            word: The word to predict
            context: Tuple of previous (n-1) words
            
        Returns:
            Probability P(word | context)
            
        EXAMPLE
        =======
        For trigram model (n=3):
            P("sat" | ("the", "cat")) 
            
        The smoothing method handles unseen n-grams.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Build full n-gram
        ngram = context + (word,)
        
        # Get smoothed probability
        return self.smoothing_method.probability(ngram, context)
    
    def log_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculate log P(word | context).
        
        Using log probabilities prevents underflow for long sentences:
        log P(w₁...wₙ) = Σᵢ log P(wᵢ | context)
        
        Args:
            word: The word to predict
            context: Tuple of previous (n-1) words
            
        Returns:
            Log probability (base 2)
        """
        prob = self.probability(word, context)
        
        # Avoid log(0)
        if prob <= 0:
            return float('-inf')
        
        return math.log2(prob)
    
    def sentence_log_probability(self, sentence: List[str]) -> float:
        """
        Calculate log probability of a sentence.
        
        log P(w₁, w₂, ..., wₘ) = Σᵢ₌₁ᵐ log P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁)
        
        Args:
            sentence: Tokenized sentence with <s> and </s>
            
        Returns:
            Log probability of sentence (base 2)
            
        WORKED EXAMPLE
        ==============
        
        Sentence: ["<s>", "the", "cat", "sat", "</s>"]
        Trigram model (n=3)
        
        Padded: ["<s>", "<s>", "the", "cat", "sat", "</s>"]
        
        log P = log P(the | <s>, <s>) 
              + log P(cat | <s>, the)
              + log P(sat | the, cat)
              + log P(</s> | cat, sat)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Pad sentence
        padded = ['<s>'] * (self.n - 1) + sentence[1:]
        
        total_log_prob = 0.0
        
        # Calculate probability for each word
        for i in range(self.n - 1, len(padded)):
            word = padded[i]
            context = tuple(padded[i - self.n + 1:i])
            
            log_prob = self.log_probability(word, context)
            total_log_prob += log_prob
        
        return total_log_prob
    
    def sentence_probability(self, sentence: List[str]) -> float:
        """
        Calculate probability of a sentence.
        
        Note: For numerical stability, use sentence_log_probability() instead.
        
        Args:
            sentence: Tokenized sentence with <s> and </s>
            
        Returns:
            Probability of sentence
        """
        log_prob = self.sentence_log_probability(sentence)
        
        if log_prob == float('-inf'):
            return 0.0
        
        return 2 ** log_prob
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """
        Calculate perplexity on a set of sentences.
        
        MATHEMATICAL DEFINITION
        =======================
        
        Perplexity is the inverse probability normalized by the number of words:
        
        PP(W) = P(w₁, w₂, ..., wₙ)^(-1/N)
        
        Where N is the total number of tokens.
        
        Equivalently, using log probabilities:
        
        PP(W) = 2^(-1/N × Σᵢ log₂ P(wᵢ | context))
        
        INTERPRETATION
        ==============
        
        - Lower perplexity = better model
        - PP can be interpreted as the average branching factor
        - If PP = 100, the model is as uncertain as choosing uniformly 
          from 100 words at each position
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            Perplexity score
            
        WORKED EXAMPLE
        ==============
        
        Suppose we have:
        - 2 sentences with total 10 tokens
        - Sum of log probabilities = -50
        
        Cross-entropy H = -(-50) / 10 = 5 bits
        Perplexity = 2^5 = 32
        
        The model is as confused as randomly choosing from 32 words.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            # Pad sentence
            padded = ['<s>'] * (self.n - 1) + sentence[1:]
            
            # Count tokens to predict (excluding start padding)
            num_tokens = len(padded) - (self.n - 1)
            total_tokens += num_tokens
            
            # Add log probability
            log_prob = self.sentence_log_probability(sentence)
            
            if log_prob == float('-inf'):
                # Handle zero probability (shouldn't happen with smoothing)
                print(f"Warning: Zero probability sentence encountered")
                return float('inf')
            
            total_log_prob += log_prob
        
        # Cross-entropy (average negative log probability)
        cross_entropy = -total_log_prob / total_tokens
        
        # Perplexity = 2^(cross-entropy)
        perplexity = 2 ** cross_entropy
        
        return perplexity
    
    def generate(self, 
                 seed: Optional[str] = None, 
                 max_tokens: int = 50,
                 temperature: float = 1.0) -> str:
        """
        Generate text using the n-gram model.
        
        Process:
        1. Start with seed text (or <s> tokens)
        2. Sample next word according to P(word | context)
        3. Repeat until </s> or max_tokens reached
        
        Args:
            seed: Optional starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            
        Returns:
            Generated text string
            
        NOTE ON TEMPERATURE
        ===================
        
        Temperature scaling modifies the probability distribution:
        
        P'(w) = P(w)^(1/T) / Σᵥ P(v)^(1/T)
        
        - T → 0: Approaches greedy (argmax)
        - T = 1: Original distribution
        - T > 1: Flatter distribution (more random)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Initialize context
        if seed:
            # Tokenize seed (simple whitespace split)
            tokens = seed.lower().split()
            # Pad to n-1 tokens
            if len(tokens) < self.n - 1:
                context = ['<s>'] * (self.n - 1 - len(tokens)) + tokens
            else:
                context = tokens[-(self.n - 1):]
        else:
            context = ['<s>'] * (self.n - 1)
        
        generated = list(context)
        
        for _ in range(max_tokens):
            # Get context tuple
            ctx = tuple(generated[-(self.n - 1):])
            
            # Get probability distribution over vocabulary
            probs = {}
            for word in self.vocabulary:
                if word == '<s>':  # Don't generate start token
                    continue
                prob = self.probability(word, ctx)
                # Apply temperature
                if temperature != 1.0:
                    prob = prob ** (1.0 / temperature)
                probs[word] = prob
            
            # Normalize
            total = sum(probs.values())
            if total == 0:
                break
            probs = {w: p / total for w, p in probs.items()}
            
            # Sample next word
            words = list(probs.keys())
            weights = list(probs.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
            
            generated.append(next_word)
            
            # Stop if end token
            if next_word == '</s>':
                break
        
        # Remove special tokens and join
        result = [w for w in generated if w not in ['<s>', '</s>']]
        return ' '.join(result)
    
    def get_top_predictions(self, 
                           context: Tuple[str, ...], 
                           k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top k most likely next words given context.
        
        Args:
            context: Tuple of previous (n-1) words
            k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples, sorted by probability
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        
        for word in self.vocabulary:
            if word == '<s>':  # Skip start token
                continue
            prob = self.probability(word, context)
            predictions.append((word, prob))
        
        # Sort by probability descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:k]
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'smoothing_name': self.smoothing_name,
                'smoothing_params': self.smoothing_params,
                'ngram_counts': self.ngram_counts,
                'context_counts': self.context_counts,
                'vocabulary': self.vocabulary,
                'total_tokens': self.total_tokens,
                'total_sentences': self.total_sentences,
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NGramLanguageModel':
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            n=data['n'],
            smoothing=data['smoothing_name'],
            smoothing_params=data['smoothing_params']
        )
        model.ngram_counts = data['ngram_counts']
        model.context_counts = data['context_counts']
        model.vocabulary = data['vocabulary']
        model.total_tokens = data['total_tokens']
        model.total_sentences = data['total_sentences']
        
        # Refit smoothing method
        model.smoothing_method = get_smoothing_method(
            model.smoothing_name,
            **model.smoothing_params
        )
        model.smoothing_method.fit(
            model.ngram_counts,
            model.context_counts,
            len(model.vocabulary),
            model.n
        )
        model.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        return model
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"NGramLanguageModel(n={self.n}, "
                f"smoothing='{self.smoothing_name}', "
                f"status={status})")


# Example usage for testing
if __name__ == "__main__":
    print("=" * 80)
    print("N-GRAM MODEL MODULE - STANDALONE TEST")
    print("=" * 80)
    print("\nThis script demonstrates the n-gram language model functionality.")
    print("For full training, use train.py which loads the Akan dataset.\n")
    
    # Create minimal example data for demonstration
    print("Creating minimal example data for demonstration...")
    train_sentences = [
        ['<s>', 'the', 'cat', 'sat', 'on', 'the', 'mat', '</s>'],
        ['<s>', 'the', 'dog', 'sat', 'on', 'the', 'floor', '</s>'],
        ['<s>', 'a', 'cat', 'is', 'on', 'the', 'mat', '</s>'],
        ['<s>', 'the', 'cat', 'is', 'sleeping', '</s>'],
        ['<s>', 'the', 'dog', 'is', 'running', '</s>'],
    ]
    
    test_sentences = [
        ['<s>', 'the', 'cat', 'sat', '</s>'],
        ['<s>', 'a', 'dog', 'is', 'on', 'the', 'floor', '</s>'],
    ]
    
    print(f"  Training sentences: {len(train_sentences)}")
    print(f"  Test sentences: {len(test_sentences)}")
    
    print("\n" + "=" * 80)
    print("TRAINING BIGRAM MODEL")
    print("=" * 80)
    
    # Train model
    model = NGramLanguageModel(n=2, smoothing='add_k', smoothing_params={'k': 0.1})
    model.fit(train_sentences)
    
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    # Calculate perplexity
    train_ppl = model.perplexity(train_sentences)
    test_ppl = model.perplexity(test_sentences)
    
    print(f"\nPerplexity Results:")
    print(f"  Training set: {train_ppl:.2f}")
    print(f"  Test set: {test_ppl:.2f}")
    print(f"\nInterpretation:")
    print(f"  Lower perplexity = better model")
    print(f"  Test perplexity shows generalization ability")
    
    print("\n" + "=" * 80)
    print("PREDICTING NEXT WORDS")
    print("=" * 80)
    
    # Get predictions
    context = ('the',)
    print(f"\nTop 5 most likely words after context: {context}")
    predictions = model.get_top_predictions(context, k=5)
    for i, (word, prob) in enumerate(predictions, 1):
        print(f"  {i}. {word}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("TEXT GENERATION")
    print("=" * 80)
    
    # Generate text
    print("\nGenerating text samples:")
    for i in range(3):
        text = model.generate(seed="the cat", max_tokens=10, temperature=0.8)
        print(f"  {i+1}. {text}")
    
    print("\n" + "=" * 80)
    print("NOTE")
    print("=" * 80)
    print("\nThis is a minimal demonstration with example data.")
    print("For full training on Akan dataset, run: python train.py")
    print("=" * 80)
