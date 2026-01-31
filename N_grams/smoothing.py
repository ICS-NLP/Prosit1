"""
Smoothing Techniques for N-gram Language Models

This module implements various smoothing techniques to handle the
zero-probability problem in n-gram models.

THE ZERO PROBABILITY PROBLEM
============================
When an n-gram in test data was never seen in training:
    P(wₙ | wₙ₋₁) = C(wₙ₋₁, wₙ) / C(wₙ₋₁) = 0 / C(wₙ₋₁) = 0

This causes:
1. Sentence probability = 0 (one zero makes entire product zero)
2. Perplexity = infinity (log(0) is undefined)

SOLUTION: SMOOTHING
==================
Smoothing redistributes probability mass from seen n-grams to unseen ones,
ensuring all n-grams have non-zero probability.
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional, List
import math


class SmoothingMethod(ABC):
    """
    Abstract base class for smoothing methods.
    
    All smoothing methods must implement:
    - fit(): Learn parameters from training data
    - probability(): Return smoothed probability of n-gram
    """
    
    @abstractmethod
    def fit(self, ngram_counts: Counter, context_counts: Counter, 
            vocab_size: int, n: int) -> None:
        """Fit the smoothing method to training data."""
        pass
    
    @abstractmethod
    def probability(self, ngram: Tuple[str, ...], 
                   context: Tuple[str, ...]) -> float:
        """Return smoothed probability P(word | context)."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the smoothing method."""
        pass


class AddKSmoothing(SmoothingMethod):
    """
    Add-k (Laplace) Smoothing
    
    MATHEMATICAL FORMULATION
    ========================
    
    Add a small constant k to all counts:
    
    P_add-k(wₙ | wₙ₋₁) = [C(wₙ₋₁, wₙ) + k] / [C(wₙ₋₁) + k × V]
    
    Where:
    - C(wₙ₋₁, wₙ) = count of bigram (wₙ₋₁, wₙ)
    - C(wₙ₋₁) = count of context wₙ₋₁
    - k = smoothing parameter (typically 0 < k ≤ 1)
    - V = vocabulary size
    
    DERIVATION
    ==========
    
    We want probabilities to sum to 1:
    
    Σᵥ P(v | context) = 1
    
    Σᵥ [C(context, v) + k] / [C(context) + k × V]
    = [Σᵥ C(context, v) + k × V] / [C(context) + k × V]
    = [C(context) + k × V] / [C(context) + k × V]
    = 1 ✓
    
    SPECIAL CASES
    =============
    - k = 1: Laplace (add-one) smoothing
    - k = 0: No smoothing (MLE)
    - k < 1: Typically better performance (e.g., k = 0.01)
    
    EXAMPLE CALCULATION
    ===================
    Vocabulary: V = 1000
    C("the", "cat") = 10  (bigram count)
    C("the") = 100        (context count)
    k = 0.1
    
    P_add-k("cat" | "the") = (10 + 0.1) / (100 + 0.1 × 1000)
                           = 10.1 / 200
                           = 0.0505
    
    Compare to MLE:
    P_MLE("cat" | "the") = 10 / 100 = 0.10
    
    For unseen bigram ("the", "dog") with C("the", "dog") = 0:
    P_add-k("dog" | "the") = (0 + 0.1) / (100 + 0.1 × 1000)
                           = 0.1 / 200
                           = 0.0005 > 0 ✓
    """
    
    def __init__(self, k: float = 1.0):
        """
        Initialize add-k smoothing.
        
        Args:
            k: Smoothing parameter (default 1.0 = Laplace)
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k
        self.ngram_counts = None
        self.context_counts = None
        self.vocab_size = None
        self.n = None
        
    def fit(self, ngram_counts: Counter, context_counts: Counter,
            vocab_size: int, n: int) -> None:
        """
        Store counts and parameters for probability computation.
        
        Args:
            ngram_counts: Counter of n-gram occurrences
            context_counts: Counter of (n-1)-gram context occurrences
            vocab_size: Size of vocabulary V
            n: Order of n-gram model
        """
        self.ngram_counts = ngram_counts
        self.context_counts = context_counts
        self.vocab_size = vocab_size
        self.n = n
        
    def probability(self, ngram: Tuple[str, ...], 
                   context: Tuple[str, ...]) -> float:
        """
        Calculate add-k smoothed probability.
        
        P(wₙ | context) = [C(ngram) + k] / [C(context) + k × V]
        
        Args:
            ngram: Full n-gram tuple (context + word)
            context: Context tuple (n-1 previous words)
            
        Returns:
            Smoothed probability (always > 0)
        """
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        
        # Add-k formula
        numerator = ngram_count + self.k
        denominator = context_count + self.k * self.vocab_size
        
        # Handle edge case
        if denominator == 0:
            return 1.0 / self.vocab_size
        
        return numerator / denominator
    
    def get_name(self) -> str:
        return f"Add-{self.k} Smoothing"


class LinearInterpolation(SmoothingMethod):
    """
    Linear Interpolation Smoothing
    
    MATHEMATICAL FORMULATION
    ========================
    
    Combine probabilities from multiple n-gram orders:
    
    For trigram model:
    P_interp(wₙ | wₙ₋₂, wₙ₋₁) = λ₁ × P(wₙ) + λ₂ × P(wₙ | wₙ₋₁) + λ₃ × P(wₙ | wₙ₋₂, wₙ₋₁)
    
    Where:
    - λ₁ + λ₂ + λ₃ = 1 (lambdas must sum to 1)
    - λᵢ ≥ 0 (all lambdas non-negative)
    
    GENERAL FORMULA
    ===============
    For an n-gram model:
    
    P_interp(wₙ | w₁...wₙ₋₁) = Σᵢ₌₁ⁿ λᵢ × P_MLE(wₙ | wₙ₋ᵢ...wₙ₋₁)
    
    INTUITION
    =========
    - Higher-order n-grams capture more context but are sparse
    - Lower-order n-grams are more robust but less specific
    - Interpolation balances both
    
    EXAMPLE CALCULATION
    ===================
    
    Trigram model with λ₁ = 0.1, λ₂ = 0.3, λ₃ = 0.6
    
    Sentence: "the cat sat"
    Computing P("sat" | "the", "cat")
    
    P(sat) = 0.001                    (unigram)
    P(sat | cat) = 0.05               (bigram)
    P(sat | the, cat) = 0.15          (trigram)
    
    P_interp = 0.1 × 0.001 + 0.3 × 0.05 + 0.6 × 0.15
             = 0.0001 + 0.015 + 0.09
             = 0.1051
    
    If trigram ("the", "cat", "sat") was never seen:
    P(sat | the, cat) = 0
    
    P_interp = 0.1 × 0.001 + 0.3 × 0.05 + 0.6 × 0
             = 0.0001 + 0.015 + 0
             = 0.0151 > 0 ✓
    
    The lower-order probabilities "back up" the unseen trigram.
    """
    
    def __init__(self, lambdas: Optional[List[float]] = None):
        """
        Initialize linear interpolation.
        
        Args:
            lambdas: List of interpolation weights [λ₁, λ₂, ..., λₙ]
                    If None, uses equal weights
        """
        self.lambdas = lambdas
        self.ngram_counts_by_order = {}  # {order: Counter}
        self.context_counts_by_order = {}
        self.vocab_size = None
        self.n = None
        
    def fit(self, ngram_counts: Counter, context_counts: Counter,
            vocab_size: int, n: int) -> None:
        """
        Fit interpolation model.
        
        For interpolation, we need counts for all orders 1 to n.
        
        Args:
            ngram_counts: Counter of n-gram occurrences
            context_counts: Counter of (n-1)-gram context occurrences
            vocab_size: Size of vocabulary
            n: Maximum n-gram order
        """
        self.vocab_size = vocab_size
        self.n = n
        
        # Set default lambdas if not provided
        if self.lambdas is None:
            # Give more weight to higher orders
            self.lambdas = [1.0 / (2 ** (n - i)) for i in range(1, n + 1)]
            # Normalize to sum to 1
            total = sum(self.lambdas)
            self.lambdas = [l / total for l in self.lambdas]
        
        if len(self.lambdas) != n:
            raise ValueError(f"Need {n} lambdas, got {len(self.lambdas)}")
        
        if abs(sum(self.lambdas) - 1.0) > 1e-6:
            raise ValueError("Lambdas must sum to 1")
        
        # Store n-gram counts
        self.ngram_counts_by_order[n] = ngram_counts
        self.context_counts_by_order[n] = context_counts
        
        # Build lower-order counts from the n-gram counts
        self._build_lower_order_counts(ngram_counts)
        
        print(f"Interpolation weights: {self.lambdas}")
        
    def _build_lower_order_counts(self, ngram_counts: Counter) -> None:
        """Build counts for all n-gram orders from the highest order counts."""
        # For each n-gram, extract all sub-n-grams
        for order in range(1, self.n):
            self.ngram_counts_by_order[order] = Counter()
            self.context_counts_by_order[order] = Counter()
        
        # Count unigrams and build up
        for ngram, count in ngram_counts.items():
            # Extract all sub-n-grams
            for order in range(1, self.n):
                # Last 'order' tokens form the sub-n-gram
                sub_ngram = ngram[-(order):]  
                self.ngram_counts_by_order[order][sub_ngram] += count
                
                # Context is everything except last token
                if order > 1:
                    sub_context = sub_ngram[:-1]
                    self.context_counts_by_order[order][sub_context] += count
        
        # For unigrams, context count is total tokens
        total_tokens = sum(self.ngram_counts_by_order[1].values())
        self.context_counts_by_order[1] = Counter({(): total_tokens})
        
    def probability(self, ngram: Tuple[str, ...], 
                   context: Tuple[str, ...]) -> float:
        """
        Calculate interpolated probability.
        
        P_interp = Σᵢ λᵢ × P_MLE(order i)
        
        Args:
            ngram: Full n-gram tuple
            context: Context tuple
            
        Returns:
            Interpolated probability
        """
        prob = 0.0
        
        for order in range(1, self.n + 1):
            lambda_i = self.lambdas[order - 1]
            
            # Get the sub-n-gram of this order
            if order == 1:
                sub_ngram = (ngram[-1],)
                sub_context = ()
            else:
                sub_ngram = ngram[-order:]
                sub_context = sub_ngram[:-1]
            
            # MLE probability for this order
            ngram_count = self.ngram_counts_by_order[order].get(sub_ngram, 0)
            context_count = self.context_counts_by_order[order].get(sub_context, 0)
            
            if context_count > 0:
                mle_prob = ngram_count / context_count
            else:
                mle_prob = 1.0 / self.vocab_size
            
            prob += lambda_i * mle_prob
        
        return prob
    
    def get_name(self) -> str:
        return f"Linear Interpolation (λ={self.lambdas})"


class KneserNeySmoothing(SmoothingMethod):
    """
    Kneser-Ney Smoothing (State-of-the-Art)
    
    MATHEMATICAL FORMULATION
    ========================
    
    Kneser-Ney uses a sophisticated discounting and interpolation scheme:
    
    P_KN(wₙ | wₙ₋₁) = max(C(wₙ₋₁, wₙ) - d, 0) / C(wₙ₋₁) + λ(wₙ₋₁) × P_continuation(wₙ)
    
    Where:
    - d = discount (typically 0.75)
    - λ(wₙ₋₁) = normalizing constant
    - P_continuation = continuation probability
    
    THE KEY INSIGHT
    ===============
    
    Standard backoff: P(word) ∝ C(word)
    
    Problem: "Francisco" has high count but only appears after "San"
    
    Kneser-Ney: P_continuation(word) ∝ |{w : C(w, word) > 0}|
    
    This measures how many DIFFERENT contexts the word appears in,
    not just how frequent it is.
    
    CONTINUATION PROBABILITY
    ========================
    
    P_continuation(wₙ) = |{w : C(w, wₙ) > 0}| / |{(w₁, w₂) : C(w₁, w₂) > 0}|
    
    Numerator: Number of unique words that precede wₙ
    Denominator: Total number of unique bigrams
    
    EXAMPLE: Why Kneser-Ney is Better
    =================================
    
    Context: "I want to go to ___"
    
    Word frequencies:
    - "Francisco": 1000 occurrences (but all in "San Francisco")
    - "London": 100 occurrences (in many contexts)
    
    Standard smoothing would favor "Francisco" (higher frequency).
    
    But with continuation probability:
    - P_cont("Francisco") = 1 / N (only follows "San")
    - P_cont("London") = 50 / N (follows many words)
    
    Kneser-Ney correctly prefers "London"!
    
    DISCOUNT ESTIMATION
    ===================
    
    The optimal discount d can be estimated as:
    
    d = n₁ / (n₁ + 2n₂)
    
    Where:
    - n₁ = number of n-grams occurring exactly once
    - n₂ = number of n-grams occurring exactly twice
    
    This is based on Good-Turing estimation.
    """
    
    def __init__(self, discount: float = 0.75):
        """
        Initialize Kneser-Ney smoothing.
        
        Args:
            discount: Discount parameter d (typically 0.75)
        """
        if not 0 < discount < 1:
            raise ValueError("Discount must be between 0 and 1")
        self.discount = discount
        self.ngram_counts = None
        self.context_counts = None
        self.vocab_size = None
        self.n = None
        
        # For continuation probability
        self.continuation_counts = None  # |{w : C(w, word) > 0}|
        self.total_unique_bigrams = 0
        
        # For backoff weights
        self.lambda_cache = {}
        
    def fit(self, ngram_counts: Counter, context_counts: Counter,
            vocab_size: int, n: int) -> None:
        """
        Fit Kneser-Ney model.
        
        This computes:
        1. Standard n-gram counts
        2. Continuation counts for backoff
        3. Lambda normalizing constants
        
        Args:
            ngram_counts: Counter of n-gram occurrences
            context_counts: Counter of (n-1)-gram context occurrences
            vocab_size: Vocabulary size
            n: N-gram order
        """
        self.ngram_counts = ngram_counts
        self.context_counts = context_counts
        self.vocab_size = vocab_size
        self.n = n
        
        # Estimate discount using Good-Turing if we have the data
        self._estimate_discount(ngram_counts)
        
        # Compute continuation counts
        self._compute_continuation_counts(ngram_counts)
        
        print(f"Kneser-Ney fitted with discount d = {self.discount:.4f}")
        
    def _estimate_discount(self, ngram_counts: Counter) -> None:
        """
        Estimate optimal discount using Good-Turing estimation.
        
        d = n₁ / (n₁ + 2n₂)
        """
        count_of_counts = Counter(ngram_counts.values())
        n1 = count_of_counts.get(1, 0)
        n2 = count_of_counts.get(2, 0)
        
        if n1 > 0 and n2 > 0:
            estimated_d = n1 / (n1 + 2 * n2)
            # Use estimated discount if reasonable, otherwise keep default
            if 0.1 < estimated_d < 0.99:
                self.discount = estimated_d
                print(f"Estimated discount: {self.discount:.4f} (n₁={n1}, n₂={n2})")
        
    def _compute_continuation_counts(self, ngram_counts: Counter) -> None:
        """
        Compute continuation probability components.
        
        continuation_count[word] = |{context : C(context, word) > 0}|
        """
        # For each word, count unique preceding contexts
        self.continuation_counts = Counter()
        preceding_contexts = defaultdict(set)
        
        for ngram, count in ngram_counts.items():
            if count > 0 and len(ngram) >= 2:
                word = ngram[-1]
                context = ngram[:-1]
                preceding_contexts[word].add(context)
        
        # Count unique contexts per word
        for word, contexts in preceding_contexts.items():
            self.continuation_counts[word] = len(contexts)
        
        # Total unique bigrams (for normalization)
        self.total_unique_bigrams = len([c for c in ngram_counts.values() if c > 0])
        
    def _get_lambda(self, context: Tuple[str, ...]) -> float:
        """
        Compute the normalizing constant λ(context).
        
        λ(context) = (d / C(context)) × |{w : C(context, w) > 0}|
        
        This ensures probabilities sum to 1.
        """
        if context in self.lambda_cache:
            return self.lambda_cache[context]
        
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 1.0
        
        # Count unique words following this context
        unique_following = 0
        for ngram in self.ngram_counts:
            if ngram[:-1] == context and self.ngram_counts[ngram] > 0:
                unique_following += 1
        
        lambda_val = (self.discount * unique_following) / context_count
        self.lambda_cache[context] = lambda_val
        
        return lambda_val
    
    def _continuation_probability(self, word: str) -> float:
        """
        Compute continuation probability P_continuation(word).
        
        P_cont(word) = |{w : C(w, word) > 0}| / |{(w₁, w₂) : C(w₁, w₂) > 0}|
        """
        cont_count = self.continuation_counts.get(word, 0)
        
        if self.total_unique_bigrams == 0:
            return 1.0 / self.vocab_size
        
        # Add small smoothing to avoid zero for unseen words
        return (cont_count + 0.1) / (self.total_unique_bigrams + 0.1 * self.vocab_size)
    
    def probability(self, ngram: Tuple[str, ...], 
                   context: Tuple[str, ...]) -> float:
        """
        Calculate Kneser-Ney smoothed probability.
        
        P_KN(w | context) = max(C(context, w) - d, 0) / C(context) 
                         + λ(context) × P_continuation(w)
        
        Args:
            ngram: Full n-gram tuple
            context: Context tuple
            
        Returns:
            Kneser-Ney smoothed probability
        """
        word = ngram[-1]
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        
        if context_count == 0:
            # No context seen - use continuation probability only
            return self._continuation_probability(word)
        
        # First term: discounted probability
        discounted = max(ngram_count - self.discount, 0) / context_count
        
        # Second term: backoff with continuation
        lambda_val = self._get_lambda(context)
        continuation = self._continuation_probability(word)
        
        return discounted + lambda_val * continuation
    
    def get_name(self) -> str:
        return f"Kneser-Ney (d={self.discount:.2f})"


# Factory function to get smoothing method
def get_smoothing_method(method: str, **kwargs) -> SmoothingMethod:
    """
    Factory function to create smoothing methods.
    
    Args:
        method: One of 'add_k', 'laplace', 'interpolation', 'kneser_ney'
        **kwargs: Method-specific parameters
        
    Returns:
        SmoothingMethod instance
    """
    methods = {
        'add_k': lambda: AddKSmoothing(k=kwargs.get('k', 0.1)),
        'laplace': lambda: AddKSmoothing(k=1.0),
        'add_one': lambda: AddKSmoothing(k=1.0),
        'interpolation': lambda: LinearInterpolation(lambdas=kwargs.get('lambdas')),
        'kneser_ney': lambda: KneserNeySmoothing(discount=kwargs.get('discount', 0.75)),
    }
    
    method = method.lower().replace('-', '_')
    if method not in methods:
        raise ValueError(f"Unknown smoothing method: {method}. "
                        f"Choose from: {list(methods.keys())}")
    
    return methods[method]()


# Example and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("SMOOTHING MODULE - STANDALONE TEST")
    print("=" * 80)
    print("\nThis script demonstrates different smoothing techniques.")
    print("Smoothing is essential to handle unseen n-grams in test data.\n")
    
    # Create example counts to demonstrate smoothing
    print("Creating example n-gram counts for demonstration...")
    ngram_counts = Counter({
        ('the', 'cat'): 10,
        ('the', 'dog'): 5,
        ('cat', 'sat'): 8,
        ('dog', 'barked'): 3,
    })
    
    context_counts = Counter({
        ('the',): 15,
        ('cat',): 8,
        ('dog',): 3,
    })
    
    vocab_size = 100
    n = 2
    
    print(f"\nExample counts:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  N-gram order: {n} (bigram)")
    print(f"  Unique n-grams: {len(ngram_counts)}")
    print(f"  Unique contexts: {len(context_counts)}")
    
    # Test each smoothing method
    methods = ['laplace', 'add_k', 'kneser_ney']
    
    test_ngrams = [
        (('the', 'cat'), ('the',)),      # Seen bigram
        (('the', 'mouse'), ('the',)),    # Unseen bigram (zero count)
    ]
    
    print("\n" + "=" * 80)
    print("TESTING SMOOTHING METHODS")
    print("=" * 80)
    
    for method_name in methods:
        print(f"\n{'-' * 80}")
        print(f"Method: {method_name.upper()}")
        print(f"{'-' * 80}")
        
        method = get_smoothing_method(method_name, k=0.1)
        method.fit(ngram_counts, context_counts, vocab_size, n)
        
        print(f"\n{method.get_name()}")
        print(f"\nTesting probabilities:")
        
        for ngram, context in test_ngrams:
            prob = method.probability(ngram, context)
            count = ngram_counts.get(ngram, 0)
            status = "SEEN" if count > 0 else "UNSEEN"
            print(f"  P('{ngram[-1]}' | '{context[-1]}') = {prob:.6f}  [{status}, count={count}]")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATION")
    print("=" * 80)
    print("\nNotice that:")
    print("  - SEEN n-grams get higher probability")
    print("  - UNSEEN n-grams get non-zero probability (thanks to smoothing)")
    print("  - Without smoothing, unseen n-grams would have probability 0")
    print("  - This prevents infinite perplexity on test data")
    print("\n" + "=" * 80)
    print("SMOOTHING TEST COMPLETE")
    print("=" * 80)
