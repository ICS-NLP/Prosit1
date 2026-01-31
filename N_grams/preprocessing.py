"""
Text Preprocessing Module for N-gram Language Models

This module provides text preprocessing utilities specifically designed
for low-resource African languages, particularly Akan.

Mathematical Foundation:
-----------------------
Preprocessing transforms raw text T into a sequence of tokens:
    T → [w₁, w₂, ..., wₙ]

The quality of preprocessing directly impacts model performance.
"""

import re
import string
import sys
from typing import List, Tuple, Optional
from collections import Counter
import unicodedata


class TextPreprocessor:
    """
    Preprocessor for preparing text data for n-gram modeling.
    
    Handles:
    - Sentence tokenization
    - Word tokenization
    - Lowercasing (optional for Akan)
    - Special token insertion (<s>, </s>, <UNK>)
    - Vocabulary building with frequency thresholds
    - Akan-specific text cleaning
    """
    
    # Special tokens
    START_TOKEN = "<s>"      # Beginning of sentence
    END_TOKEN = "</s>"       # End of sentence
    UNK_TOKEN = "<UNK>"      # Unknown word
    
    def __init__(self, 
                 lowercase: bool = False,  # Default False for Akan (preserves meaning)
                 min_word_freq: int = 2,
                 max_vocab_size: Optional[int] = None):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
                      For Akan, keep False to preserve word meanings
            min_word_freq: Minimum frequency for a word to be in vocabulary
            max_vocab_size: Maximum vocabulary size (None for unlimited)
        """
        self.lowercase = lowercase
        self.min_word_freq = min_word_freq
        self.max_vocab_size = max_vocab_size
        self.vocabulary = set()
        self.word_counts = Counter()
        self.is_fitted = False
        
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters for consistent processing.
        
        Many African languages use special characters that may have
        multiple Unicode representations. Akan uses special characters
        like ɛ, ɔ, ɲ, etc.
        
        Args:
            text: Input text
            
        Returns:
            Unicode-normalized text
        """
        # NFC normalization: compose characters
        return unicodedata.normalize('NFC', text)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters while preserving Akan characters.
        
        Process:
        1. Normalize Unicode (important for Akan special characters)
        2. Optionally lowercase
        3. Remove URLs and emails
        4. Handle excessive whitespace
        5. Preserve Akan-specific punctuation and characters
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Normalize Unicode first (critical for Akan)
        text = self._normalize_unicode(text)
        
        # Lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        # For now, we'll remove them to focus on text
        text = re.sub(r'\d+', '', text)
        
        # Preserve Akan characters (letters, including special chars like ɛ, ɔ)
        # Keep basic punctuation: period, comma, question mark, exclamation
        # Remove other special characters but preserve Akan letters
        text = re.sub(r'[^\w\s\.\?\!\,\'\-]', ' ', text)
        
        # Normalize excessive whitespace (2+ spaces → 1 space)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using a period (.), question mark (?)
        or exclamation(!).
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Clean the text first
        text = self._clean_text(text)
        
        # If text is empty after cleaning, return empty list
        if not text.strip():
            return []
        
        # Split on sentence-ending punctuation
        # Pattern: split after . ! or ? followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def tokenize_words(self, sentence: str) -> List[str]:
        """
        Split a sentence into words (tokens).
        Preserving Akan-specific characters.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of word tokens
        """
        
        # Remove punctuation (but preserve Akan letters)
        sentence = sentence.translate(
            str.maketrans('', '', string.punctuation)
        )
        
        # Split on whitespace
        words = sentence.split()
        
        # Filter empty strings
        words = [w for w in words if w]
        
        return words
    
    def fit(self, texts: List[str]) -> 'TextPreprocessor':
        """
        Build vocabulary from training texts.
        
        This method:
        1. Counts word frequencies across all texts
        2. Filters by minimum frequency
        3. Limits vocabulary size if specified
        
        Mathematical Note:
        -----------------
        Building a vocabulary V from corpus C:
        V = {w : count(w, C) >= min_freq}
        
        Words not in V are mapped to <UNK>:
        P(<UNK>) = Σ count(w) for all w not in V
        
        Args:
            texts: List of text documents
            
        Returns:
            self (fitted preprocessor)
        """
        print("=" * 80)
        print("STEP 1: BUILDING VOCABULARY FROM TRAINING DATA")
        print("=" * 80)
        
        # Count all words
        self.word_counts = Counter()
        total_sentences_processed = 0
        
        print("\nProcessing texts to count word frequencies...")
        for text_idx, text in enumerate(texts):
            sentences = self.tokenize_sentences(text)
            total_sentences_processed += len(sentences)
            
            for sentence in sentences:
                words = self.tokenize_words(sentence)
                self.word_counts.update(words)
            
            if (text_idx + 1) % 1000 == 0:
                print(f"  Processed {text_idx + 1}/{len(texts)} texts...")
        
        print(f"\nVocabulary building statistics:")
        print(f"  Total texts processed: {len(texts)}")
        print(f"  Total sentences extracted: {total_sentences_processed}")
        if len(texts) > 0:
            avg_sentences = total_sentences_processed / len(texts)
            print(f"  Average sentences per text: {avg_sentences:.1f}")
            print(f"  (Note: Each transcription may contain multiple sentences)")
        print(f"  Unique words found: {len(self.word_counts)}")
        print(f"  Total word tokens: {sum(self.word_counts.values()):,}")
        
        # Filter by minimum frequency
        filtered_words = {
            word for word, count in self.word_counts.items()
            if count >= self.min_word_freq
        }
        
        print(f"\nFiltering vocabulary:")
        print(f"  Minimum frequency threshold: {self.min_word_freq}")
        print(f"  Words meeting threshold: {len(filtered_words)}")
        print(f"  Words below threshold: {len(self.word_counts) - len(filtered_words)}")
        
        # Limit vocabulary size if specified
        if self.max_vocab_size is not None:
            # Keep most frequent words
            most_common = self.word_counts.most_common(self.max_vocab_size)
            filtered_words = {word for word, _ in most_common}
            print(f"  Limited to top {self.max_vocab_size} most frequent words")
        
        # Add special tokens
        self.vocabulary = filtered_words | {
            self.START_TOKEN, 
            self.END_TOKEN, 
            self.UNK_TOKEN
        }
        
        self.is_fitted = True
        
        print(f"\nFinal vocabulary size: {len(self.vocabulary)} words")
        print(f"  - Regular words: {len(filtered_words)}")
        print(f"  - Special tokens: 3 (<s>, </s>, <UNK>)")
        
        # Show most frequent words
        print(f"\nTop 20 most frequent words:")
        for word, count in self.word_counts.most_common(20):
            in_vocab = "✓" if word in filtered_words else "✗"
            print(f"  {in_vocab} {word}: {count:,} occurrences")
        
        print("=" * 80)
        
        return self
    
    def transform(self, text: str) -> List[List[str]]:
        """
        Transform text into tokenized sentences with special tokens.
        
        Process:
        1. Split into sentences
        2. Tokenize each sentence
        3. Replace OOV words with <UNK>
        4. Add <s> and </s> markers
        
        Args:
            text: Input text
            
        Returns:
            List of tokenized sentences
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        tokenized_sentences = []
        
        for sentence in self.tokenize_sentences(text):
            words = self.tokenize_words(sentence)
            
            # Replace OOV words with <UNK>
            words = [
                w if w in self.vocabulary else self.UNK_TOKEN
                for w in words
            ]
            
            # Add sentence boundary markers
            if words:  # Only add non-empty sentences
                words = [self.START_TOKEN] + words + [self.END_TOKEN]
                tokenized_sentences.append(words)
        
        return tokenized_sentences
    
    def fit_transform(self, texts: List[str]) -> List[List[str]]:
        """
        Fit vocabulary and transform texts in one step.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of all tokenized sentences from all texts
        """
        self.fit(texts)
        
        print("\n" + "=" * 80)
        print("STEP 2: TRANSFORMING TEXTS TO TOKENIZED SENTENCES")
        print("=" * 80)
        print("\nConverting texts to tokenized sentences...")
        
        all_sentences = []
        for text_idx, text in enumerate(texts):
            sentences = self.transform(text)
            all_sentences.extend(sentences)
            
            if (text_idx + 1) % 1000 == 0:
                print(f"  Processed {text_idx + 1}/{len(texts)} texts...")
        
        print(f"\nTransformation complete!")
        print(f"  Total tokenized sentences: {len(all_sentences)}")
        print(f"  Average sentence length: {sum(len(s) for s in all_sentences) / len(all_sentences):.1f} tokens")
        print("=" * 80)
        
        return all_sentences
    
    def get_vocabulary(self) -> set:
        """Return the vocabulary set."""
        return self.vocabulary.copy()
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocabulary)
    
    def get_word_frequency(self, word: str) -> int:
        """Get the frequency of a word in training data."""
        return self.word_counts.get(word, 0)


def prepare_ngram_data(sentences: List[List[str]], n: int) -> List[Tuple]:
    """
    Prepare n-gram training data from tokenized sentences.
    
    For n-gram modeling, we need (context, target) pairs where:
    - context = (w_i-n+1, ..., w_i-1)  [n-1 previous words]
    - target = w_i                      [current word to predict]
    
    Mathematical Definition:
    ----------------------
    An n-gram is a sequence of n consecutive tokens.
    For a sentence [w_1, w_2, ..., w_m], the n-grams are:
        (w_1, w_2, ..., w_n)
        (w_2, w_3, ..., w_n+1)
        ...
        (w_m-n+1, ..., w_m)
    
    Preparing training data for an n_gram model by converting tokenized sentences
    into overlapping n_gram examples.
    
    Args:
        sentences: List of tokenized sentences
        n: Order of n-gram (1=unigram, 2=bigram, etc.)
        
    Returns:
        List of n-gram tuples
    """
    ngrams = []
    
    for sentence in sentences:
        # Pad with additional start tokens for higher-order n-grams
        # This ensures we can predict the first few words
        if n > 1:
            padded = ['<s>'] * (n - 1) + sentence[1:]  # Already has one <s>
        else:
            padded = sentence
        
        # Extract n-grams      
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i:i + n])
            ngrams.append(ngram)
    
    return ngrams


def split_train_test(sentences: List[List[str]], 
                     test_ratio: float = 0.1,
                     random_seed: int = 42) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Split sentences into training and test sets.
    
    Args:
        sentences: List of tokenized sentences
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_sentences, test_sentences)
    """
    import random
    random.seed(random_seed)
    
    print("\n" + "=" * 80)
    print("STEP 3: SPLITTING DATA INTO TRAINING AND TEST SETS")
    print("=" * 80)
    
    # Shuffle sentences
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]
    
    print(f"\nData split complete:")
    print(f"  Total sentences: {len(sentences):,}")
    print(f"  Training sentences: {len(train):,} ({len(train)/len(sentences)*100:.1f}%)")
    print(f"  Test sentences: {len(test):,} ({len(test)/len(sentences)*100:.1f}%)")
    print(f"  Random seed: {random_seed} (for reproducibility)")
    print(f"\n  Note: Sentences > transcriptions because each transcription")
    print(f"        contains multiple sentences (split on . ! ?)")
    print("=" * 80)
    
    return train, test


def load_akan_dataset(dataset_path: str = "dataset/Akan.xlsx") -> List[str]:
    """
    Load Akan transcriptions from Excel file.
    
    Args:
        dataset_path: Path to Akan.xlsx file
        
    Returns:
        List of transcription strings
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas and openpyxl required. Install with: pip install pandas openpyxl")
    
    print("=" * 80)
    print("LOADING AKAN DATASET")
    print("=" * 80)
    print(f"\nLoading dataset from: {dataset_path}")
    
    df = pd.read_excel(dataset_path)
    
    # Find transcriptions column (case-insensitive)
    trans_col = None
    for col in df.columns:
        if col.lower() == 'transcriptions':
            trans_col = col
            break
    
    if trans_col is None:
        raise ValueError(f"'Transcriptions' column not found. Available columns: {list(df.columns)}")
    
    print(f"  Found column: '{trans_col}'")
    print(f"  Total rows: {len(df):,}")
    
    # Extract transcriptions
    transcriptions = df[trans_col].dropna().astype(str).tolist()
    
    # Filter out empty strings
    transcriptions = [t.strip() for t in transcriptions if t.strip()]
    
    print(f"  Non-empty transcriptions: {len(transcriptions):,}")
    print(f"  Null/empty transcriptions removed: {len(df) - len(transcriptions):,}")
    
    # Show sample
    print(f"\nSample transcriptions (first 3):")
    for i, trans in enumerate(transcriptions[:3], 1):
        print(f"  {i}. {trans[:100]}...")
    
    print("=" * 80)
    
    return transcriptions


# Main execution for testing preprocessing
if __name__ == "__main__":
    print("=" * 80)
    print("PREPROCESSING MODULE - STANDALONE TEST")
    print("=" * 80)
    print("\nThis script tests the preprocessing pipeline on Akan dataset.")
    print("It demonstrates: text loading, cleaning, tokenization, vocabulary building.\n")
    
    # Load Akan dataset
    try:
        transcriptions = load_akan_dataset()
    except Exception as e:
        print(f"\nERROR loading dataset: {e}")
        print("\nMake sure:")
        print("  1. dataset/Akan.xlsx exists")
        print("  2. pandas and openpyxl are installed: pip install pandas openpyxl")
        sys.exit(1)
    
    # Use a subset for testing (first 1000 for speed)
    print(f"\nUsing subset of {min(1000, len(transcriptions))} transcriptions for testing...")
    test_transcriptions = transcriptions[:1000]
    
    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = TextPreprocessor(
        lowercase=False,  # Keep Akan case (important for meaning)
        min_word_freq=2,  # Words must appear at least twice
        max_vocab_size=None
    )
    
    # Fit and transform
    print("\nFitting vocabulary and transforming texts...")
    sentences = preprocessor.fit_transform(test_transcriptions)
    
    # Show sample tokenized sentences
    print(f"\n" + "=" * 80)
    print("SAMPLE TOKENIZED SENTENCES")
    print("=" * 80)
    print(f"\nShowing first 3 tokenized sentences:")
    for i, sent in enumerate(sentences[:3], 1):
        print(f"\n{i}. Length: {len(sent)} tokens")
        print(f"   Tokens: {sent[:15]}..." if len(sent) > 15 else f"   Tokens: {sent}")
    
    # Show n-gram extraction
    print(f"\n" + "=" * 80)
    print("N-GRAM EXTRACTION DEMONSTRATION")
    print("=" * 80)
    
    sample_sentence = sentences[0]
    print(f"\nSample sentence: {sample_sentence[:20]}...")
    
    for n in [1, 2, 3]:
        ngrams = prepare_ngram_data([sample_sentence], n=n)
        print(f"\n{n}-grams extracted: {len(ngrams)}")
        print(f"  First 5 {n}-grams: {ngrams[:5]}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 80)
