# PROSIT 1: Building and Adapting Language Models

## Project Overview

This repository contains the implementation for **PROSIT 1** of the ICS554 Natural Language Processing course. The project was completed as part of an internship simulation at **Ankora**, an AI research lab based in Ghana that specializes in developing speech recognition systems.

### Project Context

Ankora needed two specialized language models:

1. **Low-Resource Language Model**: A specialized language model for a low-resource African language (Akan)
2. **Domain-Specific Model**: An English language model tuned to a specific domain (Finance)

This repository contains the implementation for the **N-gram Language Model** for the Akan language (Section B of the assignment).

---

## Team Members

- **Michael**
- **Thomas Kojo Quarshie**
- **Naa Lamle Boye**
- **Sadik Abubakari**
- **Patrick**

---

## Repository Structure

### Low-Resource Language (Akan) - Section B

This section contains implementations for building language models for the Akan language:

#### 1. **N-gram Model (This Repository)**
- **Contributors**: Team members working on n-gram implementation
- **Dataset**: Akan transcriptions from `dataset/Akan.xlsx`
- **Implementation**: Complete n-gram language model built from scratch
- **Features**:
  - Multiple n-gram orders (unigram, bigram, trigram, 4-gram)
  - Advanced smoothing techniques (Kneser-Ney, Add-k, Linear Interpolation)
  - Comprehensive evaluation metrics (perplexity, coverage, OOV rate)
  - Text generation capabilities

#### 2. **Neural Network Model with Transformer Architecture**
- **Contributors**: Thomas, Naa, and Patrick
- **Dataset**: Akan language/dataset
- **Implementation**: Transformer architecture built from scratch
- **Link**: [Will be added here]

#### 3. **LSTM Model**
- **Contributors**: Thomas, Naa, and Patrick
- **Dataset**: Akan language/dataset
- **Implementation**: LSTM-based language model
- **Link**: [Will be added here]

### Domain-Specific Model (English) - Section C

#### **Financial Domain Fine-tuning**
- **Contributors**: Michael and Sadik
- **Dataset**: Financial dataset
- **Implementation**: Fine-tuned language model for financial domain
- **Dataset Link**: [Will be added here]

---

## N-gram Model Implementation

### Overview

This implementation provides a complete n-gram language model for the Akan language, a low-resource African language. The model is built from scratch and includes:

- **Data Preprocessing**: Custom preprocessing for Akan text with Unicode normalization
- **N-gram Counting**: Efficient extraction and counting of n-grams
- **Smoothing Techniques**: Multiple smoothing methods to handle unseen n-grams
- **Evaluation**: Comprehensive metrics including perplexity, coverage, and OOV rate
- **Text Generation**: Temperature-based sampling for text generation

### Dataset

- **Source**: `dataset/Akan.xlsx`
- **Size**: 18,787 transcriptions
- **Language**: Akan (Ghana)
- **Content**: Real-world transcriptions describing scenes and activities
- **Processing**: 
  - 67,880 sentences after preprocessing (3.6 sentences per transcription on average)
  - 11,319 word vocabulary (after filtering words with frequency < 2)

### Key Results

**Best Model: 2-gram with Kneser-Ney Smoothing**

| Model | Perplexity | Cross-Entropy | Coverage (Weighted) |
|-------|------------|---------------|---------------------|
| **2-gram** | **89.11** | **6.48 bits** | **86.8%** |
| 3-gram | 100.83 | 6.66 bits | 61.0% |
| 4-gram | 144.19 | 7.17 bits | 40.0% |

**Key Findings:**
- 2-gram model achieved best performance (lowest perplexity: 89.11)
- Highest coverage: 86.8% (most frequent test patterns were seen)
- Very low OOV rate: 0.19% (almost all words are in vocabulary)
- Higher-order n-grams (3-gram, 4-gram) suffered from data sparsity

**Insight**: For this dataset size, lower-order n-grams generalize better than higher-order models due to data sparsity issues.

### Files Structure

```
N_grams/
├── preprocessing.py          # Text preprocessing and vocabulary building
├── ngram_model.py            # Core n-gram language model
├── smoothing.py              # Smoothing techniques (Kneser-Ney, Add-k, etc.)
├── evaluation.py             # Evaluation metrics (perplexity, coverage)
├── train.py                  # Main training script
├── test_model.py             # Model testing and text generation
├── examine_dataset.py        # Dataset analysis tool
├── requirements.txt          # Python dependencies
├── models/                   # Trained models
│   ├── akan_best.pkl        # 3-gram model
│   └── akan_best_best_2gram.pkl  # Best model (2-gram)
├── output.txt                # Training output with comparison
├── output2.txt               # Text generation examples
├── output3.txt               # Interactive testing output
└── README.md                 # This file
```

### Usage

#### Training a Model

```bash
# Basic training (3-gram with Kneser-Ney)
python train.py

# Compare different n-gram orders and save best
python train.py --order 3 --compare --output models/akan_best.pkl

# Custom configuration
python train.py --order 2 --smoothing kneser_ney --min_freq 2
```

#### Testing a Model

```bash
# Interactive mode
python test_model.py --model models/akan_best_best_2gram.pkl

# With specific prompt
python test_model.py --model models/akan_best_best_2gram.pkl --prompt "Nnipa" --num_samples 5
```

#### Examining the Dataset

```bash
python examine_dataset.py
```

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pandas` - Data loading from Excel
- `openpyxl` - Excel file support
- `numpy` - Numerical operations
- `nltk` - Natural language processing utilities
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

### Key Features

1. **Akan-Specific Preprocessing**
   - Unicode normalization (preserves ɛ, ɔ, etc.)
   - Case preservation (lowercase changes meaning in Akan)
   - Custom vocabulary filtering

2. **Advanced Smoothing**
   - Kneser-Ney smoothing (state-of-the-art)
   - Add-k smoothing
   - Linear interpolation
   - Good-Turing discount estimation

3. **Comprehensive Evaluation**
   - Perplexity calculation
   - Coverage analysis (unique and weighted)
   - OOV rate measurement
   - Model comparison utilities

4. **Text Generation**
   - Temperature-based sampling
   - Top-k and top-p sampling support
   - Interactive testing mode

### Technical Details

**Preprocessing Pipeline:**
1. Load transcriptions from Excel
2. Unicode normalization (NFC)
3. Sentence splitting (on . ! ?)
4. Word tokenization
5. Vocabulary building (min frequency filtering)
6. OOV handling (<UNK> token)
7. Special token insertion (<s>, </s>)

**Training Process:**
1. Extract n-grams from training sentences
2. Count n-gram and context frequencies
3. Fit smoothing method (Kneser-Ney)
4. Build probability distributions

**Evaluation Metrics:**
- **Perplexity**: Measures model's "surprise" at test data (lower is better)
- **Coverage**: Percentage of test n-grams seen in training (higher is better)
- **OOV Rate**: Percentage of words not in vocabulary (lower is better)

### Results Summary

**Training Statistics:**
- Training sentences: 61,092 (90%)
- Test sentences: 6,788 (10%)
- Vocabulary size: 11,260 words
- Unique 2-grams: 129,315
- Unique 3-grams: 311,811

**Best Model Performance (2-gram):**
- Test Perplexity: 89.11
- Coverage (weighted): 86.8%
- OOV Rate: 0.19%
- Cross-entropy: 6.48 bits

**Generated Text Examples:**
```
Prompt: "Nnipa"
Generated: "nnipa bi gyina lɔɔre kwan so"
         "nnipa no bi nso si ne nkyɛn a ɔno nso ɔne dan no mu"
         "nnipa pii ebi tetare ho wɔn ho ahyia"
```

### Code Documentation

Each code file contains detailed documentation with:
- Step-by-step examples using real Akan data
- Mathematical formulations
- Input/output transformations
- Complete walkthroughs

See individual file docstrings for detailed explanations.

### Additional Resources

- `COMPLETE_WALKTHROUGH.md`: End-to-end pipeline walkthrough
- `CODE_EXAMPLES_GUIDE.md`: Guide to understanding code examples
- `UNDERSTANDING_RESULTS.md`: Explanation of evaluation metrics

---

## Assignment Requirements

This project addresses **Section B** of PROSIT 1, which requires:

1. Building a specialized language model for a low-resource African language
2. Using n-gram models (as proposed for low-resource scenarios)
3. Demonstrating model effectiveness through evaluation
4. Providing detailed technical documentation

### Section B Questions Addressed

1. **Data Used**: Akan transcriptions from Excel file (18,787 transcriptions)
2. **N-gram vs Neural**: Agreed that n-grams are better for low-resource languages
3. **Training Process**: Detailed pipeline with Kneser-Ney smoothing
4. **Evaluation**: Perplexity, coverage, and OOV rate metrics
5. **Results**: 2-gram model achieved best performance (perplexity 89.11)
6. **Additional Work**: Systematic comparison of n-gram orders, data sparsity analysis

---

## Repository Information

**GitHub Repository**: https://github.com/ICS-NLP/Prosit1.git

**Course**: ICS554 Natural Language Processing  
**Institution**: [Your Institution]  
**Academic Year**: [Academic Year]

---

## Citation

If you use this code, please cite:

```
PROSIT 1: Building and Adapting Language Models
ICS554 Natural Language Processing
Team: Michael, Thomas Kojo Quarshie, Naa Lamle Boye, Sadik Abubakari, Patrick
Repository: https://github.com/ICS-NLP/Prosit1.git
```

---

## License

[Add license information if applicable]

---

## Contact

For questions about this implementation, please contact the team members or refer to the course materials.
