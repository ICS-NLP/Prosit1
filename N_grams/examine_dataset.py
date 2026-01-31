#!/usr/bin/env python3
"""
Script to examine the Akan dataset and identify cleaning needs.
"""

import sys
import os

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Please install it in your virtual environment:")
    print("  pip install pandas openpyxl")
    sys.exit(1)

dataset_path = "dataset/Akan.xlsx"

print("=" * 80)
print("EXAMINING AKAN DATASET")
print("=" * 80)

try:
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_excel(dataset_path)
    
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check for transcriptions column (case-insensitive)
    trans_col = None
    for col in df.columns:
        if col.lower() == 'transcriptions':
            trans_col = col
            break
    
    if trans_col is None:
        print("\nERROR: 'Transcriptions' column not found!")
        print(f"Available columns: {list(df.columns)}")
        print("\nPlease check the column name in your Excel file.")
        sys.exit(1)
    
    print(f"\nUsing column: '{trans_col}'")
    
    # Basic statistics
    print(f"\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    
    total_rows = len(df)
    non_null = df[trans_col].notna().sum()
    null_count = df[trans_col].isnull().sum()
    
    print(f"Total rows: {total_rows}")
    print(f"Non-null transcriptions: {non_null}")
    print(f"Null transcriptions: {null_count} ({null_count/total_rows*100:.1f}%)")
    
    # Sample data
    print(f"\n" + "=" * 80)
    print("SAMPLE TRANSCRIPTIONS (First 10 non-null)")
    print("=" * 80)
    
    sample = df[trans_col].dropna().head(10)
    for i, trans in enumerate(sample, 1):
        print(f"\n{i}. {str(trans)[:200]}")
    
    # Data quality analysis
    print(f"\n" + "=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    non_null_trans = df[trans_col].dropna()
    
    # Length statistics
    lengths = non_null_trans.astype(str).str.len()
    print(f"\nText length statistics:")
    print(f"  Mean: {lengths.mean():.1f} characters")
    print(f"  Median: {lengths.median():.1f} characters")
    print(f"  Min: {lengths.min()} characters")
    print(f"  Max: {lengths.max()} characters")
    
    # Word count statistics
    word_counts = non_null_trans.astype(str).str.split().str.len()
    print(f"\nWord count statistics:")
    print(f"  Mean: {word_counts.mean():.1f} words")
    print(f"  Median: {word_counts.median():.1f} words")
    print(f"  Min: {word_counts.min()} words")
    print(f"  Max: {word_counts.max()} words")
    
    # Check for empty strings
    empty_strings = (non_null_trans.astype(str).str.strip() == '').sum()
    print(f"\nEmpty strings (after stripping): {empty_strings}")
    
    # Check for very short transcriptions
    very_short = (word_counts < 3).sum()
    print(f"Very short transcriptions (< 3 words): {very_short}")
    
    # Character analysis
    print(f"\n" + "=" * 80)
    print("CHARACTER ANALYSIS")
    print("=" * 80)
    
    all_text = ' '.join(non_null_trans.astype(str))
    unique_chars = set(all_text)
    print(f"Unique characters: {len(unique_chars)}")
    print(f"Sample characters: {sorted(list(unique_chars))[:50]}")
    
    # Check for special characters/issues
    print(f"\n" + "=" * 80)
    print("POTENTIAL ISSUES TO CLEAN")
    print("=" * 80)
    
    issues = []
    
    # Check for URLs
    url_count = non_null_trans.astype(str).str.contains(r'http[s]?://', regex=True, na=False).sum()
    if url_count > 0:
        issues.append(f"URLs found: {url_count} transcriptions")
    
    # Check for email addresses
    email_count = non_null_trans.astype(str).str.contains(r'@', regex=True, na=False).sum()
    if email_count > 0:
        issues.append(f"Email addresses found: {email_count} transcriptions")
    
    # Check for excessive whitespace
    excessive_whitespace = non_null_trans.astype(str).str.contains(r'\s{3,}', regex=True, na=False).sum()
    if excessive_whitespace > 0:
        issues.append(f"Excessive whitespace: {excessive_whitespace} transcriptions")
    
    # Check for numbers (might want to keep or remove)
    number_count = non_null_trans.astype(str).str.contains(r'\d', regex=True, na=False).sum()
    if number_count > 0:
        issues.append(f"Numbers found: {number_count} transcriptions")
    
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No obvious issues detected!")
    
    # Show problematic examples
    print(f"\n" + "=" * 80)
    print("EXAMPLES OF POTENTIAL CLEANING NEEDS")
    print("=" * 80)
    
    # Show examples with URLs
    if url_count > 0:
        print("\nTranscriptions with URLs:")
        url_examples = df[df[trans_col].astype(str).str.contains(r'http', regex=True, na=False)][trans_col].head(3)
        for ex in url_examples:
            print(f"  {str(ex)[:150]}")
    
    # Show examples with excessive whitespace
    if excessive_whitespace > 0:
        print("\nTranscriptions with excessive whitespace:")
        ws_examples = df[df[trans_col].astype(str).str.contains(r'\s{3,}', regex=True, na=False)][trans_col].head(3)
        for ex in ws_examples:
            print(f"  {repr(str(ex)[:150])}")
    
    print(f"\n" + "=" * 80)
    print("DATASET EXAMINATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal usable transcriptions: {non_null - empty_strings}")
    print(f"Recommended: Use {non_null - empty_strings} transcriptions for training")
    
except FileNotFoundError:
    print(f"\nERROR: File not found at {dataset_path}")
    print("Please make sure the file exists.")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
