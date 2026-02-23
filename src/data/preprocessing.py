import re
import hashlib
import argparse
from pathlib import Path
from collections import Counter

from datasets import load_dataset, load_from_disk, Dataset
from langdetect import detect, LangDetectException
from tqdm import tqdm

class DataPreprocessor:
    """
    Preprocesses OpenWebText data for LLM training.
    
    Handles:
    - Loading data from HuggingFace
    - Filtering non-English content
    - Removing test set sentences (prevents data leakage)
    - Cleaning HTML, URLs, code, special characters
    - Saving processed dataset
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.test_sentence_hashes = set()  # Will store hashes of test sentences
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'kept': 0,
            'filtered_language': 0,
            'filtered_quality': 0,
            'filtered_too_short': 0,
        }
    
    def _extract_sentences(self, text: str) -> list:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def load_test_sets(self, test_data_path: str):
        """
        Load test sets and create hashes of all sentences.
        These sentences must be excluded from training data.
        
        Args:
            test_data_path: Path to the NLP26 test dataset
        """
        print("Loading test sets for filtering...")
        
        # Load NLP26 OpenWebText test split
        print(f"  Loading NLP26 test split from: {test_data_path}")
        test_dataset = load_from_disk(test_data_path)
        print(f"  Found {len(test_dataset):,} test documents")
        
        # Extract and hash all sentences from test set
        print("  Hashing test sentences (this takes a few minutes)...")
        
        for i, doc in enumerate(tqdm(test_dataset, desc="  Processing test docs")):
            sentences = self._extract_sentences(doc['text'])
            for sentence in sentences:
                # Normalize: lowercase and strip whitespace
                normalized = sentence.lower().strip()
                # Only hash sentences with enough content
                if len(normalized) > 50:  # Ignore very short sentences
                    sentence_hash = hashlib.md5(normalized.encode()).hexdigest()
                    self.test_sentence_hashes.add(sentence_hash)
        
        print(f"  Created {len(self.test_sentence_hashes):,} sentence hashes")
        print("  Test set loading complete!\n")


    def detect_language(self, text: str) -> str:
        """
        Detect the language of text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (e.g., 'en' for English) or 'unknown'
        """
        try:
            # Use first 1000 characters for speed
            return detect(text[:1000])
        except LangDetectException:
            return "unknown"
        except:
            return "error"
    
    def clean_html(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with HTML tags removed
        """
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def clean_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        # Remove http:// and https:// URLs
        clean = re.sub(r'https?://\S+', '', text)
        # Remove www. URLs
        clean = re.sub(r'www\.\S+', '', clean)
        return clean
    
    def clean_code_blocks(self, text: str) -> str:
        """
        Remove markdown code blocks from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with code blocks removed
        """
        # Remove markdown code blocks (```...```)
        clean = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code (`...`)
        clean = re.sub(r'`[^`]+`', '', clean)
        return clean
    
    def clean_special_characters(self, text: str) -> str:
        """
        Clean special and corrupted characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove replacement characters (indicate encoding issues)
        clean = text.replace('�', '')
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean)
        # Remove excessive punctuation repetition
        clean = re.sub(r'([.!?])\1{3,}', r'\1', clean)
        return clean.strip()
    
    def remove_test_sentences(self, text: str) -> str:
        """
        Remove sentences that appear in the test set.
        This prevents data leakage.
        
        Args:
            text: Input text
            
        Returns:
            Text with test sentences removed
        """
        if not self.test_sentence_hashes:
            return text
        
        sentences = self._extract_sentences(text)
        filtered_sentences = []
        
        for sentence in sentences:
            normalized = sentence.lower().strip()
            if len(normalized) > 50:
                sentence_hash = hashlib.md5(normalized.encode()).hexdigest()
                # Keep sentence only if it's NOT in test set
                if sentence_hash not in self.test_sentence_hashes:
                    filtered_sentences.append(sentence)
            else:
                # Keep short sentences (not worth filtering)
                filtered_sentences.append(sentence)
        
        return ' '.join(filtered_sentences)
    
    def is_valid_document(self, text: str, min_length: int = 100) -> bool:
        """
        Check if a document passes quality filters.
        
        Args:
            text: Input text
            min_length: Minimum character length
            
        Returns:
            True if document is valid, False otherwise
        """
        # Too short after cleaning
        if len(text) < min_length:
            self.stats['filtered_too_short'] += 1
            return False
        
        # Has excessive character repetition (sign of corruption)
        if re.search(r'(.)\1{20,}', text):
            self.stats['filtered_quality'] += 1
            return False
        
        # Has too many replacement characters
        if text.count('�') > 5:
            self.stats['filtered_quality'] += 1
            return False
        
        return True
    
    def process_document(self, doc: dict) -> dict | None:
        """
        Process a single document through all cleaning steps.
        
        Args:
            doc: Document with 'text' field
            
        Returns:
            Cleaned document or None if filtered out
        """
        text = doc['text']
        self.stats['total_processed'] += 1
        
        # Step 1: Language filter (do this first - it's fast to reject non-English)
        language = self.detect_language(text)
        if language != 'en':
            self.stats['filtered_language'] += 1
            return None
        
        # Step 2: Remove test set sentences (CRITICAL for preventing data leakage)
        text = self.remove_test_sentences(text)
        
        # Step 3: Clean HTML tags
        text = self.clean_html(text)
        
        # Step 4: Clean URLs
        text = self.clean_urls(text)
        
        # Step 5: Clean code blocks
        text = self.clean_code_blocks(text)
        
        # Step 6: Clean special characters
        text = self.clean_special_characters(text)
        
        # Step 7: Quality validation
        if not self.is_valid_document(text):
            return None
        
        # Document passed all filters!
        self.stats['kept'] += 1
        return {'text': text}
    
    def preprocess(
        self,
        num_samples: int,
        test_data_path: str,
        output_path: str,
    ) -> Dataset:
        """
        Main preprocessing pipeline.
        
        Args:
            num_samples: Number of training samples to collect
            test_data_path: Path to NLP26 test dataset
            output_path: Where to save processed dataset
            
        Returns:
            Processed dataset
        """
        # Step 1: Load test sets for filtering
        self.load_test_sets(test_data_path)
        
        # Step 2: Load OpenWebText with streaming
        print("Loading OpenWebText from HuggingFace (streaming)...")
        dataset_stream = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True
        )
        
        # Step 3: Process documents
        print(f"Processing documents (target: {num_samples:,} clean samples)...")
        
        processed_samples = []
        
        # We need to process more documents than num_samples because some will be filtered
        # Estimate: ~5% will be filtered, so process 10% extra to be safe
        max_to_check = int(num_samples * 1.2)
        
        for i, doc in enumerate(tqdm(dataset_stream, total=max_to_check, desc="Processing")):
            # Stop if we have enough samples
            if len(processed_samples) >= num_samples:
                break
            
            # Stop if we've checked too many without getting enough
            if i >= max_to_check:
                print(f"\nWarning: Checked {max_to_check:,} docs but only got {len(processed_samples):,} samples")
                break
            
            # Process the document
            processed = self.process_document(doc)
            if processed is not None:
                processed_samples.append(processed)
        
        # Step 4: Create dataset
        print(f"\nCreating dataset from {len(processed_samples):,} samples...")
        processed_dataset = Dataset.from_list(processed_samples)
        
        # Step 5: Save to disk
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        processed_dataset.save_to_disk(str(output_path))
        print(f"Saved processed dataset to: {output_path}")
        
        # Step 6: Print statistics
        self._print_stats()
        
        return processed_dataset
    
    def _print_stats(self):
        """Print processing statistics."""
        print("\n" + "=" * 60)
        print("PREPROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total documents processed: {self.stats['total_processed']:,}")
        print(f"Documents kept:            {self.stats['kept']:,}")
        print(f"Filtered (non-English):    {self.stats['filtered_language']:,}")
        print(f"Filtered (quality issues): {self.stats['filtered_quality']:,}")
        print(f"Filtered (too short):      {self.stats['filtered_too_short']:,}")
        
        if self.stats['total_processed'] > 0:
            keep_rate = self.stats['kept'] / self.stats['total_processed'] * 100
            print(f"\nKeep rate: {keep_rate:.1f}%")
        print("=" * 60)


def main(num_samples: int, seed: int, test_data_path: str, output_path: str):
    """
    Entry point for preprocessing.
    
    Args:
        num_samples: Number of clean samples to collect
        seed: Random seed for reproducibility
        test_data_path: Path to NLP26 test dataset
        output_path: Where to save processed data
    """
    print("=" * 60)
    print("PikoGPT Data Preprocessing")
    print("=" * 60)
    print(f"Target samples:  {num_samples:,}")
    print(f"Random seed:     {seed}")
    print(f"Test data path:  {test_data_path}")
    print(f"Output path:     {output_path}")
    print("=" * 60 + "\n")
    
    preprocessor = DataPreprocessor(seed=seed)
    dataset = preprocessor.preprocess(
        num_samples=num_samples,
        test_data_path=test_data_path,
        output_path=output_path,
    )
    
    print("\n✓ Preprocessing complete!")
    return dataset


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Preprocess OpenWebText for PikoGPT")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of clean samples to collect (default: 100000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="data/raw/NLP26_OWT_eval/test",
        help="Path to NLP26 test dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/openwebtext_clean",
        help="Where to save processed dataset"
    )
    
    args = parser.parse_args()
    
    main(
        num_samples=args.num_samples,
        seed=args.seed,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
    )