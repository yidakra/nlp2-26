"""
Preprocessor for CC-Aligned TSV parallel corpus (English-Traditional Chinese).
Converts TSV format to HuggingFace Dataset format.

File format:
domain \t en_url \t en_content \t zh_url \t zh_content

The content columns contain web text with pipe-delimited segments.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import re

import datasets
from tqdm import tqdm


@dataclass
class ParallelExample:
    """Represents a parallel text example."""
    source_text: str
    target_text: str
    domain: str
    source_url: str
    target_url: str


class CCSAlignTSVPreprocessor:
    """
    Preprocessor for CC-Aligned TSV corpus.
    
    Args:
        tsv_path: Path to the TSV file (en_XX-zh_TW.tsv)
        min_length: Minimum character length for valid text (default: 10)
        max_length: Maximum character length for valid text (default: 10000)
        val_split: Fraction for validation split (default: 0.1)
    """

    def __init__(
        self,
        tsv_path: Path,
        min_length: int = 10,
        max_length: int = 10000,
        val_split: float = 0.1,
    ):
        self.tsv_path = Path(tsv_path)
        self.min_length = min_length
        self.max_length = max_length
        self.val_split = val_split

        if not self.tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {self.tsv_path}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text:
            return False
        cleaned = self._clean_text(text)
        return self.min_length <= len(cleaned) <= self.max_length

    def _parse_line(self, line: str) -> Optional[ParallelExample]:
        """Parse a single TSV line."""
        parts = line.rstrip('\n').split('\t')
        
        if len(parts) < 5:
            return None
        
        domain = parts[0]
        en_url = parts[1]
        en_content = parts[2]
        zh_url = parts[3]
        zh_content = parts[4]
        
        # Clean text
        en_text = self._clean_text(en_content)
        zh_text = self._clean_text(zh_content)
        
        # Validate both sides
        if not self._is_valid_text(en_text) or not self._is_valid_text(zh_text):
            return None
        
        return ParallelExample(
            source_text=en_text,
            target_text=zh_text,
            domain=domain,
            source_url=en_url,
            target_url=zh_url,
        )

    def _example_generator(self):
        """
        Generator function that yields examples without loading all into memory.
        """
        skipped = 0
        processed = 0
        
        # Count total lines first
        total_lines = sum(1 for _ in open(self.tsv_path, 'r', encoding='utf-8'))
        
        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Processing CC-Aligned TSV"):
                try:
                    example = self._parse_line(line)
                    if example:
                        yield {
                            'source_text': example.source_text,
                            'target_text': example.target_text,
                            'domain': example.domain,
                            'source_url': example.source_url,
                            'target_url': example.target_url,
                        }
                        processed += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                    continue
        
        print(f"Processed {processed} valid examples, skipped {skipped}")

    def preprocess(self, output_dir: Path) -> datasets.DatasetDict:
        """
        Preprocess TSV file and save as HuggingFace Dataset.
        Streams data to avoid loading entire file into memory.
        
        Args:
            output_dir: Output directory for the dataset
            
        Returns:
            HuggingFace DatasetDict with 'train' and 'validation' splits
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define dataset schema
        features = datasets.Features({
            'source_text': datasets.Value('string'),
            'target_text': datasets.Value('string'),
            'domain': datasets.Value('string'),
            'source_url': datasets.Value('string'),
            'target_url': datasets.Value('string'),
        })
        
        # Create dataset from generator (streams data, doesn't load all to memory)
        dataset = datasets.Dataset.from_generator(
            self._example_generator,
            features=features,
            cache_dir=None,  # Disable caching to save space
        )
        
        print(f"Created dataset with {len(dataset)} total examples")
        
        # Split into train/validation
        if self.val_split > 0:
            split_dataset = dataset.train_test_split(test_size=self.val_split, seed=42)
            dataset_dict = datasets.DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
            print(f"Train split: {len(dataset_dict['train'])} examples")
            print(f"Validation split: {len(dataset_dict['validation'])} examples")
        else:
            dataset_dict = datasets.DatasetDict({'train': dataset})
        
        # Save dataset (use compression to save disk space)
        dataset_dict.save_to_disk(
            str(output_dir),
            max_shard_size='1GB',  # Split into 1GB shards
        )
        print(f"Dataset saved to {output_dir}")
        
        return dataset_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess CC-Aligned TSV corpus"
    )
    parser.add_argument(
        "--tsv-path",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "cc_aligned" / "en_XX-zh_TW.tsv",
        help="Path to TSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "processed_data" / "cc_aligned",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10000,
        help="Maximum text length",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (0.0-1.0)",
    )
    
    args = parser.parse_args()
    
    preprocessor = CCSAlignTSVPreprocessor(
        tsv_path=args.tsv_path,
        min_length=args.min_length,
        max_length=args.max_length,
        val_split=args.val_split,
    )
    
    dataset_dict = preprocessor.preprocess(args.output_dir)
    total_examples = sum(len(ds) for ds in dataset_dict.values())
    print(f"Final dataset has {total_examples} examples")
