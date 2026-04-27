"""
Preprocessor for CC-Aligned TSV parallel corpus (English-Traditional Chinese).
Converts TSV format to HuggingFace Dataset format.

File format:
domain \t en_url \t en_content \t zh_url \t zh_content

The content columns contain web text with pipe-delimited segments.
"""

from pathlib import Path
from dataclasses import dataclass
import re

import datasets  # type: ignore[import-untyped]
from tqdm import tqdm


@dataclass
class ParallelExample:
    """Represents a parallel text example."""
    en: str
    zh: str
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
        train_size: Number of training examples to keep (default: 15000)
        val_size: Number of validation examples to keep (default: 2000)
    """

    def __init__(
        self,
        tsv_path: Path,
        min_length: int = 10,
        max_length: int = 10000,
        train_size: int = 15000,
        val_size: int = 2000,
    ):
        self.tsv_path = Path(tsv_path)
        self.min_length = min_length
        self.max_length = max_length
        self.train_size = train_size
        self.val_size = val_size

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

    def _parse_line(self, line: str) -> ParallelExample | None:
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
            en=en_text,
            zh=zh_text,
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
                            'en': example.en,
                            'zh': example.zh,
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
            'en': datasets.Value('string'),
            'zh': datasets.Value('string'),
            'domain': datasets.Value('string'),
            'source_url': datasets.Value('string'),
            'target_url': datasets.Value('string'),
        })
        
        # Create dataset from generator (streams data, doesn't load all to memory)
        dataset = datasets.Dataset.from_generator(  # type: ignore[reportUnknownMemberType]
            self._example_generator,
            features=features,
            cache_dir=None,  # type: ignore[arg-type]
        )
        
        print(f"Created dataset with {len(dataset)} total examples")
        
        # Keep only the requested number of examples
        max_examples = self.train_size + self.val_size
        if len(dataset) > max_examples:
            dataset = dataset.shuffle(seed=42).select(range(max_examples))

        # Create train/validation splits from the first N examples
        train_examples = min(self.train_size, len(dataset))
        val_examples = min(self.val_size, max(0, len(dataset) - train_examples))

        train_dataset = dataset.select(range(train_examples))
        validation_dataset = dataset.select(range(train_examples, train_examples + val_examples))

        dataset_dict = datasets.DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })
        print(f"Train split: {len(dataset_dict['train'])} examples")
        print(f"Validation split: {len(dataset_dict['validation'])} examples")
        
        # Save dataset (use compression to save disk space)
        dataset_dict.save_to_disk(  # type: ignore[reportUnknownMemberType]
            str(output_dir),
            max_shard_size='1GB',
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
        "--train-size",
        type=int,
        default=15000,
        help="Number of training examples to keep",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="Number of validation examples to keep",
    )
    
    args = parser.parse_args()
    
    preprocessor = CCSAlignTSVPreprocessor(
        tsv_path=args.tsv_path,
        min_length=args.min_length,
        max_length=args.max_length,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    
    dataset_dict = preprocessor.preprocess(args.output_dir)
    total_examples = sum(len(ds) for ds in dataset_dict.values())
    print(f"Final dataset has {total_examples} examples")
