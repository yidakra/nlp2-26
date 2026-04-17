"""
Preprocessing class for Hong Kong Legislation parallel corpus.
Converts XML legislation documents (English-Chinese pairs) to HuggingFace Dataset format.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import datasets
from datasets import Dataset
from tqdm import tqdm


@dataclass
class ParallelExample:
    """Represents a parallel text pair"""
    source_text: str  # English
    target_text: str  # Traditional Chinese
    doc_id: str
    source_lang: str = "en"
    target_lang: str = "zh_TW"


class HKLegislationPreprocessor:
    """
    Preprocessor for Hong Kong Legislation parallel corpus.
    
    This class handles:
    - Discovering paired English and Traditional Chinese legislation files
    - Parsing XML legislation documents
    - Extracting and aligning parallel text segments
    - Converting to HuggingFace Dataset format
    """

    def __init__(
        self,
        data_dir: str,
        min_text_length: int = 10,
        max_text_length: Optional[int] = None,
        val_split: float = 0.1,
        verbose: bool = True,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Root directory containing en/ and zh-hant/ subdirectories
            min_text_length: Minimum characters required for a valid segment
            max_text_length: Maximum characters allowed (None for no limit)
            verbose: Whether to print progress information
        """
        self.data_dir = Path(data_dir)
        self.en_dir = self.data_dir / "en"
        self.zh_dir = self.data_dir / "zh-hant"
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.verbose = verbose
        self.val_split = val_split
        
        # XML namespace
        self.ns = {
            'hklm': 'http://www.xml.gov.hk/schemas/hklm/1.0',
            'xhtml': 'http://www.w3.org/1999/xhtml',
            'dc': 'http://purl.org/dc/elements/1.1/',
        }

    def _extract_text_from_element(self, element: ET.Element, recursive: bool = True) -> str:
        """
        Extract all text content from an XML element.
        
        Args:
            element: XML element to extract from
            recursive: Whether to extract text from child elements
            
        Returns:
            Extracted text with whitespace normalized
        """
        if element is None:
            return ""
        
        if recursive:
            # Get all text including from child elements
            text_parts = []
            if element.text:
                text_parts.append(element.text)
            for child in element:
                text_parts.append(self._extract_text_from_element(child, recursive=True))
                if child.tail:
                    text_parts.append(child.tail)
            text = "".join(text_parts)
        else:
            # Only direct text
            text = element.text or ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better alignment and processing.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common XML artifacts
        text = text.replace('\\n', ' ').replace('\\t', ' ')
        return text

    def _parse_legislation_xml(self, xml_path: str) -> Dict[str, str]:
        """
        Parse a legislation XML file and extract text content.
        
        Args:
            xml_path: Path to the XML file
            
        Returns:
            Dictionary with extracted metadata and content
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {xml_path}: {e}")
            return {}

        result = {}
        
        # Extract metadata
        meta = root.find('hklm:meta', self.ns)
        if meta is not None:
            doc_name = meta.find('hklm:docName', self.ns)
            if doc_name is not None:
                result['doc_name'] = doc_name.text or ""
            
            doc_type = meta.find('hklm:docType', self.ns)
            if doc_type is not None:
                result['doc_type'] = doc_type.text or ""

        # Extract main content
        main = root.find('hklm:main', self.ns)
        if main is not None:
            # Extract full text from main element
            full_text = self._extract_text_from_element(main, recursive=True)
            result['full_text'] = full_text
            
            # Try to extract key sections separately
            sections = []
            
            # Get long title
            long_title = main.find('.//hklm:longTitle', self.ns)
            if long_title is not None:
                title_text = self._extract_text_from_element(long_title)
                if title_text:
                    sections.append(title_text)
            
            # Get paragraphs/provisions
            for para in main.findall('.//hklm:paragraph', self.ns):
                para_text = self._extract_text_from_element(para)
                if para_text and len(para_text) > self.min_text_length:
                    sections.append(para_text)
            
            # Get content from generic content elements
            for content in main.findall('.//hklm:content', self.ns):
                content_text = self._extract_text_from_element(content)
                if content_text and len(content_text) > self.min_text_length:
                    sections.append(content_text)
            
            result['sections'] = sections
            result['num_sections'] = len(sections)
        
        return result

    def _extract_doc_id(self, en_dir: str) -> str:
        """
        Extract document ID from directory name (e.g., 'A1_en_c' -> 'A1').
        
        Args:
            en_dir: English directory name
            
        Returns:
            Document ID
        """
        # Remove language/encoding suffix
        parts = en_dir.split('_')
        if parts[-2] == 'en' and parts[-1] == 'c':
            return '_'.join(parts[:-2])
        return en_dir

    def _find_paired_documents(self) -> List[Tuple[str, str]]:
        """
        Find all paired English-Chinese legislation documents.
        
        Returns:
            List of tuples (en_xml_path, zh_xml_path)
        """
        if not self.en_dir.exists() or not self.zh_dir.exists():
            raise FileNotFoundError(
                f"Data directory structure not found. Expected:\n"
                f"  {self.en_dir}\n"
                f"  {self.zh_dir}"
            )
        
        paired_docs = []
        
        # Iterate through English directories
        for en_subdir in self.en_dir.iterdir():
            if not en_subdir.is_dir():
                continue
            
            # Extract document ID
            doc_id = self._extract_doc_id(en_subdir.name)
            
            # Construct expected Chinese directory name
            zh_subdir_name = f"{doc_id}_zh-Hant_c"
            zh_subdir = self.zh_dir / zh_subdir_name
            
            if not zh_subdir.exists():
                if self.verbose:
                    print(f"Warning: No Chinese counterpart for {en_subdir.name}")
                continue
            
            # Find XML files (usually one per directory)
            en_xmls = list(en_subdir.glob("*.xml"))
            zh_xmls = list(zh_subdir.glob("*.xml"))
            
            if not en_xmls or not zh_xmls:
                continue
            
            # Use first XML file found in each directory
            paired_docs.append((str(en_xmls[0]), str(zh_xmls[0])))
        
        return paired_docs

    def process_document_pair(self, en_path: str, zh_path: str) -> List[ParallelExample]:
        """
        Process a pair of English and Chinese legislation documents.
        
        Args:
            en_path: Path to English XML file
            zh_path: Path to Traditional Chinese XML file
            
        Returns:
            List of ParallelExample pairs
        """
        # Parse both files
        en_data = self._parse_legislation_xml(en_path)
        zh_data = self._parse_legislation_xml(zh_path)
        
        if not en_data or not zh_data:
            return []
        
        # Extract document ID
        doc_id = Path(en_path).parent.name.rsplit('_', 2)[0]
        
        examples = []
        
        # Use full text if sections aren't available
        en_text = en_data.get('full_text', '')
        zh_text = zh_data.get('full_text', '')
        
        # Normalize texts
        en_text = self._normalize_text(en_text)
        zh_text = self._normalize_text(zh_text)
        
        # Check length constraints
        if len(en_text) < self.min_text_length or len(zh_text) < self.min_text_length:
            return []
        
        if self.max_text_length is not None:
            en_text = en_text[:self.max_text_length]
            zh_text = zh_text[:self.max_text_length]
        
        # Create parallel example
        example = ParallelExample(
            source_text=en_text,
            target_text=zh_text,
            doc_id=doc_id,
        )
        examples.append(example)
        
        return examples

    def preprocess(self) -> datasets.DatasetDict:
        """
        Preprocess all paired legislation documents and create HuggingFace Dataset.
        
        Returns:
            HuggingFace DatasetDict with 'train' and 'validation' splits
        """
        if self.verbose:
            print(f"Discovering paired documents in {self.data_dir}...")
        
        paired_docs = self._find_paired_documents()
        
        if self.verbose:
            print(f"Found {len(paired_docs)} paired document(s)")
        
        all_examples = []
        
        for en_path, zh_path in tqdm(
            paired_docs,
            desc="Processing documents",
            disable=not self.verbose
        ):
            examples = self.process_document_pair(en_path, zh_path)
            all_examples.extend(examples)
        
        if self.verbose:
            print(f"Extracted {len(all_examples)} parallel examples")
        
        # Convert to dataset format
        data_dict = {
            'source_text': [ex.source_text for ex in all_examples],
            'target_text': [ex.target_text for ex in all_examples],
            'doc_id': [ex.doc_id for ex in all_examples],
            'num_tokens_source': [len(ex.source_text.split()) for ex in all_examples],
            'num_tokens_target': [len(ex.target_text.split()) for ex in all_examples],
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        if self.verbose:
            print(f"Created dataset with {len(dataset)} examples")
            print(f"Dataset schema: {dataset.features}")
        
        # Split into train/validation
        if self.val_split > 0:
            split_dataset = dataset.train_test_split(test_size=self.val_split, seed=42)
            dataset_dict = datasets.DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
            if self.verbose:
                print(f"Train split: {len(dataset_dict['train'])} examples")
                print(f"Validation split: {len(dataset_dict['validation'])} examples")
        else:
            dataset_dict = datasets.DatasetDict({'train': dataset})
        
        return dataset_dict

    def save_dataset(self, dataset: datasets.DatasetDict, output_dir: str) -> None:
        """
        Save the dataset to disk in HuggingFace format.
        
        Args:
            dataset: Dataset to save
            output_dir: Directory to save the dataset
        """
        dataset.save_to_disk(output_dir)
        if self.verbose:
            print(f"Dataset saved to {output_dir}")

    def create_dataset_from_path(self, data_dir: Optional[str] = None) -> datasets.DatasetDict:
        """
        Create a complete processed dataset from the raw data directory.
        
        Args:
            data_dir: Override data directory (uses self.data_dir if None)
            
        Returns:
            Processed HuggingFace DatasetDict with train/validation splits
        """
        if data_dir:
            self.data_dir = Path(data_dir)
            self.en_dir = self.data_dir / "en"
            self.zh_dir = self.data_dir / "zh-hant"
        
        return self.preprocess()


def create_hk_legislation_dataset(
    data_dir: str = "/home/hether/uni/nlp2-26/data/hk_legislation",
    min_text_length: int = 10,
    **kwargs
) -> datasets.DatasetDict:
    """
    Convenience function to create HK Legislation dataset.
    
    Args:
        data_dir: Path to legislation data directory
        min_text_length: Minimum segment length
        **kwargs: Additional arguments for HKLegislationPreprocessor
        
    Returns:
        HuggingFace DatasetDict with train/validation splits
    """
    preprocessor = HKLegislationPreprocessor(
        data_dir=data_dir,
        min_text_length=min_text_length,
        **kwargs
    )
    return preprocessor.preprocess()


if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Preprocess HK Legislation XML corpus.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "hk_legislation",
        help="Root directory containing en/ and zh-hant/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "processed_datasets" / "hk_legislation",
        help="Output directory for HuggingFace dataset",
    )
    parser.add_argument("--min-text-length", type=int, default=50)
    parser.add_argument("--max-text-length", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    preprocessor = HKLegislationPreprocessor(
        data_dir=str(args.data_dir),
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        val_split=args.val_split,
        verbose=not args.quiet,
    )
    dataset_dict = preprocessor.preprocess()

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    total_examples = sum(len(ds) for ds in dataset_dict.values())
    print(f"Total examples: {total_examples}")

    if "train" in dataset_dict and len(dataset_dict["train"]) > 0:
        avg_en_tokens = sum(dataset_dict["train"]["num_tokens_source"]) / len(dataset_dict["train"])
        avg_zh_tokens = sum(dataset_dict["train"]["num_tokens_target"]) / len(dataset_dict["train"])
        print(f"Average English tokens: {avg_en_tokens:.1f}")
        print(f"Average Chinese tokens: {avg_zh_tokens:.1f}")
        print("\nFirst example:")
        print(f"  EN: {dataset_dict['train'][0]['source_text'][:200]}...")
        print(f"  ZH: {dataset_dict['train'][0]['target_text'][:200]}...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save_dataset(dataset_dict, str(args.output_dir))
