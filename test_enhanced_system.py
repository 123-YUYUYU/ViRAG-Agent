"""
Test script to verify the enhanced Visual-RAG system functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_basic_imports():
    """Test basic imports of new modules"""
    print("Testing basic imports...")

    try:
        from data_types import DocBlock
        print("- DocBlock import: OK")
    except ImportError as e:
        print(f"- DocBlock import: FAILED - {e}")

    try:
        from document_processor import parse_document_with_images
        print("- Document processor import: OK")
    except ImportError as e:
        print(f"- Document processor import: FAILED - {e}")

    try:
        from retrieval.hybrid_retriever import HybridRetriever
        print("- Hybrid retriever import: OK")
    except ImportError as e:
        print(f"- Hybrid retriever import: FAILED - {e}")

    try:
        from retrieval.reranker import Reranker
        print("- Reranker import: OK")
    except ImportError as e:
        print(f"- Reranker import: FAILED - {e}")

    try:
        from agent.react_agent import EvidenceAwareVQAAgent
        print("- EvidenceAwareVQAAgent import: OK")
    except ImportError as e:
        print(f"- EvidenceAwareVQAAgent import: FAILED - {e}")

def test_docblock_creation():
    """Test DocBlock creation"""
    print("\nTesting DocBlock creation...")

    try:
        from data_types import DocBlock

        # Test creating a text block
        text_block = DocBlock(
            id="test_1",
            content="This is a sample text content.",
            block_type="text",
            metadata={"page_num": 1, "source": "test.pdf", "caption": "Sample text"}
        )
        print(f"- Text block created: {text_block.id}")

        # Test creating an image block
        image_block = DocBlock(
            id="test_2",
            content="/path/to/image.png",
            block_type="image",
            metadata={"page_num": 2, "source": "test.pdf", "ocr_text": "sample OCR text"}
        )
        print(f"- Image block created: {image_block.id}")

        print("- DocBlock creation: OK")
    except Exception as e:
        print(f"- DocBlock creation: FAILED - {e}")

def test_main_modes():
    """Test main module modes"""
    print("\nTesting main module...")

    try:
        import main
        print("- Main module import: OK")

        # Check if required functions exist
        assert hasattr(main, 'check_dependencies'), "check_dependencies function missing"
        assert hasattr(main, 'start_evidence_aware_interface'), "start_evidence_aware_interface function missing"
        print("- Main functions available: OK")

    except Exception as e:
        print(f"- Main module test: FAILED - {e}")

def run_tests():
    """Run all tests"""
    print("Running tests for Enhanced Visual-RAG System...")
    print("="*50)

    test_basic_imports()
    test_docblock_creation()
    test_main_modes()

    print("\n" + "="*50)
    print("Tests completed. Note: Missing dependencies will be resolved by installing requirements.txt")

if __name__ == "__main__":
    run_tests()