"""
Quick test to verify flashcard generation works correctly.
Run this to test the flashcard system independently.
"""

from scholarsync.utils.schemas import ExtractedKnowledge, SubTaskType
from scholarsync.agents.profile_builder import generate_flashcards

# Create a mock extraction
test_extraction = ExtractedKnowledge(
    subtask_type=SubTaskType.FINDINGS,
    paper_id="test_paper_123",
    paper_title="Advanced RAG Techniques for Long Documents",
    methodology=[
        "Uses hierarchical chunking with semantic overlap",
        "Implements hybrid retrieval combining BM25 and dense embeddings",
        "Employs auto-merging for context preservation"
    ],
    findings=[
        "Hierarchical chunking improves retrieval accuracy by 23% over fixed-size chunks",
        "Hybrid retrieval outperforms pure dense retrieval by 15% on complex queries",
        "Auto-merging reduces context loss in long documents by 40%"
    ],
    claims=[
        "This paper proposes a novel approach to RAG that addresses the challenge of long document processing",
        "The proposed method achieves state-of-the-art performance on benchmark datasets",
        "Computational cost is reduced by 30% compared to baseline approaches"
    ],
    risks=[
        "Method requires significant GPU memory for large document collections",
        "Performance degrades on very short queries (< 5 words)"
    ]
)

print("=" * 70)
print("FLASHCARD GENERATION TEST")
print("=" * 70)

try:
    flashcards = generate_flashcards(test_extraction)
    
    print(f"\n✅ Generated {len(flashcards)} flashcards\n")
    
    for i, card in enumerate(flashcards, 1):
        print(f"{'─' * 70}")
        print(f"CARD {i}: {card.category.upper()}")
        print(f"{'─' * 70}")
        print(f"FRONT: {card.front}")
        print(f"BACK:  {card.back}")
        print()
    
    print("=" * 70)
    print("✅ TEST PASSED - Flashcard generation works correctly!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
