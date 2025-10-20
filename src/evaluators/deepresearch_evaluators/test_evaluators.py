"""
Test script to verify DeepResearch evaluators with structured outputs.

This script demonstrates the improved implementation with structured output support.
"""

import json
from src.evaluators.deepresearch_evaluators import (
    CitationRecallEvaluator,
    KeyPointRecallEvaluator,
    HolisticQualityEvaluator
)

def test_citation_recall():
    """Test citation recall evaluator with sample data."""
    print("\n" + "="*60)
    print("Testing Citation Recall Evaluator")
    print("="*60)
    
    evaluator = CitationRecallEvaluator(
        model="openai.gpt-oss-120b-1:0",
        num_threads=1,
        silent_errors=True
    )
    
    system_outputs = [
        {
            "iid": "test_1",
            "answer": """The Earth orbits around the Sun in an elliptical path [1]. 
            This orbital period takes approximately 365.25 days [2].
            The Earth also rotates on its axis every 24 hours [3].
            
            References:
            [1] https://nasa.gov/earth-orbit
            [2] https://space.com/orbital-period
            [3] https://astronomy.edu/rotation
            """,
            "citations": [
                "https://nasa.gov/earth-orbit",
                "https://space.com/orbital-period",
                "https://astronomy.edu/rotation"
            ]
        }
    ]
    
    references = [{"iid": "test_1", "query": "How does Earth move?"}]
    
    try:
        result = evaluator.evaluate(system_outputs, references)
        print(f"\n✓ Evaluator: {result.evaluator_name}")
        print(f"✓ Citation Recall Score: {result.metrics['citation_recall']:.2%}")
        print(f"✓ Total Claims: {result.metrics['total_claims']}")
        print(f"✓ Supported Claims: {result.metrics['total_supported_claims']}")
        print(f"✓ Evaluation Time: {result.total_time_ms:.2f}ms")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_key_point_recall():
    """Test key point recall evaluator with sample data."""
    print("\n" + "="*60)
    print("Testing Key Point Recall Evaluator")
    print("="*60)
    
    evaluator = KeyPointRecallEvaluator(
        model="openai.gpt-oss-120b-1:0",
        num_threads=1,
        silent_errors=True
    )
    
    system_outputs = [
        {
            "iid": "test_2",
            "answer": """Climate change is primarily driven by greenhouse gas emissions from human activities.
            The main contributors include carbon dioxide from burning fossil fuels, methane from agriculture,
            and deforestation which reduces carbon absorption. Rising temperatures are causing ice caps to melt
            and sea levels to rise."""
        }
    ]
    
    references = [
        {
            "iid": "test_2",
            "query": "What causes climate change?",
            "key_points": [
                {"point_number": 1, "point_content": "Greenhouse gases from human activities cause climate change"},
                {"point_number": 2, "point_content": "Carbon dioxide emissions from fossil fuels are a major contributor"},
                {"point_number": 3, "point_content": "Deforestation reduces carbon absorption capacity"},
                {"point_number": 4, "point_content": "Ocean acidification affects marine life"}
            ]
        }
    ]
    
    try:
        result = evaluator.evaluate(system_outputs, references)
        print(f"\n✓ Evaluator: {result.evaluator_name}")
        print(f"✓ Key Point Recall: {result.metrics['key_point_recall']:.2%}")
        print(f"✓ Support Rate: {result.metrics['avg_support_rate']:.2%}")
        print(f"✓ Omitted Rate: {result.metrics['avg_omitted_rate']:.2%}")
        print(f"✓ Contradicted Rate: {result.metrics['avg_contradicted_rate']:.2%}")
        print(f"✓ Evaluation Time: {result.total_time_ms:.2f}ms")
        
        # Show detailed results
        if result.rows:
            print("\nDetailed Results:")
            row = result.rows[0]
            print(f"  - Supported: {row['supported_count']}/{row['total_key_points']}")
            print(f"  - Omitted: {row['omitted_count']}/{row['total_key_points']}")
            print(f"  - Contradicted: {row['contradicted_count']}/{row['total_key_points']}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_holistic_quality():
    """Test holistic quality evaluator with sample data."""
    print("\n" + "="*60)
    print("Testing Holistic Quality Evaluator")
    print("="*60)
    
    evaluator = HolisticQualityEvaluator(
        model="openai.gpt-oss-120b-1:0",
        num_threads=1,
        silent_errors=True
    )
    
    system_outputs = [
        {
            "iid": "test_3",
            "answer": """# The Impact of Artificial Intelligence on Healthcare

## Introduction
Artificial intelligence (AI) is revolutionizing healthcare through various applications
including diagnosis, treatment planning, and drug discovery.

## Diagnostic Applications
AI systems can analyze medical images with remarkable accuracy, often matching or exceeding
human radiologists in detecting conditions like cancer and diabetic retinopathy.

## Treatment Optimization
Machine learning algorithms help personalize treatment plans by analyzing patient data
and predicting outcomes for different therapeutic approaches.

## Challenges
However, concerns exist around data privacy, algorithmic bias, and the need for
human oversight in medical decision-making.

## Conclusion
While AI offers tremendous potential, careful implementation and regulation are essential
to ensure patient safety and ethical use.
"""
        }
    ]
    
    references = [
        {
            "iid": "test_3",
            "query": "How is AI impacting healthcare?"
        }
    ]
    
    try:
        result = evaluator.evaluate(system_outputs, references)
        print(f"\n✓ Evaluator: {result.evaluator_name}")
        print(f"✓ Overall Quality Score: {result.metrics['overall_quality']:.2f}/10")
        
        print("\nCriterion Scores:")
        for criterion in ['Clarity', 'Depth', 'Balance', 'Breadth', 'Support', 'Insightfulness']:
            score = result.metrics.get(f'avg_{criterion.lower()}', 0)
            print(f"  - {criterion}: {score:.2f}/10")
        
        print(f"\n✓ Evaluation Time: {result.total_time_ms:.2f}ms")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DeepResearch Evaluators - Test Suite")
    print("Testing implementation with structured output support")
    print("="*60)
    
    results = {
        "Citation Recall": test_citation_recall(),
        "Key Point Recall": test_key_point_recall(),
        "Holistic Quality": test_holistic_quality()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation is working correctly.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
