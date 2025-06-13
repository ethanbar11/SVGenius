#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def evaluate_path_count(reference_count, generated_count, asymmetric_penalty=True): 
    """
    Evaluates the similarity of the number of paths between two SVGs and returns a penalty score.
    The larger the difference in count, the larger the penalty.

    Parameters:
    ----------
    reference_count : int
        Number of paths in the reference SVG
    generated_count : int
        Number of paths in the generated SVG
    asymmetric_penalty : bool, optional
        Whether to use asymmetric penalty (missing paths are penalized more), default is True

    Returns:
    -------
    float
        Penalty score based on path count, range is 0-1, where 0 means perfect match (no penalty)
    """
    if reference_count == generated_count:
        return 0.0
    
    diff = abs(reference_count - generated_count)
    
    if reference_count == 0:
        if generated_count > 0:
            return 0.5  # Medium penalty if reference has no paths but generated has
        return 0.0  # Perfect match when both have no paths
    
    ratio = diff / reference_count
    base_penalty = min(1.0, ratio)
    
    if asymmetric_penalty:
        if generated_count < reference_count:
            penalty_factor = 1.5  # Higher penalty for missing paths
            penalty = min(1.0, base_penalty * penalty_factor)
        else:
            penalty_factor = 0.7  # Lighter penalty for extra paths
            penalty = min(1.0, base_penalty * penalty_factor)
    else:
        penalty = base_penalty
    
    return penalty


def add_path_count_score_to_result(result, reference_paths, generated_paths):
    """
    Adds the path count score to the comparison result dictionary.

    Parameters:
    ----------
    result : dict
        Dictionary holding the SVG comparison result
    reference_paths : list
        List of paths in the reference SVG
    generated_paths : list
        List of paths in the generated SVG
        
    Returns:
    -------
    dict
        Updated result dictionary with path count score added
    """
    ref_count = len(reference_paths)
    gen_count = len(generated_paths)
    
    count_penalty = evaluate_path_count(ref_count, gen_count)
    
    # Convert penalty to score (1 - penalty)
    count_score = 1.0 - count_penalty
    
    if 'dimension_scores' not in result:
        result['dimension_scores'] = {}
    
    result['dimension_scores']['path_count'] = count_score
    
    # Add detailed path count information
    result['path_count_details'] = {
        'reference_count': ref_count,
        'generated_count': gen_count,
        'difference': gen_count - ref_count,
        'penalty': count_penalty,
        'match_percentage': f"{(count_score * 100):.1f}%"
    }
    
    # Update overall score (assuming equal weight for all dimensions)
    dims = len(result['dimension_scores'])
    result['overall_quality_score'] = sum(
        score * 100 for score in result['dimension_scores'].values()) / dims
    
    return result


def main():
    """
    Tests the path count evaluation function under various scenarios
    """
    print("===== Testing Path Count Evaluation Function =====\n")
    
    test_cases = [
        (10, 10, "Perfect match"),
        (10, 8, "20% fewer paths"),
        (10, 5, "50% fewer paths"),
        (10, 12, "20% more paths"),
        (10, 15, "50% more paths"),
        (10, 20, "100% more paths"),
        (5, 2, "Severely missing paths"),
        (5, 10, "Excessive extra paths"),
        (1, 0, "Reference has paths but generated has none"),
        (0, 5, "Reference has no paths but generated has some"),
        (0, 0, "Neither has paths")
    ]
    
    for ref_count, gen_count, desc in test_cases:
        asym_penalty = evaluate_path_count(ref_count, gen_count, asymmetric_penalty=True)
        asym_score = 1.0 - asym_penalty
        
        sym_penalty = evaluate_path_count(ref_count, gen_count, asymmetric_penalty=False)
        sym_score = 1.0 - sym_penalty
        
        print(f"Test: {desc}")
        print(f"  Reference count: {ref_count}, Generated count: {gen_count}")
        print(f"  Asymmetric penalty: {asym_penalty:.4f} (Score: {asym_score:.4f})")
        print(f"  Symmetric penalty: {sym_penalty:.4f} (Score: {sym_score:.4f})")
        
        if ref_count > 0:
            percent_diff = ((gen_count - ref_count) / ref_count * 100)
            print(f"  Difference: {gen_count - ref_count} ({'+' if gen_count > ref_count else ''}{percent_diff:.1f}%)")
        else:
            if gen_count > 0:
                print(f"  Difference: +{gen_count} (∞%)")
            else:
                print(f"  Difference: 0 (0%)")
        print()
    
    try:
       
        
        reference_count = 10
        generated_counts = range(0, 21)
        
        asym_penalties = [evaluate_path_count(reference_count, gen_count, True) for gen_count in generated_counts]
        asym_scores = [1.0 - p for p in asym_penalties]
        
        sym_penalties = [evaluate_path_count(reference_count, gen_count, False) for gen_count in generated_counts]
        sym_scores = [1.0 - p for p in sym_penalties]
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(generated_counts, asym_penalties, 'r-', label='Asymmetric Penalty')
        plt.plot(generated_counts, sym_penalties, 'b--', label='Symmetric Penalty')
        
        plt.axvline(x=reference_count, color='g', linestyle=':', label='Reference Count')
        plt.ylabel('Penalty (0-1)')
        plt.title(f'Path Count Penalty Function (Reference count = {reference_count})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(generated_counts, asym_scores, 'r-', label='Asymmetric Score')
        plt.plot(generated_counts, sym_scores, 'b--', label='Symmetric Score')
        
        plt.axvline(x=reference_count, color='g', linestyle=':', label='Reference Count')
        plt.xlabel('Generated SVG Path Count')
        plt.ylabel('Score (0-1)')
        plt.title(f'Path Count Score Function (Score = 1 - Penalty)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('path_count_score.png')
        print("Score curve saved as 'path_count_score.png'")
        
    except ImportError:
        print("matplotlib not installed, skipping plot")
    
    print("\n===== Testing Adding Path Count Score to Result =====\n")
    
    mock_result = {
        'overall_quality_score': 75,
        'dimension_scores': {
            'shape': 0.8,
            'color': 0.7,
            'position': 0.75,
            'order': 0.7
        }
    }
    
    reference_paths = [f"path_{i}" for i in range(10)]
    generated_paths = [f"path_{i}" for i in range(8)]
    
    updated_result = add_path_count_score_to_result(mock_result, reference_paths, generated_paths)
    
    print("Updated Comparison Result:")
    print(f"  Path Count Score: {updated_result['dimension_scores']['path_count']:.4f}")
    print(f"  Path Count Details: {updated_result['path_count_details']}")
    print(f"  Updated Overall Quality Score: {updated_result['overall_quality_score']:.2f}")


if __name__ == "__main__":
    main()
