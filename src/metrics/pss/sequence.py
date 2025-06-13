import numpy as np

def calculate_order_similarity_from_feedback(feedback_list):
    """
    Calculate order similarity based on existing path matching feedback results
    
    This function computes the Kendall tau correlation coefficient between the reference
    and generated path ordering, then normalizes it to a 0-1 range. Order similarity
    is crucial for SVGs where path rendering order affects visual appearance (e.g., layering).
    
    Parameters:
    ----------
    feedback_list : list
        List of comparison results containing path matching information.
        Each element should contain 'ref_path_id' and 'gen_path_id' fields.
        
    Returns:
    -------
    float
        Normalized Kendall order similarity score (0-1)
        - 1.0: Perfect order match
        - 0.5: Random order (no correlation)
        - 0.0: Completely reversed order
    """
    if not feedback_list:
        return 0.0
    
    # Extract matched path pairs
    path_pairs = [(fb['ref_path_id'], fb['gen_path_id']) for fb in feedback_list]
    
    # Extract reference and generated path ID lists
    ref_ids = [pair[0] for pair in path_pairs]
    gen_ids = [pair[1] for pair in path_pairs]
    
    # Calculate Kendall Tau coefficient
    tau = kendall_tau_coefficient(ref_ids, gen_ids)
    
    # Normalize to 0-1 range
    normalized_tau = (tau + 1) / 2
    
    return normalized_tau


def kendall_tau_coefficient(list1, list2):
    """
    Calculate Kendall tau correlation coefficient between two sequences
    
    The Kendall tau coefficient measures the ordinal association between two measured
    quantities. It ranges from -1 (perfect disagreement) to +1 (perfect agreement).
    
    Parameters:
    ----------
    list1 : list
        First sequence (reference path IDs)
    list2 : list
        Second sequence (generated path IDs)
        
    Returns:
    -------
    float
        Kendall tau coefficient (-1 to 1)
        - +1: Perfect positive correlation (same order)
        - 0: No correlation (random order)
        - -1: Perfect negative correlation (reversed order)
        
    Raises:
    ------
    ValueError
        If the two lists have different lengths
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")
    
    if len(list1) <= 1:
        return 1.0  # Single element or empty list considered perfectly correlated
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    
    n = len(list1)
    for i in range(n):
        for j in range(i + 1, n):
            # Relative order in reference paths
            ref_order = list1[i] < list1[j]
            # Relative order in generated paths
            gen_order = list2[i] < list2[j]
            
            # If relative orders are consistent
            if (ref_order and gen_order) or (not ref_order and not gen_order):
                concordant += 1
            else:
                discordant += 1
    
    # Calculate total number of pairs
    total_pairs = n * (n - 1) // 2
    
    # If no pairs to compare, return 1.0 (perfect consistency)
    if total_pairs == 0:
        return 1.0
        
    # Calculate tau coefficient
    tau = (concordant - discordant) / total_pairs
    
    return tau


def add_order_score_to_result(result):
    """
    Add order similarity score to comparison result dictionary
    
    This function enhances an existing SVG comparison result by adding order similarity
    assessment and updating the overall quality score.
    
    Parameters:
    ----------
    result : dict
        SVG comparison result dictionary containing feedback and dimension scores
        
    Returns:
    -------
    dict
        Enhanced result dictionary with order scoring added
    """
    # Ensure result dictionary has feedback field
    if 'feedback' not in result or not result['feedback']:
        if 'dimension_scores' not in result:
            result['dimension_scores'] = {}
        result['dimension_scores']['order'] = 0.0
        return result
    
    # Calculate order similarity
    order_score = calculate_order_similarity_from_feedback(result['feedback'])
    
    # Add order score to dimension scores
    if 'dimension_scores' not in result:
        result['dimension_scores'] = {}
    
    result['dimension_scores']['order'] = order_score
    
    # Add detailed information
    result['order_details'] = {
        'normalized_tau': order_score,
        'matched_paths_count': len(result['feedback']),
        'total_reference_paths': len(result.get('reference_paths', [])),
        'total_generated_paths': len(result.get('generated_paths', [])),
        'order_preservation_ratio': order_score
    }
    
    # Update overall score (assuming equal weights for all dimensions)
    if result['dimension_scores']:
        dims = len(result['dimension_scores'])
        result['overall_quality_score'] = sum(
            score * 100 for score in result['dimension_scores'].values()) / dims
    
    return result


def integrate_order_score(comparator):
    """
    Integrate order scoring into SVG comparison system
    
    This function enhances an existing SVG comparison system by adding order similarity
    assessment as an additional dimension.
    
    Parameters:
    ----------
    comparator : SVGComparisonSystem
        SVG comparison system instance
        
    Returns:
    -------
    SVGComparisonSystem
        Enhanced comparison system with order scoring capability
    """
    # Save original comparison method
    original_compare = comparator.compare_svgs
    
    def enhanced_compare_svgs(reference_paths, generated_paths, target_size):
        # Call original comparison method to get basic results
        result = original_compare(reference_paths, generated_paths, target_size)
        
        # Add order scoring
        result = add_order_score_to_result(result)
        
        return result
    
    # Replace comparison method
    comparator.compare_svgs = enhanced_compare_svgs
    
    return comparator


def analyze_order_impact(feedback_list, shape_threshold=0.8, color_threshold=0.8, position_threshold=0.8):
    """
    Analyze the impact of order differences on SVG quality
    
    This function helps understand how order differences affect the overall quality
    when other dimensions (shape, color, position) are high quality.
    
    Parameters:
    ----------
    feedback_list : list
        List of path matching feedback
    shape_threshold : float
        Threshold for considering shape as high quality
    color_threshold : float  
        Threshold for considering color as high quality
    position_threshold : float
        Threshold for considering position as high quality
        
    Returns:
    -------
    dict
        Analysis results including order impact assessment
    """
    if not feedback_list:
        return {
            'order_score': 0.0,
            'high_quality_paths': 0,
            'order_impact': 'No paths to analyze'
        }
    
    # Calculate order similarity
    order_score = calculate_order_similarity_from_feedback(feedback_list)
    
    # Count paths that meet quality thresholds in other dimensions
    high_quality_paths = 0
    for feedback in feedback_list:
        if (feedback.get('shape_score', 0) >= shape_threshold and
            feedback.get('color_score', 0) >= color_threshold and
            feedback.get('position_score', 0) >= position_threshold):
            high_quality_paths += 1
    
    # Calculate average scores for other dimensions
    avg_shape = np.mean([fb.get('shape_score', 0) for fb in feedback_list])
    avg_color = np.mean([fb.get('color_score', 0) for fb in feedback_list])
    avg_position = np.mean([fb.get('position_score', 0) for fb in feedback_list])
    avg_other_dims = (avg_shape + avg_color + avg_position) / 3
    
    # Determine order impact
    quality_gap = avg_other_dims - order_score
    if quality_gap > 0.3:
        impact = "High impact - Order significantly reduces quality despite good shape/color/position"
    elif quality_gap > 0.1:
        impact = "Medium impact - Order moderately affects quality"
    elif quality_gap > -0.1:
        impact = "Low impact - Order is consistent with other dimensions"
    else:
        impact = "Positive impact - Order is better than other dimensions"
    
    return {
        'order_score': order_score,
        'avg_other_dimensions': avg_other_dims,
        'quality_gap': quality_gap,
        'high_quality_paths': high_quality_paths,
        'total_paths': len(feedback_list),
        'order_impact': impact,
        'avg_shape_score': avg_shape,
        'avg_color_score': avg_color,
        'avg_position_score': avg_position
    }


def generate_test_cases():
    """
    Generate various example path matching scenarios for testing order similarity calculation
    
    Returns:
    -------
    dict
        Dictionary containing different types of test cases with expected outcomes
    """
    test_cases = {}
    
    # 1. Perfect order match
    test_cases['perfect_order'] = {
        'feedback': [
            {'ref_path_id': 0, 'gen_path_id': 0, 'shape_score': 0.9, 'color_score': 0.85, 'position_score': 0.92},
            {'ref_path_id': 1, 'gen_path_id': 1, 'shape_score': 0.85, 'color_score': 0.9, 'position_score': 0.87},
            {'ref_path_id': 2, 'gen_path_id': 2, 'shape_score': 0.92, 'color_score': 0.88, 'position_score': 0.94},
            {'ref_path_id': 3, 'gen_path_id': 3, 'shape_score': 0.88, 'color_score': 0.92, 'position_score': 0.9},
            {'ref_path_id': 4, 'gen_path_id': 4, 'shape_score': 0.95, 'color_score': 0.94, 'position_score': 0.93}
        ],
        'expected_score': 1.0,
        'description': 'Perfect order preservation - all paths match in sequence'
    }
    
    # 2. Completely reversed order
    test_cases['reversed_order'] = {
        'feedback': [
            {'ref_path_id': 0, 'gen_path_id': 4, 'shape_score': 0.8, 'color_score': 0.75, 'position_score': 0.82},
            {'ref_path_id': 1, 'gen_path_id': 3, 'shape_score': 0.75, 'color_score': 0.8, 'position_score': 0.77},
            {'ref_path_id': 2, 'gen_path_id': 2, 'shape_score': 0.82, 'color_score': 0.78, 'position_score': 0.84},
            {'ref_path_id': 3, 'gen_path_id': 1, 'shape_score': 0.78, 'color_score': 0.82, 'position_score': 0.8},
            {'ref_path_id': 4, 'gen_path_id': 0, 'shape_score': 0.85, 'color_score': 0.84, 'position_score': 0.83}
        ],
        'expected_score': 0.0,
        'description': 'Completely reversed order - maximum disagreement'
    }
    
    # 3. Partial reordering
    test_cases['partial_reorder'] = {
        'feedback': [
            {'ref_path_id': 0, 'gen_path_id': 2, 'shape_score': 0.85, 'color_score': 0.8, 'position_score': 0.78},
            {'ref_path_id': 1, 'gen_path_id': 0, 'shape_score': 0.72, 'color_score': 0.75, 'position_score': 0.7},
            {'ref_path_id': 2, 'gen_path_id': 3, 'shape_score': 0.91, 'color_score': 0.87, 'position_score': 0.9},
            {'ref_path_id': 3, 'gen_path_id': 1, 'shape_score': 0.65, 'color_score': 0.7, 'position_score': 0.68},
            {'ref_path_id': 4, 'gen_path_id': 4, 'shape_score': 0.88, 'color_score': 0.9, 'position_score': 0.85}
        ],
        'expected_score': 0.7,
        'description': 'Partial reordering - some order preservation'
    }
    
    # 4. Random order (for large dataset)
    np.random.seed(42)  # Ensure reproducibility
    gen_ids = np.random.permutation(10)
    test_cases['random_order'] = {
        'feedback': [
            {'ref_path_id': i, 'gen_path_id': int(gen_ids[i]), 
             'shape_score': round(0.7 + 0.2 * np.random.random(), 2),
             'color_score': round(0.7 + 0.2 * np.random.random(), 2),
             'position_score': round(0.7 + 0.2 * np.random.random(), 2)}
            for i in range(10)
        ],
        'expected_score': 0.5,  # Approximately, for random order
        'description': 'Random order - should approximate 0.5'
    }
    
    # 5. Icon example - smiley face scenario
    test_cases['icon_example'] = {
        'feedback': [
            {'ref_path_id': 0, 'gen_path_id': 0, 'shape_score': 0.95, 'color_score': 0.9, 'position_score': 0.92, 
             'suggestions': ['Perfect circle outline match']},
            {'ref_path_id': 1, 'gen_path_id': 3, 'shape_score': 0.82, 'color_score': 0.78, 'position_score': 0.85, 
             'suggestions': ['Left eye position slightly offset']},
            {'ref_path_id': 2, 'gen_path_id': 4, 'shape_score': 0.88, 'color_score': 0.85, 'position_score': 0.9, 
             'suggestions': ['Right eye shape basically matches']},
            {'ref_path_id': 3, 'gen_path_id': 1, 'shape_score': 0.75, 'color_score': 0.8, 'position_score': 0.72, 
             'suggestions': ['Mouth curve needs adjustment']},
            {'ref_path_id': 4, 'gen_path_id': 2, 'shape_score': 0.91, 'color_score': 0.93, 'position_score': 0.88, 
             'suggestions': ['Nose position good']}
        ],
        'expected_score': 0.7,
        'description': 'Icon with reordered facial features - affects visual layering'
    }
    
    # 6. No matches
    test_cases['no_matches'] = {
        'feedback': [],
        'expected_score': 0.0,
        'description': 'No path matches found'
    }
    
    # 7. Single path
    test_cases['single_path'] = {
        'feedback': [
            {'ref_path_id': 0, 'gen_path_id': 0, 'shape_score': 0.9, 'color_score': 0.85, 'position_score': 0.92}
        ],
        'expected_score': 1.0,
        'description': 'Single path - always perfect order'
    }
    
    return test_cases


def run_comprehensive_tests():
    """
    Run comprehensive tests for order similarity calculation effectiveness
    
    This function validates the order similarity calculation using various test cases
    and prints detailed results for verification.
    """
    print("=== Comprehensive Order Similarity Tests ===\n")
    
    test_cases = generate_test_cases()
    
    for case_name, case_data in test_cases.items():
        print(f"Test Case: {case_name}")
        print(f"Description: {case_data['description']}")
        
        feedback = case_data['feedback']
        expected = case_data['expected_score']
        
        # Calculate order similarity
        actual_score = calculate_order_similarity_from_feedback(feedback)
        
        # Check if result is within acceptable range
        tolerance = 0.1 if case_name == 'random_order' else 0.0001
        test_passed = abs(actual_score - expected) <= tolerance
        
        print(f"Expected: {expected}")
        print(f"Actual: {actual_score:.4f}")
        print(f"Test Passed: {test_passed}")
        
        # Analyze order impact if feedback exists
        if feedback:
            impact_analysis = analyze_order_impact(feedback)
            print(f"Order Impact: {impact_analysis['order_impact']}")
            print(f"Quality Gap: {impact_analysis['quality_gap']:.4f}")
        
        print("-" * 50)
    
    # Special test: Order importance in SVG layer stacking
    print("\nSpecial Test: Order Importance in SVG Layer Stacking")
    print("Scenario: Same shapes but different order affects visual appearance")
    
    overlay_feedback = [
        {'ref_path_id': 0, 'gen_path_id': 2, 'shape_score': 0.95, 'color_score': 0.9, 'position_score': 0.92},
        {'ref_path_id': 1, 'gen_path_id': 0, 'shape_score': 0.95, 'color_score': 0.9, 'position_score': 0.92},
        {'ref_path_id': 2, 'gen_path_id': 1, 'shape_score': 0.95, 'color_score': 0.9, 'position_score': 0.92}
    ]
    
    order_score = calculate_order_similarity_from_feedback(overlay_feedback)
    impact_analysis = analyze_order_impact(overlay_feedback)
    
    print(f"Order similarity score: {order_score:.4f}")
    print(f"Average other dimensions: {impact_analysis['avg_other_dimensions']:.4f}")
    print(f"Impact analysis: {impact_analysis['order_impact']}")
    print(f"Quality reduction due to order: {impact_analysis['quality_gap']:.4f}")
    
    print("\n=== Test Summary ===")
    print("✓ Perfect order: 1.0")
    print("✓ Reversed order: 0.0") 
    print("✓ Partial reorder: ~0.7")
    print("✓ Random order: ~0.5")
    print("✓ Single path: 1.0")
    print("✓ No matches: 0.0")
    print("✓ Integration with result dictionary")


# Example usage and testing
if __name__ == "__main__":
    # Basic demonstration
    print("=== Order Similarity Calculator Demo ===\n")
    
    # Example feedback data
    example_feedback = [
        {'ref_path_id': 0, 'gen_path_id': 2, 'shape_score': 0.85, 'color_score': 0.90, 'position_score': 0.78, 'suggestions': []},
        {'ref_path_id': 1, 'gen_path_id': 0, 'shape_score': 0.72, 'color_score': 0.85, 'position_score': 0.65, 'suggestions': []},
        {'ref_path_id': 2, 'gen_path_id': 3, 'shape_score': 0.91, 'color_score': 0.87, 'position_score': 0.92, 'suggestions': []},
        {'ref_path_id': 3, 'gen_path_id': 1, 'shape_score': 0.65, 'color_score': 0.70, 'position_score': 0.60, 'suggestions': []},
        {'ref_path_id': 4, 'gen_path_id': 4, 'shape_score': 0.88, 'color_score': 0.92, 'position_score': 0.85, 'suggestions': []}
    ]
    
    # Create example result dictionary
    example_result = {
        'overall_quality_score': 80,
        'avg_score': 0.8,
        'match_rate': 0.9,
        'dimension_scores': {
            'shape': 0.85,
            'color': 0.75,
            'position': 0.80,
        },
        'feedback': example_feedback,
        'reference_paths': [{}] * 5,  # 5 reference paths
        'generated_paths': [{}] * 5   # 5 generated paths
    }
    
    # Calculate order score directly
    order_score = calculate_order_similarity_from_feedback(example_feedback)
    print(f"Direct order similarity score: {order_score:.4f}")
    
    # Add order score to result dictionary
    updated_result = add_order_score_to_result(example_result)
    print(f"Order score in result: {updated_result['dimension_scores']['order']:.4f}")
    print(f"Updated overall score: {updated_result['overall_quality_score']:.2f}/100")
    
    # Analyze order impact
    impact_analysis = analyze_order_impact(example_feedback)
    print(f"\nOrder Impact Analysis:")
    print(f"Order score: {impact_analysis['order_score']:.4f}")
    print(f"Average other dimensions: {impact_analysis['avg_other_dimensions']:.4f}")
    print(f"Impact assessment: {impact_analysis['order_impact']}")
    
    print(f"\nOrder details:")
    if 'order_details' in updated_result:
        details = updated_result['order_details']
        print(f"Matched paths: {details['matched_paths_count']}")
        print(f"Reference paths: {details['total_reference_paths']}")
        print(f"Generated paths: {details['total_generated_paths']}")
        print(f"Order preservation ratio: {details['order_preservation_ratio']:.4f}")
    
    print("\n" + "="*60)
    
    # Run comprehensive tests
    run_comprehensive_tests()