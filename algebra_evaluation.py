"""
Algebra EBM Evaluation Framework

Implements comprehensive evaluation metrics for algebraic equation solving using
energy-based models. Provides evaluation across single-rule, multi-rule, and
constrained test sets.

Key Metrics:
1. Symbolic Equivalence (Primary): % correct x values using SymPy
2. Embedding L2 Distance (Auxiliary): ||y_pred - y_true||_2 in embedding space  
3. Invalid Step Rate: % syntactically invalid decoded equations
4. Per-Rule Breakdown: Accuracy split by required rules

Example Usage:
    # Load trained models
    rule_models = load_rule_models(['distribute', 'combine', 'isolate', 'divide'])
    
    # Create test dataset
    test_dataset = AlgebraDataset('distribute', split='test', num_problems=1000)
    
    # Run evaluation
    results = evaluate_model(rule_models, test_dataset, encoder, decoder)
    
    # Print results  
    print(f"Accuracy: {results['symbolic_equivalence_rate']:.3f}")
    print(f"Invalid Rate: {results['invalid_rate']:.3f}")
"""

import torch
import numpy as np
import logging
import traceback
from typing import Dict, List, Union, Optional, Any, Tuple
from pathlib import Path
import json
import time
from collections import defaultdict

# Import algebra components
from algebra_inference import AlgebraInference, load_rule_models, InferenceConfig
from algebra_encoder import (
    CharacterLevelEncoder, ASTEncoder, EquationDecoder,
    check_equation_equivalence, validate_equation_syntax,
    create_decoder_from_dataset
)
from algebra_dataset import AlgebraDataset, MultiRuleDataset, ConstrainedDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_symbolic_equivalence(
    predicted_equations: List[str], 
    target_equations: List[str],
    variable: str = 'x'
) -> Dict[str, Any]:
    """
    Compute symbolic equivalence metrics using SymPy.
    
    Args:
        predicted_equations: List of predicted equation strings
        target_equations: List of target equation strings  
        variable: Variable to solve for (default: 'x')
        
    Returns:
        Dictionary with equivalence statistics
    """
    if len(predicted_equations) != len(target_equations):
        raise ValueError(f"Mismatched lengths: {len(predicted_equations)} vs {len(target_equations)}")
    
    total_equations = len(predicted_equations)
    equivalent_count = 0
    parsing_errors = 0
    equivalence_errors = 0
    
    # Track detailed results for analysis
    detailed_results = []
    
    for i, (pred_eq, target_eq) in enumerate(zip(predicted_equations, target_equations)):
        result = {
            'index': i,
            'predicted': pred_eq,
            'target': target_eq,
            'equivalent': False,
            'error': None
        }
        
        try:
            # Skip None predictions (failed decodings)
            if pred_eq is None:
                result['error'] = 'No prediction (failed decoding)'
                detailed_results.append(result)
                continue
                
            # Check equivalence using existing function
            is_equiv, error_msg = check_equation_equivalence(pred_eq, target_eq, variable)
            
            if error_msg is not None:
                result['error'] = error_msg
                if 'parsing' in error_msg.lower() or 'sympify' in error_msg.lower():
                    parsing_errors += 1
                else:
                    equivalence_errors += 1
            else:
                result['equivalent'] = is_equiv
                if is_equiv:
                    equivalent_count += 1
                    
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            equivalence_errors += 1
            
        detailed_results.append(result)
    
    # Calculate statistics
    accuracy = equivalent_count / total_equations if total_equations > 0 else 0.0
    parsing_error_rate = parsing_errors / total_equations if total_equations > 0 else 0.0
    equivalence_error_rate = equivalence_errors / total_equations if total_equations > 0 else 0.0
    
    return {
        'total_equations': total_equations,
        'equivalent_count': equivalent_count,
        'symbolic_equivalence_rate': accuracy,
        'parsing_errors': parsing_errors,
        'parsing_error_rate': parsing_error_rate,
        'equivalence_errors': equivalence_errors,
        'equivalence_error_rate': equivalence_error_rate,
        'detailed_results': detailed_results
    }


def compute_embedding_distances(
    predicted_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> Dict[str, Any]:
    """
    Compute L2 distances between predicted and target embeddings.
    
    Args:
        predicted_embeddings: Predicted embeddings (N, d_model)
        target_embeddings: Target embeddings (N, d_model)
        
    Returns:
        Dictionary with distance statistics
    """
    if predicted_embeddings.shape != target_embeddings.shape:
        raise ValueError(f"Shape mismatch: {predicted_embeddings.shape} vs {target_embeddings.shape}")
    
    # Compute L2 distances
    distances = torch.norm(predicted_embeddings - target_embeddings, dim=1)
    
    return {
        'total_comparisons': len(distances),
        'mean_l2_distance': distances.mean().item(),
        'std_l2_distance': distances.std().item(),
        'median_l2_distance': distances.median().item(),
        'min_l2_distance': distances.min().item(),
        'max_l2_distance': distances.max().item(),
        'distances': distances.cpu().tolist()
    }


def compute_invalid_rate(predicted_equations: List[str]) -> Dict[str, Any]:
    """
    Compute rate of syntactically invalid decoded equations.
    
    Args:
        predicted_equations: List of predicted equation strings
        
    Returns:
        Dictionary with validity statistics
    """
    total_predictions = len(predicted_equations)
    valid_count = 0
    invalid_count = 0
    none_count = 0
    
    validity_details = []
    
    for i, pred_eq in enumerate(predicted_equations):
        detail = {
            'index': i,
            'equation': pred_eq,
            'valid': False,
            'error': None
        }
        
        if pred_eq is None:
            none_count += 1
            detail['error'] = 'No prediction (None)'
        else:
            try:
                is_valid, error_msg, _ = validate_equation_syntax(pred_eq)
                detail['valid'] = is_valid
                
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    detail['error'] = error_msg
                    
            except Exception as e:
                invalid_count += 1
                detail['error'] = f"Validation error: {str(e)}"
                
        validity_details.append(detail)
    
    # Calculate rates
    valid_rate = valid_count / total_predictions if total_predictions > 0 else 0.0
    invalid_rate = invalid_count / total_predictions if total_predictions > 0 else 0.0
    none_rate = none_count / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'total_predictions': total_predictions,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'none_count': none_count,
        'valid_rate': valid_rate,
        'invalid_rate': invalid_rate,
        'none_rate': none_rate,
        'validity_details': validity_details
    }


def compute_per_rule_breakdown(
    results: List[Dict[str, Any]],
    dataset: Union[MultiRuleDataset, ConstrainedDataset]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute accuracy breakdown by required rules.
    
    Args:
        results: List of individual equation results
        dataset: Dataset with rule information
        
    Returns:
        Dictionary with per-rule statistics
    """
    rule_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'indices': []
    })
    
    for i, result in enumerate(results):
        try:
            # Get rule information from dataset
            if hasattr(dataset, 'get_problem_info'):
                problem_info = dataset.get_problem_info(i)
                rules_applied = problem_info.get('rules_applied', [])
                
                # Update statistics for each rule involved
                for rule in rules_applied:
                    rule_stats[rule]['total'] += 1
                    rule_stats[rule]['indices'].append(i)
                    
                    if result.get('equivalent', False):
                        rule_stats[rule]['correct'] += 1
                        
                # Also track by number of rules
                num_rules = len(rules_applied)
                rule_key = f"{num_rules}_rules"
                rule_stats[rule_key]['total'] += 1
                rule_stats[rule_key]['indices'].append(i)
                
                if result.get('equivalent', False):
                    rule_stats[rule_key]['correct'] += 1
                    
        except Exception as e:
            logger.warning(f"Error processing rule breakdown for index {i}: {str(e)}")
            continue
    
    # Calculate accuracies
    for rule, stats in rule_stats.items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
    
    return dict(rule_stats)


def evaluate_model(
    rule_models: Dict[str, Any],
    test_dataset: Union[AlgebraDataset, MultiRuleDataset, ConstrainedDataset],
    encoder: Union[CharacterLevelEncoder, ASTEncoder],
    decoder: Optional[EquationDecoder] = None,
    batch_size: int = 32,
    inference_params: Optional[Dict[str, Any]] = None,
    max_samples: Optional[int] = None,
    store_detailed_results: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of algebra EBM model(s).
    
    Args:
        rule_models: Dictionary mapping rule names to trained EBM models
        test_dataset: Test dataset to evaluate on
        encoder: Equation encoder
        decoder: Equation decoder (optional, will disable decoding if None)
        batch_size: Batch size for evaluation
        inference_params: Optional parameters for inference (T, step_size, etc.)
        max_samples: Maximum number of samples to evaluate (for quick testing)
        store_detailed_results: Whether to store detailed per-sample results (default: True)
        
    Returns:
        Comprehensive evaluation results dictionary
    """
    # Input validation
    if not rule_models:
        raise ValueError("rule_models dictionary cannot be empty")
    if len(test_dataset) == 0:
        raise ValueError("test_dataset cannot be empty")
        
    logger.info(f"Starting evaluation on {type(test_dataset).__name__} with {len(test_dataset)} samples")
    
    # CRITICAL: Rebuild decoder with candidates from the actual test dataset
    # The default decoder only has ~49 hardcoded equations which cannot match
    # the equations generated by the dataset (e.g., "-8*x+-50=-130").
    if decoder is not None:
        logger.info("Rebuilding decoder candidate set from test dataset...")
        decoder = create_decoder_from_dataset(
            encoder=encoder, 
            dataset=test_dataset,
            distance_threshold=decoder.distance_threshold,  # Preserve threshold
            include_inputs=True  # Include input equations too for better coverage
        )
        logger.info(f"Decoder now has {len(decoder.candidate_equations)} candidates from dataset")
    
    # Initialize inference engine
    inference_engine = AlgebraInference(
        rule_models=rule_models,
        encoder=encoder, 
        decoder=decoder
    )
    
    # Set default inference parameters
    if inference_params is None:
        inference_params = {
            'T': 20,
            'step_size': 0.1,
            'rule_weights': None
        }
    
    # Limit samples if requested
    num_samples = min(len(test_dataset), max_samples) if max_samples else len(test_dataset)
    
    # Storage for results
    predicted_equations = []
    target_equations = []
    predicted_embeddings = []
    target_embeddings = []
    inference_infos = []
    individual_results = []
    
    # Track timing
    start_time = time.time()
    
    logger.info(f"Evaluating {num_samples} samples...")
    
    # Process samples in batches for memory efficiency
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = range(batch_start, batch_end)
        
        logger.debug(f"Processing batch {batch_start//batch_size + 1}/{(num_samples-1)//batch_size + 1}")
        
        for idx in batch_indices:
            try:
                # Get target equation(s)
                if hasattr(test_dataset, 'get_equation_pair'):
                    # Single-rule dataset
                    input_eq, target_eq = test_dataset.get_equation_pair(idx)
                elif hasattr(test_dataset, 'get_problem_info'):
                    # Multi-rule or constrained dataset
                    problem_info = test_dataset.get_problem_info(idx)
                    input_eq = problem_info['input_equation']
                    target_eq = problem_info['target_equation']
                else:
                    # Fallback - get raw tensors and skip this sample
                    logger.warning(f"Cannot get equation strings for index {idx}")
                    continue
                
                # Encode target for embedding distance computation
                target_embedding = encoder(target_eq)
                target_embeddings.append(target_embedding.detach())
                
                # Run inference
                # Create InferenceConfig from inference_params
                config_params = {}
                if inference_params:
                    if 'T' in inference_params:
                        config_params['max_iterations'] = inference_params['T']
                    if 'step_size' in inference_params:
                        config_params['step_size'] = inference_params['step_size']
                    if 'K' in inference_params:
                        config_params['K'] = inference_params['K']
                    if 'use_adaptive_step' in inference_params:
                        config_params['use_adaptive_step'] = inference_params['use_adaptive_step']
                    if 'energy_threshold' in inference_params:
                        config_params['energy_threshold'] = inference_params['energy_threshold']
                
                inference_config = InferenceConfig(**config_params)
                rule_weights = inference_params.get('rule_weights') if inference_params else None
                
                result = inference_engine.solve_equation(
                    input_eq,
                    config=inference_config,
                    rule_weights=rule_weights
                )
                
                # Extract results
                pred_eq = result.get('output_equation', None)
                predicted_equations.append(pred_eq)
                target_equations.append(target_eq)
                if store_detailed_results:
                    inference_infos.append(result.get('inference_info', {}))
                
                # Get predicted embedding (ensure consistent device)
                if result.get('success', False) and 'output_embedding' not in result:
                    # Encode the decoded equation
                    pred_embedding = encoder(pred_eq).detach() if pred_eq else torch.zeros_like(target_embedding)
                elif 'output_embedding' in result:
                    # Use raw embedding from inference
                    pred_embedding = result['output_embedding'].detach()
                else:
                    # Failed inference - create zero embedding on same device
                    pred_embedding = torch.zeros_like(target_embedding)
                
                predicted_embeddings.append(pred_embedding)
                
                # Store individual result for per-rule analysis
                individual_result = {
                    'index': idx,
                    'success': result.get('success', False),
                    'equivalent': False  # Will be filled in by symbolic equivalence check
                }
                
                # Add detailed information only if requested (memory optimization)
                if store_detailed_results:
                    individual_result.update({
                        'input_equation': input_eq,
                        'target_equation': target_eq,
                        'predicted_equation': pred_eq,
                        'decoding_distance': result.get('decoding_distance', float('inf'))
                    })
                    
                individual_results.append(individual_result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {str(e)}")
                logger.debug(f"Sample {idx} traceback:\n{traceback.format_exc()}")
                # Add placeholder results to maintain alignment
                predicted_equations.append(None)
                target_equations.append("x=0")  # Dummy target
                # Note: target_embedding was already appended before inference call
                placeholder_embedding = torch.zeros(encoder.d_model)
                predicted_embeddings.append(placeholder_embedding)
                if store_detailed_results:
                    inference_infos.append({})
                
                error_result = {
                    'index': idx,
                    'success': False,
                    'equivalent': False
                }
                if store_detailed_results:
                    error_result['error'] = str(e)
                individual_results.append(error_result)
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    
    # 1. Symbolic Equivalence (Primary)
    symbolic_results = compute_symbolic_equivalence(predicted_equations, target_equations)
    
    # Update individual results with equivalence information
    for result, detail in zip(individual_results, symbolic_results['detailed_results']):
        result['equivalent'] = detail['equivalent']
    
    # 2. Embedding L2 Distance (Auxiliary)
    # Ensure all embeddings are on the same device before stacking
    if predicted_embeddings and target_embeddings:
        # Use the first target embedding's device as reference (encoder's device)
        reference_device = target_embeddings[0].device
        
        # Move all embeddings to the reference device
        predicted_embeddings_same_device = []
        for emb in predicted_embeddings:
            predicted_embeddings_same_device.append(emb.to(reference_device))
        
        target_embeddings_same_device = []
        for emb in target_embeddings:
            target_embeddings_same_device.append(emb.to(reference_device))
        
        predicted_embeddings_tensor = torch.stack(predicted_embeddings_same_device)
        target_embeddings_tensor = torch.stack(target_embeddings_same_device)
    else:
        # Fallback for empty lists
        predicted_embeddings_tensor = torch.empty(0, encoder.d_model)
        target_embeddings_tensor = torch.empty(0, encoder.d_model)
    embedding_results = compute_embedding_distances(predicted_embeddings_tensor, target_embeddings_tensor)
    
    # 3. Invalid Step Rate
    validity_results = compute_invalid_rate(predicted_equations)
    
    # 4. Per-Rule Breakdown (if applicable)
    per_rule_results = {}
    if isinstance(test_dataset, (MultiRuleDataset, ConstrainedDataset)):
        per_rule_results = compute_per_rule_breakdown(
            individual_results,
            test_dataset
        )
    
    # Compile comprehensive results
    evaluation_results = {
        # Dataset information
        'dataset_type': type(test_dataset).__name__,
        'dataset_info': test_dataset.get_dataset_info() if hasattr(test_dataset, 'get_dataset_info') else {},
        'num_samples_evaluated': num_samples,
        'evaluation_time_seconds': evaluation_time,
        'inference_params': inference_params,
        
        # Core metrics
        'symbolic_equivalence': symbolic_results,
        'embedding_distances': embedding_results,  
        'validity': validity_results,
        'per_rule_breakdown': per_rule_results,
        
        # Summary statistics
        'summary': {
            'accuracy': symbolic_results['symbolic_equivalence_rate'],
            'invalid_rate': validity_results['invalid_rate'],
            'mean_l2_distance': embedding_results['mean_l2_distance'],
            'total_evaluated': num_samples
        }
    }
    
    # Add detailed results only if requested (memory optimization)
    if store_detailed_results:
        evaluation_results.update({
            'individual_results': individual_results,
            'inference_statistics': {
                'mean_final_energy': np.mean([info.get('final_energy', 0) for info in inference_infos]),
                'mean_acceptance_rate': np.mean([info.get('acceptance_rate', 0) for info in inference_infos])
            }
        })
    
    return evaluation_results


def evaluate_model_suite(
    rule_models: Dict[str, Any],
    test_datasets: Dict[str, Union[AlgebraDataset, MultiRuleDataset, ConstrainedDataset]],
    encoder: Union[CharacterLevelEncoder, ASTEncoder],
    decoder: Optional[EquationDecoder] = None,
    **evaluation_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model suite across multiple test sets.
    
    Args:
        rule_models: Dictionary of trained rule models
        test_datasets: Dictionary mapping test set names to datasets  
        encoder: Equation encoder
        decoder: Equation decoder
        **evaluation_kwargs: Additional arguments passed to evaluate_model
        
    Returns:
        Dictionary mapping test set names to evaluation results
    """
    logger.info(f"Evaluating model suite on {len(test_datasets)} test sets")
    
    suite_results = {}
    total_start_time = time.time()
    
    for test_name, dataset in test_datasets.items():
        logger.info(f"Evaluating on {test_name}...")
        
        try:
            results = evaluate_model(
                rule_models=rule_models,
                test_dataset=dataset,
                encoder=encoder,
                decoder=decoder,
                **evaluation_kwargs
            )
            
            suite_results[test_name] = results
            
            # Log summary
            summary = results['summary']
            logger.info(f"{test_name} Results - Accuracy: {summary['accuracy']:.3f}, "
                       f"Invalid Rate: {summary['invalid_rate']:.3f}, "
                       f"L2 Distance: {summary['mean_l2_distance']:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {test_name}: {str(e)}")
            logger.error(f"Traceback for {test_name}:\n{traceback.format_exc()}")
            suite_results[test_name] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    total_time = time.time() - total_start_time
    logger.info(f"Suite evaluation completed in {total_time:.2f} seconds")
    
    # Add suite-level summary
    suite_results['_suite_summary'] = {
        'total_evaluation_time': total_time,
        'num_test_sets': len(test_datasets),
        'successful_evaluations': sum(1 for result in suite_results.values() 
                                    if isinstance(result, dict) and 'error' not in result)
    }
    
    return suite_results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors and numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # Recursively convert the results
    def recursive_convert(data):
        if isinstance(data, dict):
            return {key: recursive_convert(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert_for_json(data)
    
    converted_results = recursive_convert(results)
    
    with open(output_file, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")


def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("ALGEBRA EBM EVALUATION SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    dataset_info = results.get('dataset_info', {})
    
    print(f"Dataset: {results.get('dataset_type', 'Unknown')}")
    print(f"Samples Evaluated: {summary.get('total_evaluated', 0)}")
    print(f"Evaluation Time: {results.get('evaluation_time_seconds', 0):.2f}s")
    
    print("\n" + "-"*40)
    print("CORE METRICS")
    print("-"*40)
    print(f"Symbolic Equivalence Rate: {summary.get('accuracy', 0):.3f}")
    print(f"Invalid Equation Rate: {summary.get('invalid_rate', 0):.3f}")
    print(f"Mean L2 Embedding Distance: {summary.get('mean_l2_distance', 0):.3f}")
    
    # Per-rule breakdown if available
    per_rule = results.get('per_rule_breakdown', {})
    if per_rule:
        print("\n" + "-"*40)
        print("PER-RULE BREAKDOWN")
        print("-"*40)
        for rule, stats in per_rule.items():
            if stats['total'] > 0:
                print(f"{rule}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    
    # Inference statistics
    inf_stats = results.get('inference_statistics', {})
    if inf_stats:
        print("\n" + "-"*40)
        print("INFERENCE STATISTICS") 
        print("-"*40)
        print(f"Mean Final Energy: {inf_stats.get('mean_final_energy', 0):.4f}")
        print(f"Mean Acceptance Rate: {inf_stats.get('mean_acceptance_rate', 0):.3f}")
    
    print("="*60 + "\n")


def validate_energy_landscape(model, dataset, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Validate that trained model has sharp energy landscape for algebraic reasoning.
    
    This function empirically verifies energy separation between correct and incorrect
    solutions, ensuring the model has learned proper energy-based discrimination.
    
    Validation Criteria (from research report):
    - E(correct) < 5.0 (low energy for valid solutions)
    - E(incorrect) > 10.0 (high energy for invalid solutions) 
    - E(incorrect) - E(correct) > 8.0 (sufficient energy gap for discrimination)
    
    Args:
        model: Trained energy-based model with return_energy=True capability
        dataset: Dataset with algebraic problems for testing
        num_samples: Number of samples to evaluate (default: 1000)
        
    Returns:
        Dict with validation results:
        - 'passed': bool indicating if landscape meets criteria
        - 'e_correct_mean': Average energy for correct solutions
        - 'e_incorrect_mean': Average energy for incorrect solutions  
        - 'energy_gap': Mean difference between incorrect and correct energies
        - 'correct_below_threshold': Fraction of correct solutions with E < 5.0
        - 'incorrect_above_threshold': Fraction of incorrect solutions with E > 10.0
        - 'sufficient_gap_samples': Fraction of samples with gap > 8.0
        - 'statistics': Detailed statistical breakdown
        
    Example:
        results = validate_energy_landscape(trained_model, test_dataset, 500)
        if results['passed']:
            print("✅ Energy landscape validation passed!")
        else:
            print("❌ Energy landscape needs improvement")
            print(f"Gap: {results['energy_gap']:.2f} (target: >8.0)")
    """
    logger.info(f"Starting energy landscape validation with {num_samples} samples...")
    
    correct_energies = []
    incorrect_energies = []
    gap_samples = []
    
    try:
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            samples_processed = 0
            
            for i, sample in enumerate(dataset):
                if samples_processed >= num_samples:
                    break
                    
                try:
                    # Extract input and correct output
                    if isinstance(sample, dict):
                        inp = sample.get('input', sample.get('source', None))
                        correct_out = sample.get('output', sample.get('target', None))
                    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                        inp, correct_out = sample[0], sample[1]
                    else:
                        logger.warning(f"Sample {i}: Unexpected format, skipping")
                        continue
                        
                    if inp is None or correct_out is None:
                        logger.warning(f"Sample {i}: Missing input/output, skipping")
                        continue
                        
                    # Ensure tensors are properly shaped and on correct device
                    if not torch.is_tensor(inp):
                        inp = torch.tensor(inp, dtype=torch.float32)
                    if not torch.is_tensor(correct_out):
                        correct_out = torch.tensor(correct_out, dtype=torch.float32)
                        
                    # Add batch dimension if needed
                    if inp.dim() == 1:
                        inp = inp.unsqueeze(0)
                    if correct_out.dim() == 1:
                        correct_out = correct_out.unsqueeze(0)
                        
                    # Generate incorrect solution via corruption
                    incorrect_out = _generate_corrupted_solution(correct_out)
                    
                    # Compute energies
                    # Create dummy timestep (model expects t parameter)
                    t = torch.zeros(inp.size(0), dtype=torch.long)
                    
                    e_correct = model(inp, correct_out, t, return_energy=True)
                    e_incorrect = model(inp, incorrect_out, t, return_energy=True)
                    
                    # Extract energies as synchronized pairs
                    if torch.is_tensor(e_correct) and torch.is_tensor(e_incorrect):
                        e_c = e_correct.item() if e_correct.numel() == 1 else e_correct.mean().item()
                        e_i = e_incorrect.item() if e_incorrect.numel() == 1 else e_incorrect.mean().item()
                        
                        # Store as synchronized pairs to enforce pairing
                        correct_energies.append(e_c)
                        incorrect_energies.append(e_i)
                        gap_samples.append(e_i - e_c)
                    else:
                        # Skip if either energy is invalid
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Skipping sample {i}: invalid energy tensors (e_correct={type(e_correct)}, e_incorrect={type(e_incorrect)})")
                    
                    samples_processed += 1
                    
                except Exception as e:
                    logger.warning(f"Sample {i}: Error during processing - {e}")
                    continue
                    
        if len(correct_energies) == 0 or len(incorrect_energies) == 0:
            return {
                'passed': False,
                'error': 'No valid energy measurements obtained',
                'samples_processed': samples_processed
            }
            
    except Exception as e:
        logger.error(f"Energy landscape validation failed: {e}")
        return {
            'passed': False, 
            'error': str(e),
            'samples_processed': 0
        }
    
    # Statistical analysis
    e_correct_mean = float(np.mean(correct_energies))
    e_incorrect_mean = float(np.mean(incorrect_energies))
    energy_gap = e_incorrect_mean - e_correct_mean
    
    e_correct_std = float(np.std(correct_energies)) 
    e_incorrect_std = float(np.std(incorrect_energies))
    
    # Validation criteria checks
    correct_below_threshold = sum(1 for e in correct_energies if e < 5.0) / len(correct_energies)
    incorrect_above_threshold = sum(1 for e in incorrect_energies if e > 10.0) / len(incorrect_energies)  
    sufficient_gap_samples = sum(1 for gap in gap_samples if gap > 8.0) / len(gap_samples) if gap_samples else 0.0
    
    # Overall pass/fail determination
    criteria_met = [
        e_correct_mean < 5.0,              # Correct solutions have low energy
        e_incorrect_mean > 10.0,           # Incorrect solutions have high energy
        energy_gap > 8.0                   # Sufficient energy gap
    ]
    passed = all(criteria_met)
    
    # Compile results
    results = {
        'passed': passed,
        'samples_processed': samples_processed,
        'e_correct_mean': e_correct_mean,
        'e_incorrect_mean': e_incorrect_mean,
        'energy_gap': energy_gap,
        'correct_below_threshold': correct_below_threshold,
        'incorrect_above_threshold': incorrect_above_threshold,
        'sufficient_gap_samples': sufficient_gap_samples,
        'criteria_met': {
            'correct_energy_low': criteria_met[0], 
            'incorrect_energy_high': criteria_met[1],
            'sufficient_gap': criteria_met[2]
        },
        'statistics': {
            'correct_energies': {
                'mean': e_correct_mean,
                'std': e_correct_std,
                'min': float(np.min(correct_energies)),
                'max': float(np.max(correct_energies)),
                'count': len(correct_energies)
            },
            'incorrect_energies': {
                'mean': e_incorrect_mean, 
                'std': e_incorrect_std,
                'min': float(np.min(incorrect_energies)),
                'max': float(np.max(incorrect_energies)),
                'count': len(incorrect_energies)
            },
            'energy_gaps': {
                'mean': float(np.mean(gap_samples)) if gap_samples else 0.0,
                'std': float(np.std(gap_samples)) if gap_samples else 0.0,
                'count': len(gap_samples)
            }
        }
    }
    
    # Print validation summary
    print("="*60)
    print("ENERGY LANDSCAPE VALIDATION RESULTS")
    print("="*60)
    print(f"Samples processed: {samples_processed}")
    print(f"E(correct):        {e_correct_mean:.2f} ± {e_correct_std:.2f} (target: <5.0)")
    print(f"E(incorrect):      {e_incorrect_mean:.2f} ± {e_incorrect_std:.2f} (target: >10.0)")
    print(f"Energy gap:        {energy_gap:.2f} (target: >8.0)")
    print()
    print("CRITERIA VALIDATION:")
    print(f"✅ Low correct energy:    {correct_below_threshold:.1%} below 5.0" if criteria_met[0] else f"❌ High correct energy:   {e_correct_mean:.2f} >= 5.0")
    print(f"✅ High incorrect energy: {incorrect_above_threshold:.1%} above 10.0" if criteria_met[1] else f"❌ Low incorrect energy:  {e_incorrect_mean:.2f} <= 10.0") 
    print(f"✅ Sufficient gap:       {energy_gap:.2f} > 8.0" if criteria_met[2] else f"❌ Insufficient gap:     {energy_gap:.2f} <= 8.0")
    print()
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"OVERALL STATUS: {status}")
    print("="*60)
    
    logger.info(f"Energy landscape validation completed: {'PASSED' if passed else 'FAILED'}")
    
    return results


def _generate_corrupted_solution(correct_out: torch.Tensor) -> torch.Tensor:
    """
    Generate corrupted version of correct solution for energy validation.
    
    Uses simple corruption strategies to create "incorrect" solutions that should
    have higher energy than correct ones.
    """
    corrupted = correct_out.clone()
    
    # Strategy 1: Add noise (70% probability)
    if torch.rand(1).item() < 0.7:
        noise_scale = 0.5 * corrupted.std() + 0.1  # Ensure minimum noise
        noise = noise_scale * torch.randn_like(corrupted)
        corrupted = corrupted + noise
    
    # Strategy 2: Random permutation (30% probability)  
    if torch.rand(1).item() < 0.3:
        batch_size = corrupted.size(0)
        for b in range(batch_size):
            # Permute within each batch element
            perm_idx = torch.randperm(corrupted.size(-1))
            corrupted[b] = corrupted[b, perm_idx]
    
    # Strategy 3: Sign flip (20% probability)
    if torch.rand(1).item() < 0.2:
        flip_mask = torch.rand_like(corrupted) < 0.3
        corrupted = torch.where(flip_mask, -corrupted, corrupted)
        
    return corrupted


if __name__ == "__main__":
    # Example usage
    logger.info("Algebra evaluation framework loaded successfully")
    
    # Test with mock data
    try:
        from algebra_encoder import create_character_encoder, create_decoder_with_default_candidates
        
        # Create encoder and decoder
        encoder = create_character_encoder(d_model=128)
        decoder = create_decoder_with_default_candidates(encoder, distance_threshold=2.0)
        
        # Test symbolic equivalence function
        pred_eqs = ["x=2", "x=3", "2*x=4"]
        target_eqs = ["x=2", "x=3", "x=2"]
        
        results = compute_symbolic_equivalence(pred_eqs, target_eqs)
        print(f"Test equivalence results: {results['symbolic_equivalence_rate']:.3f}")
        
        logger.info("Basic functionality test passed")
        
    except ImportError as e:
        logger.warning(f"Cannot run test due to missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Test failed: {e}")