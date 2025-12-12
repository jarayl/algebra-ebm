#!/usr/bin/env python3
"""
Enhanced calibration methods to add to AlgebraInference class.

This provides the calibration improvements requested:
1. Robust dataset interface validation
2. Stratified sampling across equation complexity
3. Enhanced numerical safeguards  
4. Production-ready error handling
"""

import torch
import logging
import math
import random
from typing import Dict, Any, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


def add_calibration_methods():
    """Add the enhanced calibration methods to AlgebraInference class."""
    
    def _validate_dataset_interface(self, dataset) -> Dict[str, Any]:
        """
        Validate dataset interface and return interface capabilities.
        
        Args:
            dataset: Dataset object to validate
            
        Returns:
            Dictionary with interface information and validation results
            
        Raises:
            ValueError: If dataset interface is invalid or unsupported
        """
        interface_info = {
            'has_get_equation_pair': False,
            'has_get_problem_info': False,
            'has_getitem': False,
            'has_len': False,
            'dataset_type': type(dataset).__name__,
            'recommended_interface': None
        }
        
        # Check basic requirements
        if not hasattr(dataset, '__len__'):
            raise ValueError(f"Dataset {type(dataset)} must implement __len__ method")
        
        interface_info['has_len'] = True
        dataset_length = len(dataset)
        
        if dataset_length == 0:
            raise ValueError("Calibration dataset is empty")
        
        # Check for equation pair retrieval interfaces
        if hasattr(dataset, 'get_equation_pair') and callable(getattr(dataset, 'get_equation_pair')):
            interface_info['has_get_equation_pair'] = True
            interface_info['recommended_interface'] = 'get_equation_pair'
            
            # Test the interface with first item
            try:
                input_eq, target_eq = dataset.get_equation_pair(0)
                if not isinstance(input_eq, str) or not isinstance(target_eq, str):
                    raise ValueError("get_equation_pair must return (str, str) tuple")
                interface_info['test_success'] = True
            except Exception as e:
                raise ValueError(f"get_equation_pair interface test failed: {e}")
        
        elif hasattr(dataset, 'get_problem_info') and callable(getattr(dataset, 'get_problem_info')):
            interface_info['has_get_problem_info'] = True
            interface_info['recommended_interface'] = 'get_problem_info'
            
            # Test the interface with first item
            try:
                problem_info = dataset.get_problem_info(0)
                if not isinstance(problem_info, dict):
                    raise ValueError("get_problem_info must return dictionary")
                
                required_keys = {'input_equation', 'target_equation'}
                if not required_keys.issubset(problem_info.keys()):
                    raise ValueError(f"get_problem_info must include keys: {required_keys}")
                
                input_eq = problem_info['input_equation']
                target_eq = problem_info['target_equation']
                if not isinstance(input_eq, str) or not isinstance(target_eq, str):
                    raise ValueError("Equation strings in get_problem_info must be strings")
                    
                interface_info['test_success'] = True
            except Exception as e:
                raise ValueError(f"get_problem_info interface test failed: {e}")
        
        elif hasattr(dataset, '__getitem__'):
            interface_info['has_getitem'] = True
            interface_info['recommended_interface'] = '__getitem__'
            
            # Test the interface with first item
            try:
                data_item = dataset[0]
                if isinstance(data_item, tuple) and len(data_item) >= 2:
                    # Check if it returns string pairs or tensor pairs
                    if isinstance(data_item[0], str) and isinstance(data_item[1], str):
                        interface_info['getitem_returns'] = 'string_pair'
                    else:
                        interface_info['getitem_returns'] = 'tensor_pair'
                        # For tensor pairs, we can't directly extract equations
                        logger.warning("Dataset __getitem__ returns tensors, not equation strings. "
                                     "This may limit calibration effectiveness.")
                else:
                    raise ValueError("__getitem__ must return tuple with at least 2 elements")
                    
                interface_info['test_success'] = True
            except Exception as e:
                raise ValueError(f"__getitem__ interface test failed: {e}")
        
        else:
            raise ValueError(
                f"Dataset {type(dataset)} must implement one of: "
                f"get_equation_pair(), get_problem_info(), or __getitem__"
            )
        
        logger.info(f"Dataset interface validation passed: {interface_info['recommended_interface']}")
        return interface_info
    
    def _extract_equation_pair(self, dataset, index: int, interface_info: Dict[str, Any]) -> Tuple[str, str]:
        """
        Extract equation pair using the best available interface.
        
        Args:
            dataset: Dataset object
            index: Index to retrieve
            interface_info: Interface information from validation
            
        Returns:
            Tuple of (input_equation, target_equation) strings
            
        Raises:
            ValueError: If equation extraction fails
        """
        try:
            if interface_info['has_get_equation_pair']:
                return dataset.get_equation_pair(index)
                
            elif interface_info['has_get_problem_info']:
                problem_info = dataset.get_problem_info(index)
                return problem_info['input_equation'], problem_info['target_equation']
                
            elif interface_info['has_getitem']:
                data_item = dataset[index]
                if interface_info.get('getitem_returns') == 'string_pair':
                    return data_item[0], data_item[1]
                else:
                    # For tensor pairs, we can't extract equation strings
                    raise ValueError("Cannot extract equation strings from tensor-based dataset")
            
            else:
                raise ValueError("No valid interface available for equation extraction")
                
        except IndexError:
            raise ValueError(f"Index {index} out of range for dataset")
        except Exception as e:
            raise ValueError(f"Failed to extract equation pair at index {index}: {e}")
    
    def _stratified_sample_indices(
        self,
        dataset,
        interface_info: Dict[str, Any],
        num_samples: int,
        num_complexity_bins: int = 3
    ) -> List[int]:
        """
        Generate stratified sample indices across equation types and complexity.
        
        Args:
            dataset: Dataset object
            interface_info: Interface information from validation
            num_samples: Total number of samples to generate
            num_complexity_bins: Number of complexity bins for stratification
            
        Returns:
            List of sample indices ensuring representative distribution
        """
        dataset_size = len(dataset)
        
        # If requesting more samples than dataset size, sample with replacement
        if num_samples >= dataset_size:
            logger.warning(f"Requested {num_samples} samples but dataset only has {dataset_size} items. "
                          f"Using all available data with replacement.")
            base_indices = list(range(dataset_size))
            additional_samples = num_samples - dataset_size
            random.shuffle(base_indices)
            return base_indices + random.choices(base_indices, k=additional_samples)
        
        # For efficient stratification, sample subset for complexity analysis
        analysis_sample_size = min(500, dataset_size // 4)
        analysis_indices = random.sample(range(dataset_size), analysis_sample_size)
        
        # Categorize equations by complexity
        complexity_categories = {
            'linear': [],
            'quadratic': [],
            'cubic': [],
            'unknown': []
        }
        
        for idx in analysis_indices:
            try:
                input_eq, target_eq = self._extract_equation_pair(dataset, idx, interface_info)
                
                # Estimate complexity using input equation
                complexity = self._estimate_equation_complexity(input_eq)
                complexity_categories[complexity].append(idx)
                
            except Exception as e:
                logger.debug(f"Error analyzing equation at index {idx}: {e}")
                complexity_categories['unknown'].append(idx)
        
        # Remove empty categories
        complexity_categories = {k: v for k, v in complexity_categories.items() if v}
        
        if not complexity_categories:
            logger.warning("No equations could be categorized, falling back to random sampling")
            return random.sample(range(dataset_size), num_samples)
        
        # Compute samples per category (stratified sampling)
        num_categories = len(complexity_categories)
        base_samples_per_category = num_samples // num_categories
        remainder_samples = num_samples % num_categories
        
        stratified_indices = []
        category_names = list(complexity_categories.keys())
        
        for i, (category, category_indices) in enumerate(complexity_categories.items()):
            # Calculate samples for this category
            samples_for_category = base_samples_per_category
            if i < remainder_samples:  # Distribute remainder samples
                samples_for_category += 1
            
            # Sample from this category (with replacement if needed)
            if samples_for_category <= len(category_indices):
                selected = random.sample(category_indices, samples_for_category)
            else:
                # Need replacement sampling
                selected = random.choices(category_indices, k=samples_for_category)
            
            stratified_indices.extend(selected)
            
            logger.debug(f"Sampled {len(selected)} indices from {category} complexity "
                        f"(available: {len(category_indices)})")
        
        # Expand stratified sample to full dataset using similarity-based sampling
        if len(stratified_indices) < num_samples:
            remaining_samples = num_samples - len(stratified_indices)
            
            # Sample remaining indices uniformly from dataset
            remaining_pool = [i for i in range(dataset_size) if i not in stratified_indices]
            if remaining_pool:
                additional_indices = random.sample(remaining_pool, 
                                                 min(remaining_samples, len(remaining_pool)))
                stratified_indices.extend(additional_indices)
        
        # Final shuffle to avoid any ordering bias
        random.shuffle(stratified_indices)
        
        # Log stratification results
        final_complexity_counts = Counter()
        for idx in stratified_indices[:num_samples]:  # Take exactly num_samples
            try:
                input_eq, _ = self._extract_equation_pair(dataset, idx, interface_info)
                complexity = self._estimate_equation_complexity(input_eq)
                final_complexity_counts[complexity] += 1
            except:
                final_complexity_counts['unknown'] += 1
        
        logger.info(f"Stratified sampling completed: {dict(final_complexity_counts)}")
        
        return stratified_indices[:num_samples]
    
    def calibrate_energy_scales(
        self,
        calibration_dataset,
        num_samples: int = 1000,
        reference_rule: str = 'distribute'
    ) -> Dict[str, float]:
        """
        Calibrate energy scales across different rules to empirically equalize energy ranges.
        
        This method provides post-training calibration to measure and correct energy scale 
        differences between rule-specific models. It complements the normalization approach 
        by providing empirical validation and correction based on actual data distributions.
        
        Enhanced with robust dataset interface validation and stratified sampling for
        representative calibration across equation types and complexity levels.
        
        Args:
            calibration_dataset: Dataset to sample from for calibration
                                Must support get_equation_pair(), get_problem_info(), or __getitem__
            num_samples: Number of samples to use per rule for statistics (default: 1000)
            reference_rule: Rule to use as baseline for scaling (default: 'distribute')
            
        Returns:
            Dict mapping rule names to calibration scaling factors
            
        Raises:
            ValueError: If reference rule not found, dataset is empty, or interface invalid
            RuntimeError: If calibration fails due to numerical issues
        """
        logger.info(f"Starting energy scale calibration with {num_samples} samples per rule")
        
        # Input validation
        if reference_rule not in self.rule_models:
            raise ValueError(f"Reference rule '{reference_rule}' not found in loaded models. "
                           f"Available rules: {list(self.rule_models.keys())}")
        
        # Validate and analyze dataset interface
        try:
            interface_info = self._validate_dataset_interface(calibration_dataset)
            logger.info(f"Using dataset interface: {interface_info['recommended_interface']}")
        except Exception as e:
            raise ValueError(f"Dataset interface validation failed: {e}")
        
        # Ensure models are in eval mode for consistent calibration
        original_modes = {}
        for rule_name, model in self.rule_models.items():
            original_modes[rule_name] = model.training
            model.eval()
        
        try:
            with torch.no_grad():  # No training during calibration
                # Generate stratified sample indices for representative calibration
                logger.info("Generating stratified sample indices...")
                sample_indices = self._stratified_sample_indices(
                    calibration_dataset, interface_info, num_samples
                )
                
                # Collect energy statistics for each rule
                rule_energy_stats = {}
                
                for rule_name in self.rule_models.keys():
                    logger.info(f"Collecting energy statistics for rule: {rule_name}")
                    
                    energies = []
                    samples_collected = 0
                    failed_samples = 0
                    
                    for idx in sample_indices:
                        try:
                            # Extract equation pair using validated interface
                            input_eq, target_eq = self._extract_equation_pair(
                                calibration_dataset, idx, interface_info
                            )
                            
                            # Encode equations
                            inp_embedding = self.encoder(input_eq).unsqueeze(0).to(self.device)
                            out_embedding = self.encoder(target_eq).unsqueeze(0).to(self.device)
                            
                            # Compute energy for this rule at middle timestep for stable calibration
                            k = 5  # Use middle timestep for calibration
                            t = torch.full((1,), k, dtype=torch.long, device=self.device)
                            
                            # Use compose_energies with single rule weight
                            rule_weights = {rule_name: 1.0}
                            for other_rule in self.rule_models:
                                if other_rule != rule_name:
                                    rule_weights[other_rule] = 0.0
                            
                            energy = self.compose_energies(inp_embedding, out_embedding, k, rule_weights, t)
                            
                            # Enhanced numerical validation with more robust checking
                            if torch.isfinite(energy).all() and not torch.isnan(energy).any():
                                energy_val = energy.item()
                                # Additional sanity check for extreme values
                                if -1000 < energy_val < 1000:  # Reasonable energy range
                                    energies.append(energy_val)
                                    samples_collected += 1
                                else:
                                    logger.debug(f"Energy value {energy_val} outside reasonable range for rule {rule_name}")
                                    failed_samples += 1
                            else:
                                logger.debug(f"Non-finite energy for rule {rule_name} at sample {idx}")
                                failed_samples += 1
                                
                        except Exception as e:
                            logger.debug(f"Error processing sample {idx} for rule {rule_name}: {e}")
                            failed_samples += 1
                            continue
                    
                    # Validate sufficient samples collected
                    if len(energies) == 0:
                        raise RuntimeError(f"Failed to collect any valid energy samples for rule {rule_name}. "
                                         f"Failed samples: {failed_samples}")
                    
                    if len(energies) < num_samples * 0.1:  # Less than 10% success rate
                        logger.warning(f"Low success rate for rule {rule_name}: {len(energies)}/{num_samples} "
                                     f"({100 * len(energies) / num_samples:.1f}%)")
                    
                    # Compute robust energy distribution statistics
                    energies_tensor = torch.tensor(energies)
                    
                    # Remove outliers using IQR method for robust statistics
                    q1, q3 = torch.quantile(energies_tensor, torch.tensor([0.25, 0.75]))
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filter outliers
                    filtered_energies = energies_tensor[(energies_tensor >= lower_bound) & 
                                                       (energies_tensor <= upper_bound)]
                    
                    if len(filtered_energies) == 0:
                        logger.warning(f"All energies filtered as outliers for rule {rule_name}, using raw data")
                        filtered_energies = energies_tensor
                    
                    outliers_removed = len(energies_tensor) - len(filtered_energies)
                    if outliers_removed > 0:
                        logger.debug(f"Removed {outliers_removed} outliers from {rule_name} energy statistics")
                    
                    rule_energy_stats[rule_name] = {
                        'mean': filtered_energies.mean().item(),
                        'std': filtered_energies.std().item(),
                        'min': filtered_energies.min().item(),
                        'max': filtered_energies.max().item(),
                        'median': filtered_energies.median().item(),
                        'q1': torch.quantile(filtered_energies, 0.25).item(),
                        'q3': torch.quantile(filtered_energies, 0.75).item(),
                        'samples': len(filtered_energies),
                        'outliers_removed': outliers_removed,
                        'failed_samples': failed_samples
                    }
                    
                    stats = rule_energy_stats[rule_name]
                    logger.info(f"Rule {rule_name}: mean={stats['mean']:.4f}, "
                               f"std={stats['std']:.4f}, median={stats['median']:.4f}, "
                               f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                               f"samples={stats['samples']}")
                
                # Compute scaling factors relative to reference rule with robust statistics
                reference_stats = rule_energy_stats[reference_rule]
                
                # Use median-based scaling for robustness against outliers
                reference_scale = reference_stats['std'] if reference_stats['std'] > 1e-6 else 1.0
                
                if reference_stats['std'] < 1e-6:
                    logger.warning(f"Reference rule '{reference_rule}' has very small energy std "
                                 f"({reference_stats['std']:.2e}), using median-based scaling")
                    reference_scale = max(abs(reference_stats['median']), 1.0)
                    use_std_scaling = False
                else:
                    reference_scale = reference_stats['std']
                    use_std_scaling = True
                
                calibration_scales = {}
                
                for rule_name, stats in rule_energy_stats.items():
                    if use_std_scaling:
                        rule_scale = stats['std']
                    else:
                        rule_scale = max(abs(stats['median']), 1.0)
                    
                    # Compute relative scaling factor with enhanced numerical stability
                    if rule_scale > 1e-6:
                        scale_factor = reference_scale / rule_scale
                    else:
                        logger.warning(f"Rule {rule_name} has very small energy scale ({rule_scale:.2e}), "
                                     f"setting calibration scale to 1.0")
                        scale_factor = 1.0
                    
                    # Enhanced validation of calibration scales with tighter bounds
                    if not (0.05 <= scale_factor <= 20.0):
                        logger.warning(f"Calibration scale for {rule_name} is outside reasonable range "
                                     f"({scale_factor:.4f}), clamping to [0.05, 20.0]")
                        scale_factor = max(0.05, min(scale_factor, 20.0))
                    
                    # Additional check for extreme scaling that might indicate model issues
                    if scale_factor > 10.0 or scale_factor < 0.1:
                        logger.warning(f"Large calibration adjustment needed for {rule_name}: {scale_factor:.4f}. "
                                     f"This may indicate training or data distribution issues.")
                    
                    calibration_scales[rule_name] = scale_factor
                    
                    logger.info(f"Calibration scale for {rule_name}: {scale_factor:.4f}")
                
                logger.info(f"Energy scale calibration completed successfully. "
                           f"Reference rule: {reference_rule}")
                
                # Log final scaling summary
                for rule_name, scale in calibration_scales.items():
                    original_stats = rule_energy_stats[rule_name]
                    logger.info(f"{rule_name}: scale={scale:.4f}, "
                               f"original_std={original_stats['std']:.4f}, "
                               f"samples={original_stats['samples']}")
                
                return calibration_scales
                
        except Exception as e:
            logger.error(f"Energy scale calibration failed: {e}")
            raise RuntimeError(f"Calibration failed: {e}")
            
        finally:
            # Restore original training modes
            for rule_name, model in self.rule_models.items():
                model.train(original_modes[rule_name])
    
    # Return the methods to be added to the class
    return {
        '_validate_dataset_interface': _validate_dataset_interface,
        '_extract_equation_pair': _extract_equation_pair,
        '_stratified_sample_indices': _stratified_sample_indices,
        'calibrate_energy_scales': calibrate_energy_scales
    }


if __name__ == "__main__":
    print("Enhanced calibration methods ready to be added to AlgebraInference class")