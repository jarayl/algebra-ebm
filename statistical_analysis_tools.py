"""
Statistical Analysis Tools for T4: Extended Statistical Analysis Framework

This module provides additional statistical analysis capabilities that complement
the main statistical testing framework. Focuses on advanced statistical methods,
hypothesis testing, and specialized metrics for algebraic reasoning validation.

Classes:
- AdvancedStatisticalAnalyzer: Extended statistical analysis capabilities
- HypothesisTestSuite: Comprehensive hypothesis testing framework
- DistributionAnalyzer: Advanced distribution analysis tools
- PowerAnalysisCalculator: Statistical power and sample size calculations
"""

import numpy as np
import scipy.stats as stats
from scipy import special
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class DistributionProperties:
    """Comprehensive distribution properties for equation sets."""
    
    # Basic statistics
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float
    
    # Distribution shape
    normality_pvalue: float
    is_normal: bool
    distribution_type: str
    
    # Advanced properties
    entropy: float
    mutual_information: float
    effective_sample_size: int
    outlier_percentage: float


@dataclass
class HypothesisTestResults:
    """Results from comprehensive hypothesis testing."""
    
    normality_tests: Dict[str, Dict[str, float]]
    distribution_comparison_tests: Dict[str, Dict[str, float]]
    independence_tests: Dict[str, Dict[str, float]]
    goodness_of_fit_tests: Dict[str, Dict[str, float]]
    effect_size_measures: Dict[str, float]
    
    # Overall assessment
    statistical_assumptions_met: bool
    test_recommendations: List[str]


@dataclass
class PowerAnalysisResults:
    """Results from statistical power analysis."""
    
    achieved_power: float
    required_sample_size: int
    effect_size: float
    significance_level: float
    
    # Power curve data
    power_curve_sample_sizes: List[int]
    power_curve_powers: List[float]
    
    # Recommendations
    adequate_power: bool
    recommended_action: str


class AdvancedStatisticalAnalyzer:
    """
    Advanced statistical analysis capabilities for equation set evaluation.
    
    Provides sophisticated statistical methods for validating algebraic reasoning
    systems beyond basic descriptive statistics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_distribution_properties(self, data: np.ndarray) -> DistributionProperties:
        """
        Comprehensive analysis of data distribution properties.
        
        Args:
            data: 1D array of numerical data
            
        Returns:
            DistributionProperties with comprehensive analysis
        """
        if len(data) == 0:
            return self._empty_distribution_properties()
        
        # Basic statistics
        mean = float(np.mean(data))
        variance = float(np.var(data, ddof=1)) if len(data) > 1 else 0.0
        std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
        
        # Normality testing
        normality_pvalue = 0.0
        is_normal = False
        if len(data) >= 3:  # Minimum for Shapiro-Wilk
            try:
                _, normality_pvalue = stats.shapiro(data[:5000])  # Limit for computational efficiency
                is_normal = normality_pvalue > 0.05
            except Exception:
                normality_pvalue = 0.0
                is_normal = False
        
        # Distribution type identification
        distribution_type = self._identify_distribution_type(data)
        
        # Entropy calculation
        entropy = self._calculate_entropy(data)
        
        # Mutual information (with itself, as a baseline measure)
        mutual_information = self._calculate_mutual_information(data)
        
        # Effective sample size
        effective_sample_size = self._calculate_effective_sample_size(data)
        
        # Outlier detection
        outlier_percentage = self._calculate_outlier_percentage(data)
        
        return DistributionProperties(
            mean=mean,
            variance=variance,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            normality_pvalue=float(normality_pvalue),
            is_normal=is_normal,
            distribution_type=distribution_type,
            entropy=entropy,
            mutual_information=mutual_information,
            effective_sample_size=effective_sample_size,
            outlier_percentage=outlier_percentage
        )
    
    def compare_distributions(self, 
                            data1: np.ndarray, 
                            data2: np.ndarray,
                            alpha: float = 0.05) -> Dict[str, Any]:
        """
        Comprehensive comparison of two distributions.
        
        Args:
            data1: First dataset
            data2: Second dataset
            alpha: Significance level
            
        Returns:
            Dictionary with comparison results
        """
        if len(data1) == 0 or len(data2) == 0:
            return {'error': 'Empty dataset provided'}
        
        results = {}
        
        # Two-sample tests
        try:
            # Student's t-test
            t_stat, t_pval = stats.ttest_ind(data1, data2)
            results['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_pval),
                'significant': t_pval < alpha
            }
        except Exception as e:
            results['t_test'] = {'error': str(e)}
        
        try:
            # Mann-Whitney U test (non-parametric)
            mw_stat, mw_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            results['mann_whitney'] = {
                'statistic': float(mw_stat),
                'p_value': float(mw_pval),
                'significant': mw_pval < alpha
            }
        except Exception as e:
            results['mann_whitney'] = {'error': str(e)}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(data1, data2)
            results['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pval),
                'significant': ks_pval < alpha
            }
        except Exception as e:
            results['ks_test'] = {'error': str(e)}
        
        try:
            # Welch's t-test (unequal variances)
            welch_stat, welch_pval = stats.ttest_ind(data1, data2, equal_var=False)
            results['welch_t_test'] = {
                'statistic': float(welch_stat),
                'p_value': float(welch_pval),
                'significant': welch_pval < alpha
            }
        except Exception as e:
            results['welch_t_test'] = {'error': str(e)}
        
        # Effect size measures
        try:
            cohens_d = self._calculate_cohens_d(data1, data2)
            results['effect_size'] = {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        except Exception as e:
            results['effect_size'] = {'error': str(e)}
        
        return results
    
    def analyze_clustering_structure(self, 
                                   features: np.ndarray, 
                                   max_clusters: int = 10) -> Dict[str, Any]:
        """
        Analyze clustering structure in equation features.
        
        Args:
            features: 2D array of features (n_samples, n_features)
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with clustering analysis results
        """
        if features.shape[0] < 2:
            return {'error': 'Insufficient data for clustering analysis'}
        
        results = {}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Elbow method for optimal number of clusters
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_clusters + 1, features.shape[0]))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)
                inertias.append(kmeans.inertia_)
                
                # Silhouette score
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(features_scaled, cluster_labels)
                silhouette_scores.append(sil_score)
                
            except Exception:
                continue
        
        if inertias and silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            results['clustering_analysis'] = {
                'optimal_clusters': int(optimal_k),
                'silhouette_scores': dict(zip(k_range, silhouette_scores)),
                'inertias': dict(zip(k_range, inertias)),
                'max_silhouette_score': float(max(silhouette_scores))
            }
        
        return results
    
    def _empty_distribution_properties(self) -> DistributionProperties:
        """Return empty distribution properties for edge cases."""
        return DistributionProperties(
            mean=0.0, variance=0.0, std=0.0, skewness=0.0, kurtosis=0.0,
            normality_pvalue=0.0, is_normal=False, distribution_type='unknown',
            entropy=0.0, mutual_information=0.0, effective_sample_size=0,
            outlier_percentage=0.0
        )
    
    def _identify_distribution_type(self, data: np.ndarray) -> str:
        """Identify the likely distribution type of the data."""
        if len(data) < 10:
            return 'insufficient_data'
        
        # Test against common distributions
        distributions = {
            'normal': stats.norm,
            'uniform': stats.uniform,
            'exponential': stats.expon,
            'laplace': stats.laplace
        }
        
        best_fit = 'unknown'
        best_pval = 0.0
        
        for name, distribution in distributions.items():
            try:
                params = distribution.fit(data)
                ks_stat, ks_pval = stats.kstest(data, distribution.cdf, args=params)
                
                if ks_pval > best_pval:
                    best_pval = ks_pval
                    best_fit = name
            except Exception:
                continue
        
        return best_fit if best_pval > 0.05 else 'unknown'
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data distribution."""
        if len(data) == 0:
            return 0.0
        
        try:
            # Bin the data and calculate entropy
            hist, _ = np.histogram(data, bins=min(50, len(set(data))))
            probs = hist / len(data)
            probs = probs[probs > 0]
            return float(stats.entropy(probs))
        except Exception:
            return 0.0
    
    def _calculate_mutual_information(self, data: np.ndarray) -> float:
        """Calculate mutual information metric."""
        if len(data) < 2:
            return 0.0
        
        try:
            # Create lagged version for mutual information calculation
            data_lag = np.roll(data, 1)[1:]
            data_orig = data[:-1]
            
            # Discretize for mutual information
            bins = min(10, len(set(data)))
            data_orig_binned = np.digitize(data_orig, np.histogram(data_orig, bins=bins)[1])
            data_lag_binned = np.digitize(data_lag, np.histogram(data_lag, bins=bins)[1])
            
            return float(mutual_info_score(data_orig_binned, data_lag_binned))
        except Exception:
            return 0.0
    
    def _calculate_effective_sample_size(self, data: np.ndarray) -> int:
        """Calculate effective sample size accounting for autocorrelation."""
        n = len(data)
        if n < 2:
            return n
        
        try:
            # Simple autocorrelation-based effective sample size
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative autocorrelation
            neg_idx = np.where(autocorr < 0)[0]
            tau = neg_idx[0] if len(neg_idx) > 0 else 1
            
            eff_size = n / (1 + 2 * tau)
            return max(1, int(eff_size))
        except Exception:
            return n
    
    def _calculate_outlier_percentage(self, data: np.ndarray) -> float:
        """Calculate percentage of outliers using IQR method."""
        if len(data) < 4:
            return 0.0
        
        try:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((data < lower_bound) | (data > upper_bound))
            return float(outliers) / len(data) * 100
        except Exception:
            return 0.0
    
    def _calculate_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(data1), len(data2)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + 
                             (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return float((np.mean(data1) - np.mean(data2)) / pooled_std)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'


class HypothesisTestSuite:
    """
    Comprehensive hypothesis testing framework for statistical validation.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_comprehensive_tests(self, 
                               data: np.ndarray,
                               reference_data: Optional[np.ndarray] = None) -> HypothesisTestResults:
        """
        Run comprehensive hypothesis tests on data.
        
        Args:
            data: Primary dataset to test
            reference_data: Optional reference dataset for comparison
            
        Returns:
            HypothesisTestResults with comprehensive test results
        """
        # Normality tests
        normality_tests = self._run_normality_tests(data)
        
        # Distribution comparison tests (if reference data provided)
        distribution_tests = {}
        if reference_data is not None:
            distribution_tests = self._run_distribution_comparison_tests(data, reference_data)
        
        # Independence tests
        independence_tests = self._run_independence_tests(data)
        
        # Goodness of fit tests
        goodness_tests = self._run_goodness_of_fit_tests(data)
        
        # Effect size measures
        effect_sizes = {}
        if reference_data is not None:
            effect_sizes = self._calculate_effect_sizes(data, reference_data)
        
        # Overall assessment
        assumptions_met = self._assess_statistical_assumptions(
            normality_tests, distribution_tests, independence_tests
        )
        
        recommendations = self._generate_test_recommendations(
            normality_tests, distribution_tests, assumptions_met
        )
        
        return HypothesisTestResults(
            normality_tests=normality_tests,
            distribution_comparison_tests=distribution_tests,
            independence_tests=independence_tests,
            goodness_of_fit_tests=goodness_tests,
            effect_size_measures=effect_sizes,
            statistical_assumptions_met=assumptions_met,
            test_recommendations=recommendations
        )
    
    def _run_normality_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run multiple normality tests."""
        tests = {}
        
        if len(data) >= 3:
            # Shapiro-Wilk test
            try:
                stat, pval = stats.shapiro(data[:5000])  # Limit sample size
                tests['shapiro_wilk'] = {
                    'statistic': float(stat),
                    'p_value': float(pval),
                    'is_normal': pval > self.alpha
                }
            except Exception as e:
                tests['shapiro_wilk'] = {'error': str(e)}
        
        if len(data) >= 8:
            # D'Agostino's test
            try:
                stat, pval = stats.normaltest(data)
                tests['dagostino'] = {
                    'statistic': float(stat),
                    'p_value': float(pval),
                    'is_normal': pval > self.alpha
                }
            except Exception as e:
                tests['dagostino'] = {'error': str(e)}
        
        if len(data) >= 50:
            # Anderson-Darling test
            try:
                result = stats.anderson(data, dist='norm')
                critical_value = result.critical_values[2]  # 5% significance
                tests['anderson_darling'] = {
                    'statistic': float(result.statistic),
                    'critical_value': float(critical_value),
                    'is_normal': result.statistic < critical_value
                }
            except Exception as e:
                tests['anderson_darling'] = {'error': str(e)}
        
        return tests
    
    def _run_distribution_comparison_tests(self, 
                                         data1: np.ndarray, 
                                         data2: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run distribution comparison tests."""
        tests = {}
        
        # Two-sample Kolmogorov-Smirnov test
        try:
            stat, pval = stats.ks_2samp(data1, data2)
            tests['ks_2sample'] = {
                'statistic': float(stat),
                'p_value': float(pval),
                'distributions_equal': pval > self.alpha
            }
        except Exception as e:
            tests['ks_2sample'] = {'error': str(e)}
        
        # Mann-Whitney U test
        try:
            stat, pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            tests['mann_whitney'] = {
                'statistic': float(stat),
                'p_value': float(pval),
                'medians_equal': pval > self.alpha
            }
        except Exception as e:
            tests['mann_whitney'] = {'error': str(e)}
        
        # Mood's test for equal scale parameters
        try:
            stat, pval = stats.mood(data1, data2)
            tests['mood'] = {
                'statistic': float(stat),
                'p_value': float(pval),
                'scales_equal': pval > self.alpha
            }
        except Exception as e:
            tests['mood'] = {'error': str(e)}
        
        return tests
    
    def _run_independence_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run independence tests."""
        tests = {}
        
        if len(data) >= 10:
            # Runs test for randomness
            try:
                median = np.median(data)
                runs, n_runs = 0, 1
                
                for i in range(1, len(data)):
                    if (data[i] >= median) != (data[i-1] >= median):
                        n_runs += 1
                
                # Expected runs and variance
                n1 = np.sum(data >= median)
                n2 = len(data) - n1
                
                if n1 > 0 and n2 > 0:
                    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
                    
                    if var_runs > 0:
                        z_stat = (n_runs - expected_runs) / np.sqrt(var_runs)
                        pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        
                        tests['runs_test'] = {
                            'statistic': float(z_stat),
                            'p_value': float(pval),
                            'is_random': pval > self.alpha
                        }
            except Exception as e:
                tests['runs_test'] = {'error': str(e)}
        
        return tests
    
    def _run_goodness_of_fit_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run goodness of fit tests."""
        tests = {}
        
        if len(data) >= 10:
            # Test for uniform distribution
            try:
                stat, pval = stats.kstest(data, 'uniform', 
                                        args=(data.min(), data.max() - data.min()))
                tests['uniform_fit'] = {
                    'statistic': float(stat),
                    'p_value': float(pval),
                    'fits_uniform': pval > self.alpha
                }
            except Exception as e:
                tests['uniform_fit'] = {'error': str(e)}
            
            # Test for normal distribution
            try:
                stat, pval = stats.kstest(data, 'norm', 
                                        args=(np.mean(data), np.std(data)))
                tests['normal_fit'] = {
                    'statistic': float(stat),
                    'p_value': float(pval),
                    'fits_normal': pval > self.alpha
                }
            except Exception as e:
                tests['normal_fit'] = {'error': str(e)}
        
        return tests
    
    def _calculate_effect_sizes(self, 
                              data1: np.ndarray, 
                              data2: np.ndarray) -> Dict[str, float]:
        """Calculate effect size measures."""
        effect_sizes = {}
        
        try:
            # Cohen's d
            n1, n2 = len(data1), len(data2)
            if n1 > 1 and n2 > 1:
                pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + 
                                     (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
                if pooled_std > 0:
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    effect_sizes['cohens_d'] = float(cohens_d)
        except Exception:
            pass
        
        try:
            # Glass's delta
            if np.std(data2) > 0:
                glass_delta = (np.mean(data1) - np.mean(data2)) / np.std(data2)
                effect_sizes['glass_delta'] = float(glass_delta)
        except Exception:
            pass
        
        return effect_sizes
    
    def _assess_statistical_assumptions(self, 
                                      normality: Dict,
                                      distribution: Dict,
                                      independence: Dict) -> bool:
        """Assess if statistical assumptions are met."""
        # Check normality (at least one test should pass)
        normality_ok = False
        for test_name, test_result in normality.items():
            if 'is_normal' in test_result and test_result['is_normal']:
                normality_ok = True
                break
        
        # Check independence
        independence_ok = True
        for test_name, test_result in independence.items():
            if 'is_random' in test_result and not test_result['is_random']:
                independence_ok = False
                break
        
        return normality_ok and independence_ok
    
    def _generate_test_recommendations(self, 
                                     normality: Dict,
                                     distribution: Dict,
                                     assumptions_met: bool) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not assumptions_met:
            recommendations.append("Statistical assumptions not met - consider non-parametric tests")
        
        # Check normality results
        normal_tests_passed = sum(1 for test in normality.values() 
                                 if test.get('is_normal', False))
        
        if normal_tests_passed == 0:
            recommendations.append("Data not normally distributed - use non-parametric methods")
        elif normal_tests_passed < len(normality) / 2:
            recommendations.append("Questionable normality - verify with additional tests")
        
        # Distribution comparison recommendations
        if distribution:
            significant_diffs = sum(1 for test in distribution.values()
                                  if test.get('p_value', 1.0) < 0.05)
            if significant_diffs > 0:
                recommendations.append("Significant differences detected between distributions")
        
        if not recommendations:
            recommendations.append("Statistical assumptions satisfied - parametric tests appropriate")
        
        return recommendations


class PowerAnalysisCalculator:
    """
    Statistical power and sample size calculations for experimental design.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_power_analysis(self,
                               observed_data: np.ndarray,
                               expected_effect_size: float = 0.5,
                               alpha: float = 0.05,
                               desired_power: float = 0.8) -> PowerAnalysisResults:
        """
        Calculate comprehensive power analysis.
        
        Args:
            observed_data: Current data for analysis
            expected_effect_size: Expected Cohen's d effect size
            alpha: Significance level
            desired_power: Desired statistical power
            
        Returns:
            PowerAnalysisResults with power analysis
        """
        current_n = len(observed_data)
        
        # Calculate achieved power with current sample size
        achieved_power = self._calculate_power(current_n, expected_effect_size, alpha)
        
        # Calculate required sample size for desired power
        required_n = self._calculate_required_sample_size(expected_effect_size, alpha, desired_power)
        
        # Generate power curve
        sample_sizes = list(range(10, max(500, required_n + 100), 20))
        power_curve = [self._calculate_power(n, expected_effect_size, alpha) for n in sample_sizes]
        
        # Assessment and recommendations
        adequate_power = achieved_power >= desired_power
        
        if adequate_power:
            recommendation = f"Current sample size (n={current_n}) provides adequate power"
        else:
            recommendation = f"Increase sample size to n={required_n} for adequate power"
        
        return PowerAnalysisResults(
            achieved_power=achieved_power,
            required_sample_size=required_n,
            effect_size=expected_effect_size,
            significance_level=alpha,
            power_curve_sample_sizes=sample_sizes,
            power_curve_powers=power_curve,
            adequate_power=adequate_power,
            recommended_action=recommendation
        )
    
    def _calculate_power(self, n: int, effect_size: float, alpha: float) -> float:
        """
        Calculate statistical power for given parameters.
        
        Uses approximation for two-sample t-test power calculation.
        """
        if n < 2:
            return 0.0
        
        try:
            # Critical value for two-tailed test
            t_critical = stats.t.ppf(1 - alpha/2, df=2*n-2)
            
            # Non-centrality parameter
            ncp = effect_size * np.sqrt(n/2)
            
            # Power calculation using non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, df=2*n-2, nc=ncp)
            power += stats.nct.cdf(-t_critical, df=2*n-2, nc=ncp)
            
            return float(np.clip(power, 0, 1))
        except Exception:
            return 0.0
    
    def _calculate_required_sample_size(self, 
                                      effect_size: float, 
                                      alpha: float, 
                                      power: float) -> int:
        """
        Calculate required sample size for given power.
        
        Uses iterative approach to find required sample size.
        """
        if effect_size <= 0 or power >= 1:
            return 1000  # Return large number for invalid inputs
        
        # Binary search for required sample size
        low, high = 2, 10000
        
        while low < high:
            mid = (low + high) // 2
            calculated_power = self._calculate_power(mid, effect_size, alpha)
            
            if calculated_power >= power:
                high = mid
            else:
                low = mid + 1
        
        return low


# Factory function to create the complete statistical analysis suite
def create_extended_statistical_suite() -> Dict[str, Any]:
    """
    Create a complete suite of advanced statistical analysis tools.
    
    Returns:
        Dictionary with all statistical analysis components
    """
    return {
        'advanced_analyzer': AdvancedStatisticalAnalyzer(),
        'hypothesis_tester': HypothesisTestSuite(),
        'power_calculator': PowerAnalysisCalculator(),
        'suite_ready': True
    }


if __name__ == "__main__":
    # Demo usage
    suite = create_extended_statistical_suite()
    
    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(0.5, 1, 100)
    
    # Advanced analysis
    analyzer = suite['advanced_analyzer']
    props = analyzer.analyze_distribution_properties(data1)
    print(f"Distribution analysis: entropy={props.entropy:.3f}, outliers={props.outlier_percentage:.1f}%")
    
    # Hypothesis testing
    tester = suite['hypothesis_tester']
    test_results = tester.run_comprehensive_tests(data1, data2)
    print(f"Hypothesis tests: assumptions_met={test_results.statistical_assumptions_met}")
    
    # Power analysis
    power_calc = suite['power_calculator']
    power_results = power_calc.calculate_power_analysis(data1)
    print(f"Power analysis: achieved_power={power_results.achieved_power:.3f}")