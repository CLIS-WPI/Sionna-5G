# integrity/dataset_integrity_checker.py
# Advanced Dataset Integrity Verification System
# Performs comprehensive statistical and structural analysis of generated MIMO datasets
# Ensures dataset quality through multi-level validation and statistical testing

import h5py
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from logging import Logger
from core.metrics_calculator import MetricsCalculator
from config.system_parameters import SystemParameters
from utill.logging_config import LoggerManager

class MIMODatasetIntegrityChecker:
    def __init__(self, dataset_path: str, system_params: Optional[SystemParameters] = None):
        """
        Initialize the dataset integrity checker

        Args:
            dataset_path (str): Path to the HDF5 dataset file
            system_params (Optional[SystemParameters]): System parameters configuration
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.system_params = system_params or SystemParameters()
        self.metrics_calculator = MetricsCalculator(self.system_params)
        self.logger = LoggerManager.get_logger(__name__)
        
        # Use validation thresholds from MetricsCalculator
        self.validation_thresholds = self.metrics_calculator.validation_thresholds

    def verify_complex_data(self) -> Dict[str, Any]:
        """
        Verify complex data types and shapes in the dataset
        
        Returns:
            Dict[str, Any]: Verification results for complex data
        """
        results = {}
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                if 'channel_data' not in f:
                    return {
                        'status': False,
                        'error': 'Missing channel_data group'
                    }
                    
                channel_response = f['channel_data']['channel_response'][:]
                results = {
                    'status': True,
                    'dtype': str(channel_response.dtype),
                    'is_complex': np.iscomplexobj(channel_response),
                    'shape': channel_response.shape,
                    'validation': {
                        'correct_dtype': channel_response.dtype in [np.complex64, np.complex128],
                        'correct_shape': len(channel_response.shape) == 3
                    }
                }
                
                self.logger.info(f"Channel Response verification results:")
                self.logger.info(f"  dtype: {results['dtype']}")
                self.logger.info(f"  complex values: {results['is_complex']}")
                self.logger.info(f"  shape: {results['shape']}")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Complex data verification failed: {str(e)}")
            return {
                'status': False,
                'error': str(e)
            }
            
    def _validate_metrics(self, metrics_data: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
        """
        Validate metrics against thresholds
        
        Args:
            metrics_data: Dictionary of metrics arrays
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []
        
        for metric_name, data in metrics_data.items():
            if metric_name not in self.validation_thresholds:
                continue
                
            thresholds = self.validation_thresholds[metric_name]
            
            # Check for invalid values
            invalid_mask = (data < thresholds['min']) | (data > thresholds['max'])
            num_invalid = np.sum(invalid_mask)
            
            if num_invalid > 0:
                errors.append(
                    f"{metric_name}: Found {num_invalid} values outside valid range "
                    f"[{thresholds['min']}, {thresholds['max']}]"
                )
                
            # Check for non-finite values
            num_non_finite = np.sum(~np.isfinite(data))
            if num_non_finite > 0:
                errors.append(f"{metric_name}: Found {num_non_finite} non-finite values")
                
        return len(errors) == 0, errors
        
    def __enter__(self):
        """
        Context manager entry point
        Opens the HDF5 file for reading

        Returns:
            MIMODatasetIntegrityChecker: Instance of the checker
        """
        self.dataset = h5py.File(self.dataset_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point
        Closes the HDF5 file
        """
        if self.dataset:
            self.dataset.close()

    def check_dataset_integrity(self) -> Dict[str, Any]:
        """Perform comprehensive dataset integrity checks"""
        integrity_report = {
            'overall_status': True,
            'total_samples': 0,
            'metrics_validation': {},
            'configuration': {},
            'errors': [],
            'warnings': [],
            'validation_details': {},  # Added comma here
            'complex_data_verification': {}
        }
        
        try:
            # Check dataset structure
            if not self._check_dataset_structure():
                integrity_report['overall_status'] = False
                return integrity_report
                
            with h5py.File(self.dataset_path, 'r') as f:
                # Validate configuration
                config_result = self._validate_configuration(f)
                integrity_report['configuration'] = config_result
                
                # Validate metrics
                channel_data = f['channel_data']
                metrics_data = {
                    'spectral_efficiency': channel_data['spectral_efficiency'][:],
                    'effective_snr': channel_data['effective_snr'][:],
                    'eigenvalues': channel_data['eigenvalues'][:]
                }
                
                is_valid, errors = self._validate_metrics(metrics_data)
                integrity_report['metrics_validation'] = {
                    'valid': is_valid,
                    'errors': errors
                }
                
                if not is_valid:
                    integrity_report['overall_status'] = False
                    integrity_report['errors'].extend(errors)
                
                # Add statistics
                integrity_report['validation_details'] = self._calculate_statistics(metrics_data)
                
                complex_verification = self.verify_complex_data()
                integrity_report['complex_data_verification'] = complex_verification
                if not complex_verification.get('status', False):
                    integrity_report['overall_status'] = False
            return integrity_report
            
        except Exception as e:
            self.logger.error(f"Integrity check failed: {str(e)}")
            integrity_report['overall_status'] = False
            integrity_report['errors'].append(str(e))
            return integrity_report

    def _check_dataset_structure(self) -> bool:
        """Validate overall dataset structure"""
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                # Check main groups
                required_groups = ['channel_data', 'configuration']
                for group in required_groups:
                    if group not in f:
                        self.logger.error(f"Missing required group: {group}")
                        return False

                # Check channel data structure
                channel_group = f['channel_data']
                required_datasets = [
                    'channel_response',
                    'spectral_efficiency',
                    'effective_snr',
                    'condition_number',
                    'eigenvalues'
                ]
                
                for dataset in required_datasets:
                    if dataset not in channel_group:
                        self.logger.error(f"Missing required dataset: {dataset}")
                        return False

                # Verify dataset shapes
                base_shape = channel_group['channel_response'].shape
                expected_samples = self.system_params.total_samples
                
                if base_shape[0] != expected_samples:
                    self.logger.error(
                        f"Sample count mismatch: found {base_shape[0]}, "
                        f"expected {expected_samples}"
                    )
                    return False

                return True
                
        except Exception as e:
            self.logger.error(f"Structure check failed: {str(e)}")
            return False

    def _verify_path_loss_values(self, path_loss_data: tf.Tensor) -> Tuple[bool, List[str]]:
        """
        Verify path loss values are physically meaningful
        
        Args:
            path_loss_data (tf.Tensor): Path loss values to verify
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []
        min_valid_pl = 20.0
        max_valid_pl = 160.0
        
        try:
            # Convert to tensor if not already
            path_loss_data = tf.convert_to_tensor(path_loss_data)
            
            # Check for invalid values with detailed reporting
            too_low = tf.math.less(path_loss_data, min_valid_pl)
            too_high = tf.math.greater(path_loss_data, max_valid_pl)
            not_finite = tf.math.logical_not(tf.math.is_finite(path_loss_data))
            
            num_too_low = tf.reduce_sum(tf.cast(too_low, tf.int32))
            num_too_high = tf.reduce_sum(tf.cast(too_high, tf.int32))
            num_not_finite = tf.reduce_sum(tf.cast(not_finite, tf.int32))
            
            if num_too_low > 0:
                invalid_low = tf.boolean_mask(path_loss_data, too_low)
                errors.append(
                    f"Found {num_too_low} values below minimum ({min_valid_pl} dB). "
                    f"Minimum found: {tf.reduce_min(invalid_low):.2f} dB"
                )
                
            if num_too_high > 0:
                invalid_high = tf.boolean_mask(path_loss_data, too_high)
                errors.append(
                    f"Found {num_too_high} values above maximum ({max_valid_pl} dB). "
                    f"Maximum found: {tf.reduce_max(invalid_high):.2f} dB"
                )
                
            if num_not_finite > 0:
                errors.append(f"Found {num_not_finite} non-finite values")
                
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error verifying path loss values: {str(e)}")
            return False, errors
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """
        Validate dataset configuration parameters

        Returns:
            Dict[str, Any]: Configuration validation results
        """
        config = {}
        config_group = self.dataset['configuration']

        required_attrs = [
            'num_tx', 'num_rx', 'carrier_frequency',
            'num_subcarriers', 'subcarrier_spacing'
        ]

        for attr in required_attrs:
            if attr not in config_group.attrs:
                config[attr] = {'status': False, 'message': 'Missing attribute'}
            else:
                config[attr] = {'status': True, 'value': config_group.attrs[attr]}

        return config

    def _check_modulation_scheme_integrity(self, mod_scheme: str) -> Dict[str, Any]:
        """
        Check integrity of a specific modulation scheme

        Args:
            mod_scheme (str): Modulation scheme to check

        Returns:
            Dict[str, Any]: Modulation scheme integrity report
        """
        mod_group = self.dataset['modulation_data'][mod_scheme]

        report = {
            'integrity': True,
            'datasets': {},
            'samples': mod_group['channel_response'].shape[0]
        }

        datasets = [
            'channel_response', 'sinr', 'spectral_efficiency',
            'effective_snr', 'eigenvalues', 'ber', 'throughput'
        ]

        for dataset_name in datasets:
            dataset = mod_group[dataset_name]

            # Basic checks
            dataset_report = self._analyze_dataset(dataset, dataset_name)
            report['datasets'][dataset_name] = dataset_report
            report['integrity'] &= dataset_report['valid']

        return report

    def _analyze_dataset(self, dataset: h5py.Dataset, name: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of a dataset

        Args:
            dataset (h5py.Dataset): Dataset to analyze
            name (str): Dataset name

        Returns:
            Dict[str, Any]: Dataset analysis report
        """
        data = dataset[:]

        return {
            'valid': True,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'non_finite_ratio': np.mean(~np.isfinite(data)) if np.issubdtype(data.dtype, np.number) else 0,
            'statistics': {
                'mean': np.mean(data) if np.issubdtype(data.dtype, np.number) else None,
                'std': np.std(data) if np.issubdtype(data.dtype, np.number) else None,
                'min': np.min(data) if np.issubdtype(data.dtype, np.number) else None,
                'max': np.max(data) if np.issubdtype(data.dtype, np.number) else None
            }
        }

    def _perform_statistical_checks(self) -> Dict[str, Any]:
        """
        Perform advanced statistical checks across the dataset

        Returns:
            Dict[str, Any]: Statistical analysis results
        """
        statistical_checks = {}

        for mod_scheme in self.modulation_schemes:
            mod_group = self.dataset['modulation_data'][mod_scheme]

            # Correlation analysis
            metrics = ['sinr', 'spectral_efficiency', 'effective_snr']
            for metric in metrics:
                try:
                    data = mod_group[metric][:]
                    statistical_checks[f'{mod_scheme}_{metric}_ks_test'] = self._perform_distribution_test(data)
                except Exception as e:
                    statistical_checks[f'{mod_scheme}_{metric}_ks_test'] = {'error': str(e)}

        return statistical_checks

    def _perform_distribution_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform Kolmogorov-Smirnov distribution test

        Args:
            data (np.ndarray): Data to test

        Returns:
            Dict[str, Any]: Distribution test results
        """
        from scipy import stats

        # Normality test
        _, p_value = stats.normaltest(data)

        return {
            'normality_pvalue': p_value,
            'is_normal_dist': p_value > 0.05
        }


def verify_dataset_integrity(dataset_path: str) -> Dict[str, Any]:
    """
    Standalone function to verify dataset integrity

    Args:
        dataset_path (str): Path to the HDF5 dataset

    Returns:
        Dict[str, Any]: Comprehensive integrity check results
    """
    with MIMODatasetIntegrityChecker(dataset_path) as checker:
        return checker.check_dataset_integrity()

# In dataset_integrity_checker.py, add explicit size verification:
def validate_consistency(self, f):
    try:
        base_size = None
        sizes = {}
        
        # Check modulation data sizes
        for mod_scheme in f['modulation_data']:
            mod_size = f['modulation_data'][mod_scheme]['channel_response'].shape[0]
            sizes[f"modulation_{mod_scheme}"] = mod_size
            
            if base_size is None:
                base_size = mod_size
            elif mod_size != base_size:
                self.logger.error(f"Size mismatch in {mod_scheme}: {mod_size} vs {base_size}")
                return False
        
        # Check path loss data sizes
        pl_size = f['path_loss_data']['fspl'].shape[0]
        sizes["path_loss"] = pl_size
        
        if pl_size != base_size:
            self.logger.error(f"Path loss data size mismatch: {pl_size} vs {base_size}")
            return False
            
        self.logger.info(f"Dataset size consistency verified: {sizes}")
        return True
    except Exception as e:
        self.logger.error(f"Consistency validation failed: {str(e)}")
        return False
    
def verify_dataset(self) -> bool:
    """
    Verify dataset integrity
    """
    try:
        logger = logging.getLogger()
        
        # Check if dataset is opened
        if self.dataset is None:
            logger.error("Dataset file is not opened")
            return False
            
        # Log dataset structure
        logger.debug(f"Dataset groups: {list(self.dataset.keys())}")
        
        # Check modulation data
        if 'modulation_data' not in self.dataset:
            logger.error("Missing modulation_data group in dataset")
            return False
            
        # Check each modulation scheme
        for scheme in self.modulation_schemes:
            if scheme not in self.dataset['modulation_data']:
                logger.error(f"Missing {scheme} data in modulation_data group")
                return False
            
            # Log data shape and content summary
            data = self.dataset['modulation_data'][scheme]
            logger.debug(f"{scheme} data shape: {data.shape}")
            logger.debug(f"{scheme} data statistics: min={np.min(data)}, max={np.max(data)}")
            
        return True
        
    except Exception as e:
        logger.error(f"Dataset verification failed with error: {str(e)}", exc_info=True)
        return False
    
def main():
    """
    Example usage and demonstration of dataset integrity checking
    """
    dataset_path = 'dataset/mimo_dataset.h5'

    try:
        integrity_report = verify_dataset_integrity(dataset_path)

        print("\nDataset Integrity Report:")
        print(f"Overall Status: {'✅ PASSED' if integrity_report['overall_status'] else '❌ FAILED'}")

        # Print detailed results
        for mod_scheme, details in integrity_report['modulation_schemes'].items():
            print(f"\nModulation Scheme: {mod_scheme}")
            print(f"  Samples: {details['samples']}")
            print(f"  Integrity: {'✅ VALID' if details['integrity'] else '❌ INVALID'}")

    except Exception as e:
        print(f"Error during dataset integrity verification: {e}")


if __name__ == "__main__":
    main()
