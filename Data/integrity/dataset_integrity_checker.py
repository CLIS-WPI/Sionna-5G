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
class MIMODatasetIntegrityChecker:
    """
    Advanced dataset integrity verification for MIMO communication datasets
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset integrity checker

        Args:
            dataset_path (str): Path to the HDF5 dataset file
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.modulation_schemes = ['QPSK', '16QAM', '64QAM']

        # Update thresholds based on observed data ranges
        self.validation_thresholds = {
            'eigenvalues': {'min': 0.0, 'max': 1.0},
            'effective_snr': {'min': -25.1, 'max': 30.0},
            'spectral_efficiency': {'min': 0.0, 'max': 40.0},
            'ber': {'min': 0.0, 'max': 0.51},
            'sinr': {'min': -20.0, 'max': 30.0}
        }

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
        """
        Perform comprehensive dataset integrity checks with enhanced validation
        
        Returns:
            Dict[str, Any]: Detailed integrity check results including:
                - overall_status: bool indicating if all checks passed
                - total_samples: total number of samples in dataset
                - modulation_schemes: detailed results for each modulation scheme
                - configuration: configuration validation results
                - statistical_checks: results of statistical analysis
                - errors: list of any errors encountered
                - warnings: list of non-critical issues
                - validation_details: detailed validation metrics
        """
        integrity_report = {
            'overall_status': True,
            'total_samples': 0,
            'modulation_schemes': {},
            'configuration': {},
            'statistical_checks': {},
            'errors': [],
            'warnings': [],
            'validation_details': {}
        }

        try:
            # Check dataset structure with enhanced validation
            try:
                structure_valid = self._check_dataset_structure()
                integrity_report['overall_status'] &= structure_valid
                if not structure_valid:
                    integrity_report['errors'].append("Dataset structure validation failed")
                    
                # Additional structure validation
                with h5py.File(self.dataset_path, 'r') as f:
                    # Verify required groups exist
                    required_groups = ['modulation_data', 'path_loss_data', 'configuration']
                    for group in required_groups:
                        if group not in f:
                            integrity_report['errors'].append(f"Missing required group: {group}")
                            integrity_report['overall_status'] = False
                    
                    # Check group attributes
                    if 'configuration' in f:
                        required_attrs = ['creation_date', 'total_samples']
                        for attr in required_attrs:
                            if attr not in f['configuration'].attrs:
                                integrity_report['warnings'].append(
                                    f"Missing configuration attribute: {attr}"
                                )
                    
            except Exception as e:
                integrity_report['errors'].append(f"Structure check error: {str(e)}")
                integrity_report['overall_status'] = False

            # Enhanced configuration validation
            try:
                config_result = self._validate_configuration()
                integrity_report['configuration'] = config_result
                if not config_result.get('valid', False):
                    integrity_report['overall_status'] = False
                    integrity_report['errors'].append("Configuration validation failed")
                    
                # Additional configuration checks
                if 'num_tx' in config_result and 'num_rx' in config_result:
                    if config_result['num_tx']['value'] <= 0 or config_result['num_rx']['value'] <= 0:
                        integrity_report['errors'].append("Invalid antenna configuration")
                        integrity_report['overall_status'] = False
                        
            except Exception as e:
                integrity_report['errors'].append(f"Configuration validation error: {str(e)}")
                integrity_report['overall_status'] = False

            # Enhanced path loss validation
            try:
                with h5py.File(self.dataset_path, 'r') as f:
                    if 'path_loss_data' in f:
                        for dataset_name in ['fspl', 'scenario_pathloss']:
                            if dataset_name not in f['path_loss_data']:
                                integrity_report['errors'].append(
                                    f"Missing path loss dataset: {dataset_name}"
                                )
                                integrity_report['overall_status'] = False
                                continue
                                
                            path_loss_data = f['path_loss_data'][dataset_name][:]
                            
                            # Validate path loss values
                            is_valid, pl_errors = self._verify_path_loss_values(path_loss_data)
                            
                            if not is_valid:
                                integrity_report['overall_status'] = False
                                integrity_report['errors'].extend([
                                    f"Path loss validation failed for {dataset_name}:"
                                ] + pl_errors)
                                
                            # Additional path loss statistics
                            pl_stats = {
                                'mean': float(np.mean(path_loss_data)),
                                'std': float(np.std(path_loss_data)),
                                'min': float(np.min(path_loss_data)),
                                'max': float(np.max(path_loss_data)),
                                'zeros': int(np.sum(path_loss_data == 0)),
                                'non_finite': int(np.sum(~np.isfinite(path_loss_data)))
                            }
                            
                            integrity_report['validation_details'][f'{dataset_name}_stats'] = pl_stats
                            
                            # Check for suspicious values
                            if pl_stats['zeros'] > 0:
                                integrity_report['warnings'].append(
                                    f"Found {pl_stats['zeros']} zero values in {dataset_name}"
                                )
                            if pl_stats['non_finite'] > 0:
                                integrity_report['errors'].append(
                                    f"Found {pl_stats['non_finite']} non-finite values in {dataset_name}"
                                )
                                integrity_report['overall_status'] = False
                                
                    else:
                        integrity_report['errors'].append("Missing path_loss_data group")
                        integrity_report['overall_status'] = False
                        
            except Exception as e:
                integrity_report['errors'].append(f"Error checking path loss data: {str(e)}")
                integrity_report['overall_status'] = False

            # Enhanced modulation scheme validation
            total_samples = 0
            try:
                with h5py.File(self.dataset_path, 'r') as f:
                    if 'modulation_data' not in f:
                        integrity_report['errors'].append("Missing modulation_data group")
                        integrity_report['overall_status'] = False
                    else:
                        for mod_scheme in self.modulation_schemes:
                            try:
                                # Check modulation scheme existence
                                if mod_scheme not in f['modulation_data']:
                                    integrity_report['errors'].append(
                                        f"Missing data for modulation scheme: {mod_scheme}"
                                    )
                                    integrity_report['overall_status'] = False
                                    continue
                                    
                                # Validate modulation scheme data
                                mod_report = self._check_modulation_scheme_integrity(mod_scheme)
                                integrity_report['modulation_schemes'][mod_scheme] = mod_report
                                
                                if not mod_report.get('integrity', False):
                                    integrity_report['errors'].append(
                                        f"Integrity check failed for modulation scheme: {mod_scheme}"
                                    )
                                    integrity_report['overall_status'] = False
                                
                                # Count samples
                                samples = mod_report.get('samples', 0)
                                total_samples += samples
                                
                                # Additional modulation scheme validation
                                mod_group = f['modulation_data'][mod_scheme]
                                required_datasets = [
                                    'channel_response', 'sinr', 'spectral_efficiency',
                                    'effective_snr', 'eigenvalues', 'ber', 'throughput'
                                ]
                                
                                for dataset in required_datasets:
                                    if dataset not in mod_group:
                                        integrity_report['errors'].append(
                                            f"Missing {dataset} dataset for {mod_scheme}"
                                        )
                                        integrity_report['overall_status'] = False
                                        continue
                                        
                                    data = mod_group[dataset][:]
                                    
                                    # Dataset statistics
                                    stats = {
                                        'mean': float(np.mean(np.abs(data))),
                                        'std': float(np.std(np.abs(data))),
                                        'min': float(np.min(np.abs(data))),
                                        'max': float(np.max(np.abs(data))),
                                        'zeros': int(np.sum(data == 0)),
                                        'non_finite': int(np.sum(~np.isfinite(data)))
                                    }
                                    
                                    integrity_report['validation_details'][f'{mod_scheme}_{dataset}_stats'] = stats
                                    
                                    # Check for invalid values
                                    if stats['non_finite'] > 0:
                                        integrity_report['errors'].append(
                                            f"Found {stats['non_finite']} non-finite values in {mod_scheme}/{dataset}"
                                        )
                                        integrity_report['overall_status'] = False
                                        
                                    if stats['zeros'] > 0 and dataset not in ['ber']:  # BER can be zero
                                        integrity_report['warnings'].append(
                                            f"Found {stats['zeros']} zero values in {mod_scheme}/{dataset}"
                                        )
                                        
                            except Exception as e:
                                integrity_report['errors'].append(
                                    f"Error checking modulation scheme {mod_scheme}: {str(e)}"
                                )
                                integrity_report['overall_status'] = False
                                integrity_report['modulation_schemes'][mod_scheme] = {
                                    'integrity': False,
                                    'error': str(e)
                                }
                                
            except Exception as e:
                integrity_report['errors'].append(f"Error accessing dataset file: {str(e)}")
                integrity_report['overall_status'] = False

            integrity_report['total_samples'] = total_samples

            # Enhanced statistical checks
            try:
                stat_checks = self._perform_statistical_checks()
                integrity_report['statistical_checks'] = stat_checks
                if not stat_checks.get('valid', False):
                    integrity_report['overall_status'] = False
                    integrity_report['errors'].append("Statistical checks failed")
                    
            except Exception as e:
                integrity_report['errors'].append(f"Statistical check error: {str(e)}")
                integrity_report['overall_status'] = False

            # Final validation checks
            if total_samples == 0:
                integrity_report['overall_status'] = False
                integrity_report['errors'].append("No valid samples found in dataset")
                
            # Check sample count consistency
            if 'configuration' in integrity_report and 'total_samples' in integrity_report['configuration']:
                expected_samples = integrity_report['configuration']['total_samples'].get('value', 0)
                if total_samples != expected_samples:
                    integrity_report['errors'].append(
                        f"Sample count mismatch: found {total_samples}, expected {expected_samples}"
                    )
                    integrity_report['overall_status'] = False

            # Add timestamp and validation version
            integrity_report['timestamp'] = datetime.now().isoformat()
            integrity_report['validation_version'] = '2.0.0'

            return integrity_report

        except Exception as e:
            return {
                'overall_status': False,
                'error': str(e),
                'errors': [f"Critical error during integrity check: {str(e)}"],
                'timestamp': datetime.now().isoformat(),
                'validation_version': '2.0.0'
            }

    def _check_dataset_structure(self) -> bool:
        """
        Validate overall dataset structure

        Returns:
            bool: True if structure is valid
        """
        required_groups = ['modulation_data', 'path_loss_data', 'configuration']

        for group in required_groups:
            if group not in self.dataset:
                print(f"Missing required group: {group}")
                return False

        return True

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
