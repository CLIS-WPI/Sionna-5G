import h5py
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from typing import Dict, Any

class MIMODatasetValidator:
    """
    Comprehensive MIMO dataset validation tool
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.expected_ranges = {
            "ber": {
                "range": (0, 1),
                "description": "Bit Error Rate"
            },
            "sinr": {
                "range": (-10, 40),
                "description": "Signal-to-Interference-plus-Noise Ratio"
            },
            "spectral_efficiency": {
                "range": (0, 12),
                "description": "Spectral Efficiency"
            },
            "effective_snr": {
                "range": (-10, 30),
                "description": "Effective SNR"
            },
            "eigenvalues": {
                "shape": (-1, 4),
                "description": "Channel Matrix Eigenvalues"
            },
            "channel_response": {
                "shape": (-1, 4, 4),
                "description": "Complex Channel Response"
            }
        }

    def check_statistics(self, data: np.ndarray, feature_name: str, 
                        expected_range=None, expected_shape=None) -> Dict[str, Any]:
        """Calculate comprehensive statistics for dataset features"""
        stats = {
            "shape": data.shape,
            "dtype": str(data.dtype)
        }

        if np.iscomplexobj(data):
            stats.update({
                "magnitude_stats": {
                    "min": float(np.min(np.abs(data))),
                    "max": float(np.max(np.abs(data))),
                    "mean": float(np.mean(np.abs(data))),
                    "std": float(np.std(np.abs(data)))
                },
                "phase_stats": {
                    "min": float(np.min(np.angle(data))),
                    "max": float(np.max(np.angle(data))),
                    "mean": float(np.mean(np.angle(data))),
                    "std": float(np.std(np.angle(data)))
                }
            })
        else:
            stats.update({
                "numeric_stats": {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data))
                }
            })

        if expected_range:
            min_val, max_val = expected_range
            invalid_count = np.sum((np.real(data) < min_val) | (np.real(data) > max_val))
            stats["range_check"] = {
                "expected_range": expected_range,
                "in_range": invalid_count == 0,
                "invalid_count": int(invalid_count)
            }

        if expected_shape:
            stats["shape_check"] = {
                "expected_shape": expected_shape,
                "shape_valid": data.shape[1:] == expected_shape[1:]
            }

        return stats

    def validate_path_loss(self, h5_file: h5py.File) -> Dict[str, Any]:
        """Validate path loss measurements at root level"""
        results = {}
        
        if 'path_loss_data' not in h5_file:
            print("Warning: No path loss data found at root level")
            return results
            
        path_loss_group = h5_file['path_loss_data']
        
        for path_feature in ['fspl', 'scenario_pathloss']:
            if path_feature not in path_loss_group:
                print(f"Warning: {path_feature} not found in root path loss data")
                continue
                
            data = np.array(path_loss_group[path_feature])
            
            # Add validation for NaN and infinite values
            nan_values = np.sum(np.isnan(data))
            inf_values = np.sum(np.isinf(data))
            
            total_values = data.size
            zero_values = np.sum(data == 0.0)
            low_values = np.sum((data > 0.0) & (data < 20.0))
            high_values = np.sum(data > 160.0)
            valid_values = np.sum((data >= 20.0) & (data <= 160.0))
            
            zero_percent = (zero_values/total_values*100)
            low_percent = (low_values/total_values*100)
            high_percent = (high_values/total_values*100)
            valid_percent = (valid_values/total_values*100)
            
            print(f"\nRoot Level - {path_feature} Path Loss Validation:")
            print(f"Total samples: {total_values}")
            print(f"Valid values (20-160 dB): {valid_values} ({valid_percent:.2f}%)")
            print(f"Invalid values breakdown:")
            print(f"  - Zero values: {zero_values} ({zero_percent:.2f}%)")
            print(f"  - Below 20dB: {low_values} ({low_percent:.2f}%)")
            print(f"  - Above 160dB: {high_values} ({high_percent:.2f}%)")
            print(f"  - NaN values: {nan_values}")
            print(f"  - Infinite values: {inf_values}")
            
            stats = self.check_statistics(
                data,
                f"root/{path_feature}",
                expected_range=(20, 160)
            )
            
            results[path_feature] = {
                **stats,
                'validation_stats': {
                    'total_samples': int(total_values),
                    'valid_samples': int(valid_values),
                    'valid_percentage': float(valid_percent),
                    'invalid_counts': {
                        'zero_values': int(zero_values),
                        'below_20db': int(low_values),
                        'above_160db': int(high_values),
                        'nan_values': int(nan_values),
                        'inf_values': int(inf_values)
                    },
                    'invalid_percentages': {
                        'zero_values': float(zero_percent),
                        'below_20db': float(low_percent),
                        'above_160db': float(high_percent)
                    }
                }
            }
            
            if zero_values > 0 or low_values > 0:
                print(f"Warning: {path_feature} contains {zero_values + low_values} values below minimum threshold (20 dB)")
            if high_values > 0:
                print(f"Warning: {path_feature} contains {high_values} values above maximum threshold (160 dB)")
            if nan_values > 0 or inf_values > 0:
                print(f"Warning: {path_feature} contains {nan_values} NaN values and {inf_values} infinite values")
                
        return results
            

    def validate_modulation_data(self, group: h5py.Group) -> Dict[str, Any]:
        """Validate data for each modulation scheme"""
        results = {}
        for mod_scheme in group:
            mod_results = {}
            print(f"\nValidating {mod_scheme} modulation scheme...")
            
            # Validate regular features
            for feature in group[mod_scheme]:
                if feature in self.expected_ranges:
                    data = np.array(group[mod_scheme][feature])
                    mod_results[feature] = self.check_statistics(
                        data,
                        f"{mod_scheme}/{feature}",
                        expected_range=self.expected_ranges[feature].get("range"),
                        expected_shape=self.expected_ranges[feature].get("shape")
                    )
            
            results[mod_scheme] = mod_results
        return results

    def validate_dataset(self) -> Dict[str, Any]:
        """Main validation function"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file '{self.file_path}' not found.")

        validation_results = {
            "file_path": self.file_path,
            "validation_time": datetime.now().isoformat(),
            "structure": {},
            "modulation_data": {},
            "configuration": {},
            "path_loss_data": {}  # Add this line
        }

        try:
            with h5py.File(self.file_path, 'r') as f:
                validation_results["structure"]["groups"] = list(f.keys())
                
                if "configuration" in f:
                    validation_results["configuration"] = {
                        key: value for key, value in f["configuration"].attrs.items()
                    }

                if "modulation_data" in f:
                    validation_results["modulation_data"] = self.validate_modulation_data(f["modulation_data"])
                else:
                    validation_results["errors"] = validation_results.get("errors", []) + ["Missing modulation_data group"]

                # Add path loss validation
                validation_results["path_loss_data"] = self.validate_path_loss(f)

                total_size = sum(
                    dataset.size * dataset.dtype.itemsize 
                    for dataset in f.values() 
                    if isinstance(dataset, h5py.Dataset)
                )
                validation_results["dataset_size_bytes"] = total_size

        except Exception as e:
            validation_results["status"] = "failed"
            validation_results["error"] = str(e)
        else:
            validation_results["status"] = "success"

        return validation_results

def print_validation_results(results: Dict[str, Any]):
    """Print formatted validation results"""
    print("\n=== MIMO Dataset Validation Results ===")
    print(f"File: {results['file_path']}")
    print(f"Status: {results['status']}")
    print(f"Dataset Size: {results['dataset_size_bytes'] / (1024*1024):.2f} MB")
    
    print("\nConfiguration:")
    for key, value in results['configuration'].items():
        print(f"  {key}: {value}")

    # Add path loss data section
    if 'path_loss_data' in results and results['path_loss_data']:
        print("\nPath Loss Data:")
        for feature, stats in results['path_loss_data'].items():
            print(f"  {feature}:")
            if 'validation_stats' in stats:
                vs = stats['validation_stats']
                print(f"    Valid samples: {vs['valid_samples']} ({vs['valid_percentage']:.2f}%)")
                print(f"    Invalid counts:")
                print(f"      Zero values: {vs['invalid_counts']['zero_values']}")
                print(f"      Below 20dB: {vs['invalid_counts']['below_20db']}")
                print(f"      Above 160dB: {vs['invalid_counts']['above_160db']}")

    print("\nModulation Schemes:")
    
    for mod_scheme, mod_data in results['modulation_data'].items():
        print(f"\n{mod_scheme}:")
        for feature, stats in mod_data.items():
            if feature == 'path_loss_data':
                print(f"  Path Loss Data:")
                for pl_feature, pl_stats in stats.items():
                    print(f"    {pl_feature}:")
                    if 'numeric_stats' in pl_stats:
                        print(f"      Range: [{pl_stats['numeric_stats']['min']:.4f}, {pl_stats['numeric_stats']['max']:.4f}]")
                        print(f"      Mean: {pl_stats['numeric_stats']['mean']:.4f}")
                        print(f"      Std: {pl_stats['numeric_stats']['std']:.4f}")
            else:
                print(f"  {feature}:")
                print(f"    Shape: {stats['shape']}")
                if 'numeric_stats' in stats:
                    print(f"    Range: [{stats['numeric_stats']['min']:.4f}, {stats['numeric_stats']['max']:.4f}]")
                    print(f"    Mean: {stats['numeric_stats']['mean']:.4f}")
                    print(f"    Std: {stats['numeric_stats']['std']:.4f}")

def main():
    file_path = "dataset/mimo_dataset_20241223_174336.h5"
    validator = MIMODatasetValidator(file_path)
    
    try:
        results = validator.validate_dataset()
        print_validation_results(results)
    except Exception as e:
        print(f"Error validating dataset: {e}")

if __name__ == "__main__":
    main()