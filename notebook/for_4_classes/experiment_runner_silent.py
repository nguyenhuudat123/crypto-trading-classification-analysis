"""
Experiment Runner - Silent Version
Only shows progress bar and final results
"""

import pandas as pd
import torch
import warnings
from tqdm import tqdm
import sys

warnings.filterwarnings('ignore')

from data_preparation_silent import (
    load_bitcoin_data, add_technical_indicators, normalize_features,
    create_target_labels, prepare_traditional_ml_data, prepare_pytorch_data
)
from training_utils_silent import train_xgboost_model, train_pytorch_model_wrapper

class SilentExperimentRunner:
    """Silent experiment runner - minimal output"""
    
    def __init__(self, data_path, output_filename, sequence_length=10):
        self.data_path = data_path
        self.output_filename = output_filename
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_single_experiment(self, model_name, lookahead, threshold, df_base):
        """Run single experiment silently"""
        try:
            df_processed = create_target_labels(df_base, lookahead, threshold)
            
            if model_name == 'XGBoost':
                X_train, X_test, y_train, y_test, test_indices = prepare_traditional_ml_data(df_processed)
                accuracy, loss_count, loss_mean, transaction = train_xgboost_model(
                    X_train, X_test, y_train, y_test, df_processed, test_indices
                )
            else:
                data_result = prepare_pytorch_data(df_processed, self.sequence_length, self.device)
                if data_result[0] is None:
                    return None
                
                X_train, X_test, y_train, y_test = data_result
                if len(X_train) == 0 or len(X_test) == 0:
                    return None
                
                accuracy, loss_count, loss_mean, transaction = train_pytorch_model_wrapper(
                    model_name, X_train, X_test, y_train, y_test, df_processed, 
                    self.device, self.sequence_length
                )
            
            return {
                'model': model_name,
                'lookahead': lookahead,
                'threshold': threshold,
                'accuracy': accuracy,
                'loss_count': loss_count,
                'loss_mean': loss_mean,
                'transaction': transaction
            }
        
        except Exception:
            return None

    def run_grid_search(self, lookahead_values, threshold_values, model_types):
        """Run grid search with only progress bar"""
        
        # Suppress all output during data loading
        old_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w') if sys.platform != 'win32' else open('nul', 'w')
        
        try:
            df_raw = load_bitcoin_data(self.data_path)
            df_with_indicators = add_technical_indicators(df_raw)
            df_normalized = normalize_features(df_with_indicators)
            df_base = df_normalized.dropna()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        
        all_results = []
        total_experiments = len(lookahead_values) * len(threshold_values) * len(model_types)
        
        with tqdm(total=total_experiments, desc="Running experiments", ncols=100) as pbar:
            for model_name in model_types:
                for lookahead in lookahead_values:
                    for threshold in threshold_values:
                        result = self.run_single_experiment(model_name, lookahead, threshold, df_base)
                        
                        if result is not None:
                            all_results.append(result)
                        
                        pbar.set_postfix({
                            "Model": model_name,
                            "Status": "✓" if result else "✗"
                        })
                        pbar.update(1)
        
        results_df = pd.DataFrame(all_results)
        
        if not results_df.empty:
            results_df.to_csv(self.output_filename, index=False)
        
        return results_df

    def display_clean_results(self, results_df):
        """Display only the final results table"""
        if results_df.empty:
            print("No results generated.")
            return
        
        print(f"\n{'='*80}")
        print("BITCOIN PRICE PREDICTION RESULTS")
        print(f"{'='*80}")
        
        # Summary table
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<10} {'Lookahead':<10} {'Threshold':<12} {'Accuracy':<10} {'Loss Count':<12} {'Loss Mean':<10}")
        print("-" * 80)
        
        # Sort by accuracy
        sorted_results = results_df.sort_values('accuracy', ascending=False)
        
        for _, row in sorted_results.iterrows():
            print(f"{row['model']:<10} {row['lookahead']:<10} {row['threshold']:<12.3f} "
                  f"{row['accuracy']:<10.3f} {row['loss_count']:<12} {row['loss_mean']:<10.4f}")
        
        # Best result
        champion = results_df.loc[results_df['accuracy'].idxmax()]
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION:")
        print(f"Model: {champion['model']}")
        print(f"Accuracy: {champion['accuracy']:.3f} ({champion['accuracy']*100:.1f}%)")
        print(f"Lookahead: {champion['lookahead']} periods")
        print(f"Threshold: {champion['threshold']:.3f}")
        print(f"Loss Events: {champion['loss_count']}")
        print(f"Average Loss: {champion['loss_mean']:.4f}")
        print(f"{'='*80}")

def run_silent_experiment(data_path, output_filename, lookahead_values, threshold_values, model_types):
    """Run experiment with minimal output"""
    
    runner = SilentExperimentRunner(data_path, output_filename)
    results = runner.run_grid_search(lookahead_values, threshold_values, model_types)
    runner.display_clean_results(results)
    
    return results