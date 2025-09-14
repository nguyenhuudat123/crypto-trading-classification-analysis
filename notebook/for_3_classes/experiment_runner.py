"""
Experiment Runner for Bitcoin Price Prediction
Main module for running comprehensive grid search experiments
"""

import pandas as pd
import torch
import warnings
from tqdm import tqdm

# Import custom modules
from data_preparation import (
    load_bitcoin_data, add_technical_indicators, normalize_features,
    create_target_labels, prepare_traditional_ml_data, prepare_pytorch_data
)
from training_utils import train_xgboost_model, train_pytorch_model_wrapper

warnings.filterwarnings('ignore')

class ExperimentRunner:
    """Class to handle all experiment operations"""
    
    def __init__(self, data_path=None, output_filename='bitcoin_prediction_results.csv', 
                 sequence_length=10, verbose=False):
        """
        Initialize experiment runner
        
        Args:
            data_path: Path to Bitcoin data CSV file
            output_filename: Name for output results file
            sequence_length: Sequence length for deep learning models
            verbose: Whether to print detailed progress
        """
        self.data_path = data_path
        self.output_filename = output_filename
        self.sequence_length = sequence_length
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_single_experiment(self, model_name, lookahead, threshold, df_base):
        """Run a single experiment for given parameters"""
        try:
            if self.verbose:
                print(f"Running {model_name} with lookahead={lookahead}, threshold={threshold}")
            
            # Create target labels
            df_processed = create_target_labels(df_base, lookahead, threshold)
            
            if model_name == 'XGBoost':
                # Traditional ML approach
                X_train, X_test, y_train, y_test, test_indices = prepare_traditional_ml_data(df_processed)
                accuracy, loss_count, loss_mean, transaction = train_xgboost_model(
                    X_train, X_test, y_train, y_test, df_processed, test_indices
                )
            else:
                # PyTorch deep learning approach
                data_result = prepare_pytorch_data(df_processed, self.sequence_length, self.device)
                
                if data_result[0] is None:  # Check if data preparation failed
                    if self.verbose:
                        print(f"Skipping {model_name} - insufficient data")
                    return None
                
                X_train, X_test, y_train, y_test = data_result
                
                if len(X_train) == 0 or len(X_test) == 0:
                    if self.verbose:
                        print(f"Skipping {model_name} - empty datasets")
                    return None
                
                accuracy, loss_count, loss_mean, transaction = train_pytorch_model_wrapper(
                    model_name, X_train, X_test, y_train, y_test, df_processed, 
                    self.device, self.sequence_length, self.verbose
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
        
        except Exception as e:
            print(f"Error in {model_name} with L={lookahead}, T={threshold}: {str(e)}")
            return None

    def run_grid_search(self, lookahead_values, threshold_values, model_types=None):
        """
        Run grid search across all models and parameters
        
        Args:
            lookahead_values: List of lookahead days to test
            threshold_values: List of threshold values to test
            model_types: List of model types to test (default: all models)
        """
        print(f"Using device: {self.device}")
        print("Loading and preparing Bitcoin data...")
        
        # Load and prepare base data
        df_raw = load_bitcoin_data(self.data_path)
        if df_raw is None:
            print("Failed to load data. Exiting.")
            return None
            
        df_with_indicators = add_technical_indicators(df_raw)
        df_normalized = normalize_features(df_with_indicators)
        df_base = df_normalized.dropna()
        
        print(f"Final dataset shape: {df_base.shape}")
        
        # Default model types
        if model_types is None:
            model_types = ['XGBoost', 'LSTM', 'GRU', 'CNN']
        
        # Initialize results storage
        all_results = []
        total_experiments = len(lookahead_values) * len(threshold_values) * len(model_types)
        
        print(f"Starting grid search with {total_experiments} experiments...")
        print(f"Lookahead values: {lookahead_values}")
        print(f"Threshold values: {threshold_values}")
        print(f"Model types: {model_types}")
        
        # Run experiments with progress bar
        with tqdm(total=total_experiments, desc="Grid Search Progress") as pbar:
            for model_name in model_types:
                for lookahead in lookahead_values:
                    for threshold in threshold_values:
                        result = self.run_single_experiment(model_name, lookahead, threshold, df_base)
                        
                        if result is not None:
                            all_results.append(result)
                        
                        # Update progress
                        pbar.set_postfix({
                            "Model": model_name,
                            "L": lookahead,
                            "T": f"{threshold:.2f}"
                        })
                        pbar.update(1)
        
        results_df = pd.DataFrame(all_results)
        
        # Save results
        if not results_df.empty:
            self.save_results(results_df)
        
        return results_df

    def display_results(self, results_df):
        """Display comprehensive results in organized format"""
        if results_df.empty:
            print("No results to display!")
            return
        
        print(f"\n{'='*70}")
        print("COMPLETE EXPERIMENTAL RESULTS")
        print(f"{'='*70}")
        print(results_df.round(4))
        
        print(f"\n{'='*70}")
        print("BEST PERFORMANCE BY MODEL TYPE")
        print(f"{'='*70}")
        if len(results_df) > 0:
            best_by_model = results_df.loc[results_df.groupby('model')['accuracy'].idxmax()]
            display_cols = ['model', 'lookahead', 'threshold', 'accuracy', 'loss_count', 'loss_mean']
            print(best_by_model[display_cols].round(4))
        
        print(f"\n{'='*70}")
        print("OVERALL CHAMPION CONFIGURATION")
        print(f"{'='*70}")
        if len(results_df) > 0:
            champion = results_df.loc[results_df['accuracy'].idxmax()]
            print(f"Best Model: {champion['model']}")
            print(f"Optimal Lookahead: {champion['lookahead']} days")
            print(f"Optimal Threshold: {champion['threshold']:.3f}")
            print(f"Peak Accuracy: {champion['accuracy']:.4f} ({champion['accuracy']*100:.2f}%)")
            print(f"Loss Events: {champion['loss_count']}")
            print(f"Average Loss: {champion['loss_mean']:.6f}")

    def save_results(self, results_df):
        """Save results to CSV file"""
        try:
            results_df.to_csv(self.output_filename, index=False)
            print(f"\nResults saved to: {self.output_filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

def quick_run(data_path, output_filename='results.csv', 
              lookahead_values=[3, 5, 8], threshold_values=[0.02, 0.03, 0.04],
              model_types=None, verbose=False):
    """
    Quick function to run experiments with custom parameters
    
    Args:
        data_path: Path to data file
        output_filename: Output file name
        lookahead_values: List of lookahead values
        threshold_values: List of threshold values  
        model_types: List of models to test
        verbose: Verbose output
    """
    runner = ExperimentRunner(data_path, output_filename, verbose=verbose)
    results = runner.run_grid_search(lookahead_values, threshold_values, model_types)
    
    if results is not None and not results.empty:
        runner.display_results(results)
        return results
    else:
        print("No results generated.")
        return None