import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import pandas as pd


class ModelEvaluator:
    """
    Handles model evaluation and visualization.
    """
    
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def evaluate_predictions(self, 
                           X_test: torch.Tensor, 
                           y_test: torch.Tensor, 
                           mean_y: torch.Tensor, 
                           std_y: torch.Tensor, 
                           show_all: bool = False) -> dict:
        """
        Evaluate model predictions on test set.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            mean_y: Mean value for denormalization
            std_y: Standard deviation for denormalization
            show_all: Whether to show all predictions or just first 20
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*70)
        print("TEST SET PREDICTIONS vs ACTUAL PRICES")
        print("="*70)
        
        # Move data to device
        X_test = X_test.to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            test_predictions_norm = self.model(X_test)
        
        # Denormalize to get actual dollar prices
        predictions_actual = test_predictions_norm * std_y + mean_y
        targets_actual = y_test * std_y + mean_y
        
        # Calculate metrics
        differences = predictions_actual - targets_actual
        mae = differences.abs().mean().item()
        rmse = (differences ** 2).mean().sqrt().item()
        mape = (differences.abs() / targets_actual * 100).mean().item()
        
        print(f"\nTest Set Performance:")
        print(f"  Total samples: {len(X_test)}")
        print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Determine how many to show
        num_to_show = len(predictions_actual) if show_all else min(20, len(predictions_actual))
        
        print(f"\n{'First ' + str(num_to_show) if not show_all else 'All ' + str(num_to_show)} Predictions:")
        print("-"*70)
        print(f"{'#':>4} | {'Predicted':>12} | {'Actual':>12} | {'Difference':>12} | {'Error %':>10}")
        print("-"*70)
        
        for i in range(num_to_show):
            pred = predictions_actual[i].item()
            actual = targets_actual[i].item()
            diff = pred - actual
            error_pct = abs(diff) / actual * 100
            
            print(f"{i+1:4d} | ${pred:11.2f} | ${actual:11.2f} | ${diff:+11.2f} | {error_pct:9.2f}%")
        
        if not show_all and len(predictions_actual) > num_to_show:
            print(f"  ... ({len(predictions_actual) - num_to_show} more samples)")
        
        print("-"*70)
        avg_diff = differences.mean().item()
        avg_error_pct = mape
        print(f"{'AVG':>4} | {'':>12} | {'':>12} | ${avg_diff:+11.2f} | {avg_error_pct:9.2f}%")
        print("="*70)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': predictions_actual.cpu().numpy(),
            'targets': targets_actual.cpu().numpy(),
            'differences': differences.cpu().numpy()
        }
    
    def plot_training_history(self, 
                             train_losses: List[float], 
                             test_losses: List[float],
                             save_path: Optional[str] = None) -> None:
        """
        Plot training and test loss over epochs.
        
        Args:
            train_losses: List of training losses
            test_losses: List of test losses
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Regular scale
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', alpha=0.7)
        plt.plot(test_losses, label='Test Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Log scale (better for seeing details)
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss', alpha=0.7)
        plt.plot(test_losses, label='Test Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to '{save_path}'")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, 
                                  predictions: np.ndarray, 
                                  targets: np.ndarray,
                                  title: str = "Predictions vs Actual Prices",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot predictions against actual values.
        
        Args:
            predictions: Predicted values
            targets: Actual values
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Time series plot
        plt.subplot(1, 2, 2)
        indices = range(min(100, len(predictions)))  # Show first 100 points
        plt.plot(indices, targets[indices], label='Actual', alpha=0.7)
        plt.plot(indices, predictions[indices], label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Price')
        plt.title('Time Series Comparison (First 100 samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to '{save_path}'")
        
        plt.show()
    
    def plot_residuals(self, 
                      predictions: np.ndarray, 
                      targets: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Plot residual analysis.
        
        Args:
            predictions: Predicted values
            targets: Actual values
            save_path: Path to save the plot (optional)
        """
        residuals = targets - predictions
        
        plt.figure(figsize=(15, 5))
        
        # Residuals vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Histogram of residuals
        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(1, 3, 3)
        from scipy import stats
        stats.probplot(residuals.flatten(), dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residuals plot saved to '{save_path}'")
        
        plt.show()
    
    def calculate_metrics(self, 
                         predictions: np.ndarray, 
                         targets: np.ndarray) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted values
            targets: Actual values
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Basic metrics
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        # Additional metrics
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        # Direction accuracy (for time series)
        if len(predictions) > 1:
            actual_direction = np.diff(targets) > 0
            pred_direction = np.diff(predictions) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Direction_Accuracy': direction_accuracy,
            'Mean_Error': np.mean(predictions - targets),
            'Std_Error': np.std(predictions - targets)
        }
    
    def generate_report(self, 
                       X_test: torch.Tensor, 
                       y_test: torch.Tensor, 
                       mean_y: torch.Tensor, 
                       std_y: torch.Tensor,
                       train_losses: List[float],
                       test_losses: List[float],
                       save_path: Optional[str] = None) -> dict:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            mean_y: Mean value for denormalization
            std_y: Standard deviation for denormalization
            train_losses: Training loss history
            test_losses: Test loss history
            save_path: Path to save the report (optional)
        
        Returns:
            Dictionary with all evaluation results
        """
        # Get predictions and metrics
        eval_results = self.evaluate_predictions(X_test, y_test, mean_y, std_y, show_all=False)
        
        # Calculate additional metrics
        metrics = self.calculate_metrics(eval_results['predictions'], eval_results['targets'])
        
        # Create report
        report = {
            'model_performance': metrics,
            'training_summary': {
                'final_train_loss': train_losses[-1],
                'final_test_loss': test_losses[-1],
                'total_epochs': len(train_losses),
                'best_test_loss': min(test_losses),
                'best_epoch': test_losses.index(min(test_losses)) + 1
            },
            'data_summary': {
                'test_samples': len(X_test),
                'prediction_range': {
                    'min': float(eval_results['predictions'].min()),
                    'max': float(eval_results['predictions'].max())
                },
                'actual_range': {
                    'min': float(eval_results['targets'].min()),
                    'max': float(eval_results['targets'].max())
                }
            }
        }
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Model Performance:")
        print(f"  MAE: ${metrics['MAE']:.2f}")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R² Score: {metrics['R2']:.4f}")
        print(f"  Direction Accuracy: {metrics['Direction_Accuracy']:.2f}%")
        print(f"\nTraining Summary:")
        print(f"  Final Train Loss: {report['training_summary']['final_train_loss']:.4f}")
        print(f"  Final Test Loss: {report['training_summary']['final_test_loss']:.4f}")
        print(f"  Best Test Loss: {report['training_summary']['best_test_loss']:.4f} (Epoch {report['training_summary']['best_epoch']})")
        print("="*70)
        
        # Save report if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Evaluation report saved to '{save_path}'")
        
        return report
