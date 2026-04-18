import pickle
import torch
import matplotlib.pyplot as plt

def evaluate_predictions(model, X_test, y_test, mean_y, std_y, show_all=False):
    print("\n" + "="*70)
    print("TEST SET PREDICTIONS vs ACTUAL PRICES")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        test_predictions_norm = model(X_test)
    
    predictions_actual = test_predictions_norm * std_y + mean_y
    targets_actual = y_test * std_y + mean_y
    
    differences = predictions_actual - targets_actual
    mae = differences.abs().mean().item()
    rmse = (differences ** 2).mean().sqrt().item()
    mape = (differences.abs() / targets_actual * 100).mean().item()
    
    print(f"\nTest Set Performance:")
    print(f"  Total samples: {len(X_test)}")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
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

def plot_training_history(train_losses, test_losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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
    plt.show()

def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = LSTMModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint