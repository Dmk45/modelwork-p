import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_extractor import StockDataManager
from data_loader import DataLoader
from model_trainer import LSTMModel, ModelTrainer
from visualizer_evaluator import ModelEvaluator


def parse_symbols(symbols: str) -> List[str]:
    parsed = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    if not parsed:
        raise ValueError("At least one symbol is required.")
    return parsed


def build_default_architecture(
    num_features: int,
    hidden_size: int,
    num_layers: int,
    dense_size: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "type": "LSTM",
            "input_size": num_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": hidden_size,
        },
        {
            "type": "linear",
            "input_size": hidden_size,
            "hidden_size": dense_size,
            "output_size": dense_size,
        },
        {
            "type": "linear",
            "input_size": dense_size,
            "hidden_size": 1,
            "output_size": 1,
        },
    ]


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = parse_symbols(args.symbols)

    print("=" * 70)
    print("FULL STACK TRAINING PIPELINE")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Report path: {args.report_path}")
    print("=" * 70)

    # 1) Extract + preprocess data and save to disk
    extractor = StockDataManager(save_dir=args.data_dir)
    extractor.prepare_for_training(
        symbols=symbols,
        seq_length=args.seq_length,
        prediction_days=args.prediction_days,
        total_days=args.total_days,
        train_ratio=args.train_ratio,
        normalize=True,
        save=False,
    )
    extractor.save_data(filename=args.data_file)

    # 2) Load + validate prepared data
    loader = DataLoader(data_dir=args.data_dir)
    loaded_data = loader.load_data(filename=args.data_file)
    loader.validate_data(loaded_data)
    prepared_data = loader.prepare_training_data(
        loaded_data,
        train_ratio=args.train_ratio,
        reshape_y=True,
    )

    # 3) Build and train model
    model = LSTMModel()
    struct = build_default_architecture(
        num_features=prepared_data["X_train"].shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dense_size=args.dense_size,
    )
    model.create(prepared_data["X_train"], struct)

    trainer_device = None if args.device == "auto" else args.device
    trainer = ModelTrainer(model, device=trainer_device)
    train_losses, test_losses = trainer.train_model(
        prepared_data["X_train"],
        prepared_data["y_train"],
        prepared_data["X_test"],
        prepared_data["y_test"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 4) Save model checkpoint
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(
        filepath=str(model_path),
        struct=struct,
        train_losses=train_losses,
        test_losses=test_losses,
        final_train_loss=train_losses[-1],
        final_test_loss=test_losses[-1] if test_losses else float("nan"),
        normalization=prepared_data.get("normalization"),
    )

    # 5) Evaluate model and save report
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    normalization = prepared_data.get("normalization")
    if normalization is None:
        raise ValueError("Normalization parameters are required for evaluation.")

    evaluator = ModelEvaluator(model, device=trainer.device)
    report = evaluator.generate_report(
        prepared_data["X_test"],
        prepared_data["y_test"],
        normalization["mean_y"],
        normalization["std_y"],
        train_losses,
        test_losses,
        save_path=str(report_path),
    )

    print("\nPipeline complete.")
    print(f"Saved model checkpoint: {model_path}")
    print(f"Saved evaluation report: {report_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full stock-model pipeline from extraction through evaluation."
    )
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL")
    parser.add_argument("--seq-length", type=int, default=60)
    parser.add_argument("--prediction-days", type=int, default=1)
    parser.add_argument("--total-days", type=int, default=365)
    parser.add_argument("--train-ratio", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dense-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--data-dir", type=str, default="stock_data")
    parser.add_argument("--data-file", type=str, default="stock_data.pkl")
    parser.add_argument("--model-path", type=str, default="stock_predictor_model.pth")
    parser.add_argument("--report-path", type=str, default="evaluation_report.json")
    return parser


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    cli_args = arg_parser.parse_args()
    run_pipeline(cli_args)
