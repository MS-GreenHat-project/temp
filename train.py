import argparse
import yaml
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import mlflow
import os

# Azure ML Run context
try:
    from azureml.core.run import Run
    azure_run = Run.get_context()
    IS_AZURE_RUN = not isinstance(azure_run, str)
except:
    azure_run = None
    IS_AZURE_RUN = False

def log_param(key, value):
    mlflow.log_param(key, value)
    if IS_AZURE_RUN:
        azure_run.log(key, value)

def log_metric(key, value):
    mlflow.log_metric(key, value)
    if IS_AZURE_RUN:
        azure_run.log(key, value)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="yolov8n.pt")
    parser.add_argument("--data-folder", type=str, required=True, help="AzureML에서 마운트된 데이터셋 폴더")
    parser.add_argument("--output-dir", type=str, required=True, help="AzureML output 폴더")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.937)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[INFO] Using model: {args.model_path}")
    print(f"[INFO] Using dataset: {args.data_folder}")
    print(f"[INFO] Output dir: {args.output_dir}")

    # === [1] data.yaml 생성 ===
    data_yaml = {
        'path': args.data_folder,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {0: 'helmet', 1: 'no_helmet'}
    }
    data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    # === [2] MLflow 시작 ===
    mlflow.start_run()
    log_param("model_path", args.model_path)
    log_param("epochs", args.epochs)
    log_param("imgsz", args.imgsz)
    log_param("batch", args.batch)
    log_param("lr0", args.lr0)
    log_param("momentum", args.momentum)

    # === [3] YOLO 학습 ===
    from ultralytics import YOLO
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    project_dir = Path(args.output_dir)
    exp_dir = project_dir / f"exp_{timestamp}"

    model = YOLO(args.model_path)
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        momentum=args.momentum,
        project=str(project_dir),
        name=f"exp_{timestamp}",
        exist_ok=True
    )

    # === [4] Metric 로깅 ===
    results_csv = exp_dir / "results.csv"
    weights_dir = exp_dir / "weights"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        latest = df.iloc[-1]
        for k, v in {
            "precision": latest.get("metrics/precision(B)", 0.0),
            "recall": latest.get("metrics/recall(B)", 0.0),
            "mAP50": latest.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": latest.get("metrics/mAP50-95(B)", 0.0)
        }.items():
            log_metric(k, v)

    # === [5] 모델 및 결과 저장 ===
    final_dir = project_dir / "final"
    final_dir.mkdir(exist_ok=True)

    if (weights_dir / "best.pt").exists():
        shutil.copy(weights_dir / "best.pt", final_dir / "best.pt")
    if results_csv.exists():
        shutil.copy(results_csv, final_dir / "results.csv")

    mlflow.log_artifacts(str(final_dir), artifact_path="model")

    if IS_AZURE_RUN:
        if (final_dir / "best.pt").exists():
            azure_run.upload_file(name="best.pt", path_or_stream=str(final_dir / "best.pt"))
        if results_csv.exists():
            azure_run.upload_file(name="results.csv", path_or_stream=str(final_dir / "results.csv"))

    mlflow.end_run()

if __name__ == "__main__":
    main() 