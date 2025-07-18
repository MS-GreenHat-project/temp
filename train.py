import argparse
import yaml
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import mlflow
import os
from sklearn.model_selection import train_test_split
import glob

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
    parser.add_argument("--data-folder", type=str, required=True, help="zureml:/subscriptions/ca3a7121-8c69-4769-a3c4-01dbafb3872d/resourceGroups/Green-Hat/providers/Microsoft.MachineLearningServices/workspaces/greenhat-ai/data/labeling-data/versions/5")
    parser.add_argument("--output-dir", type=str, required=True, help="/mnt/azureml/cr/j/2c9d42b6649043a881fd02c079f5d1ca/cap/data-capability/wd/model_output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.937)
    return parser.parse_args()

def split_and_prepare_yolo_dataset(data_folder, output_dir, val_ratio=0.1, test_ratio=0.1):
    images_dir = os.path.join(data_folder, 'images')
    labels_dir = os.path.join(data_folder, 'labels')

    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
    label_files = [os.path.join(labels_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in image_files]

    n = len(image_files)
    print(f"[INFO] Found {n} images in {images_dir}")
    
    # 라벨 파일 내용 확인
    print("[DEBUG] Checking data folder structure:")
    print(f"[DEBUG] Images dir: {images_dir} (exists: {os.path.exists(images_dir)})")
    print(f"[DEBUG] Labels dir: {labels_dir} (exists: {os.path.exists(labels_dir)})")
    print(f"[DEBUG] Images in directory: {len(image_files)}")
    print(f"[DEBUG] Labels in directory: {len(label_files)}")
    
    # 라벨 파일 샘플 확인
    valid_labels = 0
    class_counts = {}
    for i, label_file in enumerate(label_files[:5]):  # 처음 5개만 확인
        if os.path.exists(label_file):
            valid_labels += 1
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        lines = content.split('\n')
                        print(f"[DEBUG] Label file {i+1}: {os.path.basename(label_file)}")
                        print(f"[DEBUG] Content: {content}")
                        for line in lines:
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            except Exception as e:
                print(f"[ERROR] Error reading {label_file}: {e}")
    
    print(f"[DEBUG] Valid label files: {valid_labels}/{len(label_files[:5])}")
    print(f"[DEBUG] Class distribution in sample: {class_counts}")
    
    if n < 3:
        # 데이터가 3장 미만이면 모두 train에 할당
        print(f"[WARNING] Only {n} images found. All images will be used for training.")
        splits = {
            'train': (image_files, label_files),
            'valid': ([], []),
            'test': ([], [])
        }
    else:
        # split
        valtest_ratio = val_ratio + test_ratio
        train_imgs, valtest_imgs, train_lbls, valtest_lbls = train_test_split(
            image_files, label_files, test_size=valtest_ratio, random_state=42
        )
        if test_ratio > 0 and len(valtest_imgs) > 1:
            val_size = val_ratio / valtest_ratio
            val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
                valtest_imgs, valtest_lbls, test_size=(1 - val_size), random_state=42
            )
        else:
            val_imgs, test_imgs, val_lbls, test_lbls = valtest_imgs, [], valtest_lbls, []
        splits = {
            'train': (train_imgs, train_lbls),
            'valid': (val_imgs, val_lbls),
            'test': (test_imgs, test_lbls)
        }

    for split, (imgs, lbls) in splits.items():
        split_img_dir = os.path.join(output_dir, split, 'images')
        split_lbl_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        print(f"[INFO] Copying {len(imgs)} images to {split}/images")
        for img, lbl in zip(imgs, lbls):
            shutil.copy(img, os.path.join(split_img_dir, os.path.basename(img)))
            if os.path.exists(lbl):
                shutil.copy(lbl, os.path.join(split_lbl_dir, os.path.basename(lbl)))
            else:
                print(f"[경고] 라벨 파일 없음: {lbl}")

    return {
        'train': os.path.join(output_dir, 'train', 'images'),
        'valid': os.path.join(output_dir, 'valid', 'images'),
        'test': os.path.join(output_dir, 'test', 'images')
    }

def main():
    args = parse_args()
    print(f"[INFO] Using model: {args.model_path}")
    print(f"[INFO] Using dataset: {args.data_folder}")
    print(f"[INFO] Output dir: {args.output_dir}")

    # === [1] 데이터 분할 및 폴더 생성 ===
    split_dirs = split_and_prepare_yolo_dataset(args.data_folder, args.output_dir, val_ratio=0.1, test_ratio=0.1)

    # === [2] data.yaml 생성 ===
    data_yaml = {
        'path': args.output_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {0: 'helmet', 1: 'head'} 
    }
    data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"[DEBUG] Created data.yaml with classes: {data_yaml['names']}")
    
    # 실제 라벨 파일에서 클래스 분포 확인
    print("[DEBUG] Analyzing actual label files...")
    train_labels_dir = os.path.join(args.output_dir, 'train', 'labels')
    if os.path.exists(train_labels_dir):
        class_counts = {}
        total_labels = 0
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(train_labels_dir, label_file)
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                    total_labels += 1
                except Exception as e:
                    print(f"[ERROR] Error reading {label_path}: {e}")
        
        print(f"[DEBUG] Total labels found: {total_labels}")
        print(f"[DEBUG] Class distribution in train labels: {class_counts}")
        
        # 클래스명 매핑 확인
        for class_id, count in class_counts.items():
            class_name = data_yaml['names'].get(class_id, f"unknown_{class_id}")
            print(f"[DEBUG] Class {class_id} ({class_name}): {count} instances")

    # === [3] MLflow 시작 ===
    mlflow.start_run()
    log_param("model_path", args.model_path)
    log_param("epochs", args.epochs)
    log_param("imgsz", args.imgsz)
    log_param("batch", args.batch)
    log_param("lr0", args.lr0)
    log_param("momentum", args.momentum)

    # === [4] YOLO 학습 ===
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
        exist_ok=True,
        patience=50,  # Early stopping patience
        save_period=1,  # 매 에포크마다 저장
        verbose=True,  # 상세한 로그 출력
        plots=True,  # 학습 그래프 생성
        save=True,  # 모델 저장
        device=0  # GPU 사용
    )

    # === [5] Metric 로깅 ===
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

    # === [6] 모델 및 결과 저장 ===
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

    mlflow.end_run()

if __name__ == "__main__":
    main() 