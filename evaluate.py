import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import mlflow
import os
import json
from ultralytics import YOLO
import shutil
import glob

# Azure ML Run context
try:
    from azureml.core.run import Run
    azure_run = Run.get_context()
    IS_AZURE_RUN = not isinstance(azure_run, str)
except:
    azure_run = None
    IS_AZURE_RUN = False

def log_metric(key, value):
    mlflow.log_metric(key, value)
    if IS_AZURE_RUN:
        azure_run.log(key, value)

def log_param(key, value):
    mlflow.log_param(key, value)
    if IS_AZURE_RUN:
        azure_run.log(key, value)

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 모델 단일 평가")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="평가할 모델 경로 (best.pt 파일)")
    parser.add_argument("--data-folder", type=str, required=True,
                       help="테스트 데이터 폴더 경로")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="결과 저장 디렉토리")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="신뢰도 임계값")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                       help="IoU 임계값")
    return parser.parse_args()

def load_model(model_path):
    """모델 로드 및 정보 반환"""
    try:
        model = YOLO(model_path)
        model_info = {
            'path': model_path,
            'name': Path(model_path).stem,
            'model': model
        }
        print(f"[INFO] 모델 로드 성공: {model_path}")
        return model_info
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {model_path}, 에러: {e}")
        return None

def evaluate_model(model_info, data_yaml_path, conf_threshold=0.25, iou_threshold=0.45):
    """모델 평가 수행"""
    if not model_info:
        return None
    
    print(f"[INFO] 모델 평가 중: {model_info['name']}")
    
    try:
        # 모델 평가
        results = model_info['model'].val(
            data=data_yaml_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True
        )
        
        # 결과 추출
        metrics = results.results_dict
        
        evaluation_result = {
            'model_name': model_info['name'],
            'precision': metrics.get('metrics/precision(B)', 0.0),
            'recall': metrics.get('metrics/recall(B)', 0.0),
            'mAP50': metrics.get('metrics/mAP50(B)', 0.0),
            'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0.0),
            'f1_score': metrics.get('metrics/f1(B)', 0.0),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        print(f"[INFO] 평가 완료: {model_info['name']}")
        print(f"  - Precision: {evaluation_result['precision']:.4f}")
        print(f"  - Recall: {evaluation_result['recall']:.4f}")
        print(f"  - mAP50: {evaluation_result['mAP50']:.4f}")
        print(f"  - mAP50-95: {evaluation_result['mAP50-95']:.4f}")
        print(f"  - F1-Score: {evaluation_result['f1_score']:.4f}")
        
        return evaluation_result
        
    except Exception as e:
        print(f"[ERROR] 모델 평가 실패: {model_info['name']}, 에러: {e}")
        return None

def generate_evaluation_report(result, output_dir):
    """평가 리포트 생성"""
    if not result:
        return None
    
    # 리포트 생성
    report = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': result['model_name'],
        'model_path': result.get('path', 'N/A'),
        'detailed_results': result,
        'summary': {
            'precision': result['precision'],
            'recall': result['recall'],
            'mAP50': result['mAP50'],
            'mAP50-95': result['mAP50-95'],
            'f1_score': result['f1_score']
        },
        'evaluation_settings': {
            'conf_threshold': result['conf_threshold'],
            'iou_threshold': result['iou_threshold']
        }
    }
    
    # JSON 파일로 저장
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 텍스트 리포트 생성
    txt_report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("YOLO 모델 평가 리포트\n")
        f.write("=" * 60 + "\n")
        f.write(f"평가 날짜: {report['evaluation_date']}\n")
        f.write(f"모델명: {report['model_name']}\n")
        f.write(f"모델 경로: {report['model_path']}\n\n")
        
        f.write("평가 설정:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Confidence Threshold: {report['evaluation_settings']['conf_threshold']}\n")
        f.write(f"  IoU Threshold: {report['evaluation_settings']['iou_threshold']}\n\n")
        
        f.write("성능 결과:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Precision: {result['precision']:.4f}\n")
        f.write(f"  Recall: {result['recall']:.4f}\n")
        f.write(f"  mAP50: {result['mAP50']:.4f}\n")
        f.write(f"  mAP50-95: {result['mAP50-95']:.4f}\n")
        f.write(f"  F1-Score: {result['f1_score']:.4f}\n\n")
        
        # 성능 해석
        f.write("성능 해석:\n")
        f.write("-" * 40 + "\n")
        if result['precision'] > 0.8:
            f.write("  Precision: 우수 (0.8 이상)\n")
        elif result['precision'] > 0.6:
            f.write("  Precision: 양호 (0.6-0.8)\n")
        elif result['precision'] > 0.4:
            f.write("  Precision: 보통 (0.4-0.6)\n")
        else:
            f.write("  Precision: 개선 필요 (0.4 미만)\n")
            
        if result['recall'] > 0.8:
            f.write("  Recall: 우수 (0.8 이상)\n")
        elif result['recall'] > 0.6:
            f.write("  Recall: 양호 (0.6-0.8)\n")
        elif result['recall'] > 0.4:
            f.write("  Recall: 보통 (0.4-0.6)\n")
        else:
            f.write("  Recall: 개선 필요 (0.4 미만)\n")
            
        if result['mAP50'] > 0.8:
            f.write("  mAP50: 우수 (0.8 이상)\n")
        elif result['mAP50'] > 0.6:
            f.write("  mAP50: 양호 (0.6-0.8)\n")
        elif result['mAP50'] > 0.4:
            f.write("  mAP50: 보통 (0.4-0.6)\n")
        else:
            f.write("  mAP50: 개선 필요 (0.4 미만)\n")
    
    print(f"[INFO] 평가 리포트 저장: {report_path}")
    print(f"[INFO] 텍스트 리포트 저장: {txt_report_path}")
    
    return report_path, txt_report_path

def main():
    args = parse_args()
    
    print("=" * 60)
    print("YOLO 모델 단일 평가 시작")
    print("=" * 60)
    print(f"[INFO] 모델 경로: {args.model_path}")
    print(f"[INFO] 데이터 폴더: {args.data_folder}")
    print(f"[INFO] 출력 디렉토리: {args.output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # data.yaml 파일 경로 (임시 디렉토리에 생성)
    images_folder = os.path.join(args.data_folder, 'images')
    labels_folder = os.path.join(args.data_folder, 'labels')
    
    if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
        print(f"[ERROR] images 또는 labels 폴더를 찾을 수 없습니다: {args.data_folder}")
        print(f"[INFO] images 폴더: {images_folder}")
        print(f"[INFO] labels 폴더: {labels_folder}")
        return
    
    # 임시 디렉토리에 data.yaml 생성
    import tempfile
    temp_dir = tempfile.mkdtemp()
    data_yaml_path = os.path.join(temp_dir, 'data.yaml')
    
    data_yaml_content = f"""path: {args.data_folder}
train: images
val: images
test: images

nc: 2
names: ['helmet', 'no-helmet']
"""
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    print(f"[INFO] data.yaml 파일 생성: {data_yaml_path}")
    
    # MLflow 시작
    mlflow.start_run()
    log_param("model_path", args.model_path)
    log_param("conf_threshold", args.conf_threshold)
    log_param("iou_threshold", args.iou_threshold)
    
    # 모델 로드
    model_path = args.model_path
    if os.path.isdir(model_path):
        # 폴더에서 best.pt 파일 찾기
        best_model_path = os.path.join(model_path, 'best.pt')
        if not os.path.exists(best_model_path):
            # final 폴더 안에서 찾기
            best_model_path = os.path.join(model_path, 'final', 'best.pt')
        if not os.path.exists(best_model_path):
            # 폴더 내의 모든 .pt 파일 중 가장 최근 것 찾기
            pt_files = glob.glob(os.path.join(model_path, '**/*.pt'), recursive=True)
            if pt_files:
                best_model_path = max(pt_files, key=os.path.getctime)
            else:
                print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")
                return
        model_path = best_model_path
    
    model_info = load_model(model_path)
    if not model_info:
        print("[ERROR] 모델을 로드할 수 없습니다.")
        return
    
    # 모델 평가
    result = evaluate_model(
        model_info, 
        data_yaml_path, 
        args.conf_threshold, 
        args.iou_threshold
    )
    
    if not result:
        print("[ERROR] 모델 평가에 실패했습니다.")
        return
    
    # 리포트 생성
    report_files = generate_evaluation_report(result, args.output_dir)
    
    # MLflow에 결과 로깅
    for metric, value in result.items():
        if metric not in ['model_name', 'timestamp', 'conf_threshold', 'iou_threshold']:
            log_metric(metric, value)
    
    # 아티팩트 업로드
    if IS_AZURE_RUN:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 리포트 파일들 업로드
        for file_path in [args.output_dir]:
            for file_name in os.listdir(file_path):
                full_path = os.path.join(file_path, file_name)
                if os.path.isfile(full_path):
                    azure_run.upload_file(
                        name=f"evaluation_{file_name}_{timestamp}", 
                        path_or_stream=full_path
                    )
    
    # MLflow 아티팩트 로깅
    mlflow.log_artifacts(args.output_dir, artifact_path="evaluation_results")
    
    print("\n" + "=" * 60)
    print("평가 완료!")
    print("=" * 60)
    
    # 최종 결과 요약
    print(f"\n모델: {result['model_name']}")
    print("-" * 40)
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  mAP50: {result['mAP50']:.4f}")
    print(f"  mAP50-95: {result['mAP50-95']:.4f}")
    print(f"  F1-Score: {result['f1_score']:.4f}")
    
    mlflow.end_run()

if __name__ == "__main__":
    main() 