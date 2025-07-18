import argparse
import json
import os
import shutil
from tqdm import tqdm
import glob

# argparse로 입력값 받기
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-json-folder', type=str, required=True, help='COCO annotation json 파일 경로')
    parser.add_argument('--image-folder', type=str, required=True, help='이미지 파일들이 들어있는 폴더')
    parser.add_argument('--output-folder', type=str, required=True, help='YOLO 포맷 데이터셋이 저장될 폴더')
    return parser.parse_args()

def find_latest_json(folder):
    json_files = glob.glob(os.path.join(folder, '**', '*.json'), recursive=True)
    if not json_files:
        raise FileNotFoundError("No json file found in coco_json_folder")
    latest_json = max(json_files, key=os.path.getmtime)
    return latest_json

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    images_dir = os.path.join(args.output_folder, 'images')
    labels_dir = os.path.join(args.output_folder, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    coco_json_path = find_latest_json(args.coco_json_folder)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    class_name_to_id = {cat['name']: i for i, cat in enumerate(coco['categories'])}

    # 이미지 복사
    for img in tqdm(coco['images'], desc='Copying images'):
        src = os.path.join(args.image_folder, os.path.basename(img['file_name']))
        dst = os.path.join(images_dir, os.path.basename(img['file_name']))
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"[경고] 이미지 파일 없음: {src}")

    # 라벨 생성
    for ann in tqdm(coco['annotations'], desc='Converting labels'):
        img = images[ann['image_id']]
        img_w, img_h = img['width'], img['height']
        cat_id = ann['category_id']
        class_id = class_name_to_id[categories[cat_id]]

        # COCO bbox: [x_min, y_min, width, height] -> YOLO: [x_center, y_center, width, height] (정규화)
        x, y, w, h = ann['bbox']
        # 절대좌표를 정규화 (0~1 범위로)
        # x /= img_w; w /= img_w; y /= img_h; h /= img_h
        x_center = x + w / 2
        y_center = y + h / 2

        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
        img_name = os.path.splitext(os.path.basename(img['file_name']))[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        with open(label_path, 'a') as f:
            f.write(label_line)

    print(f"[완료] YOLO 포맷 데이터셋이 {args.output_folder}에 생성되었습니다.")

if __name__ == "__main__":
    main() 