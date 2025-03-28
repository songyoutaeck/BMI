from flask import Flask, render_template, request
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import os
import math
import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
import joblib
import json
import shutil
import os
import time

# Flask 애플리케이션 생성
app = Flask(__name__,template_folder='/home/yutaek/yutaek/templates')

# 경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static/outputs') 
DEPTH_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static/outputs_depth')
DEPTH_JSON_PATH = os.path.join(OUTPUT_FOLDER, "depth_paths.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEPTH_OUTPUT_FOLDER, exist_ok=True)

# 모델 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detection_model = YOLO('/home/yutaek/yutaek/yolov8x-pose-p6.pt')  # Detection 모델
pose_model = YOLO('/home/yutaek/yutaek/yolov8_pose_modify.pt')  # Pose 모델
depth_pipeline = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device=0)

# SAM 모델 로드
CHECKPOINT_PATH = "/home/yutaek/yutaek/segment-anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
hgb_model_path = "/home/yutaek/yutaek/hgb_model.pkl"
lgbm_model_path = "/home/yutaek/yutaek/lgbm_model.pkl"
catboost_model_path = "/home/yutaek/yutaek/catboost_model.pkl"



hgb_model = joblib.load(hgb_model_path)
lgbm_model = joblib.load(lgbm_model_path)
catboost_model = joblib.load(catboost_model_path)


class Identity(nn.Module):
    def forward(self, x):
        return x

class CustomEfficientNetB7(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomEfficientNetB7, self).__init__()
        self.base_model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        self.base_model.classifier = Identity()  # Replace the classifier with an identity layer
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        output = self.fc(features)
        return features, output  # Return both features and output



@app.route("/")
def index():
    """파일 업로드 폼"""
    return render_template('index.html')

@app.route("/process", methods=["POST"])
def process_image():
    """YOLO, Depth, SAM, and Keypoint 처리"""
    file = request.files.get("file")
    if not file:
        return "No file uploaded.", 400

    # 이미지 저장
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # 사람 영역 추출
    cropped_path = crop_person(input_path)

    # CSV 저장 경로 설정
    output_csv = os.path.join(OUTPUT_FOLDER, "measurements.csv")
    measurements, keypoint_image_path  = extract_keypoints_and_measurements(cropped_path, output_csv)
 
 

    # Depth 예측
    depth_path = process_depth(cropped_path)
    depth_data = {}
    if os.path.exists(DEPTH_JSON_PATH):
        with open(DEPTH_JSON_PATH, "r") as json_file:
            depth_data = json.load(json_file)
    depth_data[os.path.basename(cropped_path)] = depth_path
    with open(DEPTH_JSON_PATH, "w") as json_file:
        json.dump(depth_data, json_file)
    latest_depth_path = depth_data.get(os.path.basename(cropped_path))  

    # SAM으로 마스크 생성 및 저장
    mask_paths = process_sam(latest_depth_path)  # 수정: Depth로 생성된 이미지 경로를 SAM에 입력

    start_time = time.time()
    
    # Cropped 처리 시간
    cropped_start_time = time.time()
    cropped_path = crop_person(input_path)
    cropped_duration = time.time() - cropped_start_time
    print(f"Cropped Duration: {cropped_duration:.4f} seconds")

    # Depth 처리 시간
    depth_start_time = time.time()
    depth_path = process_depth(cropped_path)
    depth_duration = time.time() - depth_start_time
    print(f"Depth Duration: {depth_duration:.4f} seconds")

    # SAM 처리 시간
    sam_start_time = time.time()
    mask_paths = process_sam(depth_path)
    sam_duration = time.time() - sam_start_time
    print(f"SAM Duration: {sam_duration:.4f} seconds")

    total_duration = time.time() - start_time
    print(f"Total Duration: {total_duration:.4f} seconds")

    total_cropped_time = 0
    total_depth_time = 0
    total_sam_time = 0
    total_processes = 0

    # 여러번 반복하여 평균을 구할 수 있는 예시
    num_iterations = 5  # 예를 들어 5번의 반복을 통해 평균을 구한다고 가정

    for _ in range(num_iterations):
        # 크롭 처리 시간
        cropped_start_time = time.time()
        cropped_path = crop_person(input_path)
        cropped_duration = time.time() - cropped_start_time
        total_cropped_time += cropped_duration

        # Depth 처리 시간
        depth_start_time = time.time()
        depth_path = process_depth(cropped_path)
        depth_duration = time.time() - depth_start_time
        total_depth_time += depth_duration

        # SAM 처리 시간
        sam_start_time = time.time()
        mask_paths = process_sam(depth_path)
        sam_duration = time.time() - sam_start_time
        total_sam_time += sam_duration

        total_processes += 1

    # 평균 응답 시간 계산
    average_cropped_time = total_cropped_time / total_processes
    average_depth_time = total_depth_time / total_processes
    average_sam_time = total_sam_time / total_processes

    # 전체 평균 처리 시간 계산
    total_time = total_cropped_time + total_depth_time + total_sam_time
    average_total_time = total_time / total_processes

    print(f"Average Cropped Duration: {average_cropped_time:.4f} seconds")
    print(f"Average Depth Duration: {average_depth_time:.4f} seconds")
    print(f"Average SAM Duration: {average_sam_time:.4f} seconds")
    print(f"Average Total Duration: {average_total_time:.4f} seconds")

    # HTML 결과 렌더링
    return render_template(
        'result.html', 
        cropped_image=f"outputs/{os.path.basename(cropped_path)}",
        depth_image=f"outputs_depth/{os.path.basename(latest_depth_path)}",
        keypoint_image=f"outputs/{os.path.basename(keypoint_image_path)}",
        masks=mask_paths,
        measurements=measurements
        
    )



@app.route("/select_masks", methods=["POST"])
def select_masks():
    """사용자가 선택한 마스크를 처리하고 저장"""
    start_time = time.time()

    selected_masks = request.form.getlist("selected_masks")
    if not selected_masks:
        return "No masks selected.", 400

    selected_images_folder = os.path.join(OUTPUT_FOLDER, "selected_images")
    os.makedirs(selected_images_folder, exist_ok=True)

    # JSON 파일에서 Depth 경로 로드
    if not os.path.exists(DEPTH_JSON_PATH):
        return "Depth information not found.", 400

    with open(DEPTH_JSON_PATH, "r") as json_file:
        depth_data = json.load(json_file)

    # Get the last saved depth path
    last_depth_key = list(depth_data.keys())[-1] if depth_data else None
    last_depth_path = depth_data.get(last_depth_key) if last_depth_key else None

    results = []
    csv_data = []
    for mask_path in selected_masks:
        mask_full_path = os.path.join(OUTPUT_FOLDER, os.path.basename(mask_path))
        if os.path.exists(mask_full_path):
            # 선택된 마스크 저장
            selected_image_path = os.path.join(selected_images_folder, f"selected_{os.path.basename(mask_path)}")
            Image.open(mask_full_path).save(selected_image_path)

            # Depth Image Path 설정
            depth_img_path = last_depth_path  # Always use the last saved depth path

            if not depth_img_path or not os.path.exists(depth_img_path):
                print(f"[ERROR] Depth image not found for {mask_full_path}. Skipping this mask.")
                depth_img_path = None  # 없는 경우 None으로 설정


            img_mask = np.array(Image.open(mask_full_path).convert('L'))  # Load mask as grayscale
            y_neck, y_hip, y_knee, y_waist = 50, 150, 250, 100  # Example y-coordinates

            white_pixel_area_below_neck = compute_white_pixel_area(img_mask, y_neck, img_mask.shape[0]) if y_neck is not None else 0
            white_pixel_area_below_hip = compute_white_pixel_area(img_mask, y_hip, img_mask.shape[0]) if y_hip is not None else 0
            white_pixel_area_below_knee = compute_white_pixel_area(img_mask, y_knee, img_mask.shape[0]) if y_knee is not None else 0
            white_pixel_area_between_wt = compute_white_pixel_area(img_mask, y_waist, y_hip) if y_waist is not None and y_hip is not None else 0
            total_white_pixel_area = compute_white_pixel_area(img_mask, 0, img_mask.shape[0])

            white_pixel_ratio = white_pixel_area_between_wt / total_white_pixel_area if total_white_pixel_area > 0 else 0
            white_neck_ratio = white_pixel_area_below_neck / total_white_pixel_area if total_white_pixel_area > 0 else 0
            white_hip_ratio = white_pixel_area_below_hip / total_white_pixel_area if total_white_pixel_area > 0 else 0
            white_knee_ratio = white_pixel_area_below_knee / total_white_pixel_area if total_white_pixel_area > 0 else 0

            results.append({
                'mask_path': mask_full_path,
                'selected_image_path': selected_image_path,
                'depth_img_path': depth_img_path,  # 수정: 항상 포함되도록 함
                'white_pixel_area_below_neck': white_pixel_area_below_neck,
                'white_pixel_area_below_hip': white_pixel_area_below_hip,
                'white_pixel_area_below_knee': white_pixel_area_below_knee,
                'white_pixel_area_between_wt': white_pixel_area_between_wt,
                'total_white_pixel_area': total_white_pixel_area,
                'white_pixel_ratio': white_pixel_ratio,
                'white_neck_ratio': white_neck_ratio,
                'white_hip_ratio': white_hip_ratio,
                'white_knee_ratio': white_knee_ratio
            })

            # CSV 데이터 추가
            csv_data.append({
                'Mask Path': mask_full_path,
                'Selected Image Path': selected_image_path,
                'Depth Image Path': depth_img_path if depth_img_path else "Not Found",  # None일 경우 처리
                'White Pixel Area Below Neck': white_pixel_area_below_neck,
                'White Pixel Area Below Hip': white_pixel_area_below_hip,
                'White Pixel Area Below Knee': white_pixel_area_below_knee,
                'White Pixel Area Between Waist And Hip': white_pixel_area_between_wt,
                'Total White Pixel Area': total_white_pixel_area,
                'White Pixel Ratio': white_pixel_ratio,
                'White Neck Ratio': white_neck_ratio,
                'White Hip Ratio': white_hip_ratio,
                'White Knee Ratio': white_knee_ratio
            })

    # Save to CSV
    csv_path = os.path.join(OUTPUT_FOLDER, "selected_masks_analysis.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)

    # 모델 로드
    model_path = "/home/yutaek/yutaek/best_model_epoch_36_val_loss_1.4145.pth"
    model = CustomEfficientNetB7(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Extract features for depth and mask images
    for row in results:
        depth_img_path = row['depth_img_path']
        mask_img_path = row['selected_image_path']

        if depth_img_path is None:
            continue  # Depth 이미지가 없으면 건너뜀

        # Load images
        depth_img = Image.open(depth_img_path).convert('RGB')
        mask_img = Image.open(mask_img_path).convert('L')

        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        depth_img = transform(depth_img)
        mask_img = transform(mask_img)

        # Apply mask
        mask_img = (mask_img > 0).float()
        depth_img = depth_img * mask_img

        # Add batch dimension
        depth_img = depth_img.unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features, _ = model(depth_img)

        # Convert features to list and store in the result
        features_np = features.squeeze().cpu().numpy()

        # Define output_csv if not already defined
        output_csv = os.path.join(OUTPUT_FOLDER, "measurements.csv")

        # Read measurements from CSV
        measurements_df = pd.read_csv(output_csv)
        if measurements_df.empty:
            print(f"[ERROR] Measurements CSV is empty: {output_csv}")
            return "Measurements data missing.", 500

        # Extract the most recent row (assuming each row corresponds to an image in order)
        recent_measurements = measurements_df.iloc[-1]
        additional_features = [
            recent_measurements.get("Head_Hip_Ratio", 0),
            recent_measurements.get("Head_Waist_Ratio", 0),
            recent_measurements.get("Head_Height_Ratio", 0),
            recent_measurements.get("Hip_Waist_Ratio", 0),
            recent_measurements.get("Hip_Height_Ratio", 0),
            recent_measurements.get("Waist_Height_Ratio", 0),
            row.get('white_pixel_ratio', 0),
            row.get('white_neck_ratio', 0),
            row.get('white_hip_ratio', 0),
            row.get('white_knee_ratio', 0),
        ]

        # Combine features
        combined_features = np.concatenate((features_np, additional_features))

        # Save combined features in a new CSV with one feature per column
        combined_features_csv_path = os.path.join(OUTPUT_FOLDER, "combined_features_2570.csv")

        # Convert combined features to DataFrame and save to CSV
        combined_features_df = pd.DataFrame([combined_features])
        if not os.path.exists(combined_features_csv_path):
            combined_features_df.to_csv(combined_features_csv_path, index=False, header=False)
        else:
            with open(combined_features_csv_path, 'a') as f:
                combined_features_df.to_csv(f, index=False, header=False)

        print(f"Combined features saved to {combined_features_csv_path}")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total processing time: {total_duration:.4f} seconds")


    return render_template('selected_result4.html', results=results)

@app.route("/predict_bmi", methods=["GET", "POST"])
def predict_bmi():
    """2560개 features + 추가 features로 예측한 BMI 결과 및 키포인트 측정값 출력"""

    start_time = time.time()
    # Load combined features CSV
    combined_features_csv_path = os.path.join(OUTPUT_FOLDER, "combined_features_2570.csv")
    if not os.path.exists(combined_features_csv_path):
        return "Feature data not found. Please process an image first.", 400

    # Load the last row of features
    combined_features_df = pd.read_csv(combined_features_csv_path, header=None)
    if combined_features_df.empty:
        return "No features available for prediction.", 400
 
    if combined_features_df.empty:
        return "No features available for prediction.", 400

    features = combined_features_df.iloc[-1].values.reshape(1, -1)
    print("CSV 마지막 행 (예측 입력 데이터):", combined_features_df.iloc[-1].values)
 

    # Predict BMI using each model
    hgb_pred = hgb_model.predict(features)[0]
    lgbm_pred = lgbm_model.predict(features)[0]
    catboost_pred = catboost_model.predict(features)[0]

        # 모델 로딩 확인 로그
    print("Model Loaded (HGB):", hgb_model)
    print("Model Loaded (LGBM):", lgbm_model)
    print("Model Loaded (CatBoost):", catboost_model)
 
    # Ensemble prediction (average)
    ensemble_pred = (hgb_pred + lgbm_pred + catboost_pred) / 3
    output_csv = os.path.join(OUTPUT_FOLDER, "measurements.csv")
    measurements = {}
    if os.path.exists(output_csv):
        measurements_df = pd.read_csv(output_csv)
        measurements = measurements_df.iloc[-1].to_dict() if not measurements_df.empty else {}

    # Load measurements.csv path from MEASUREMENTS_JSON_PATH 
    
    print(f"Features CSV 경로: {combined_features_csv_path}")
    print(f"CSV 내용 미리보기: {combined_features_df.head()}")
    print(f"불러온 특성 미리보기: {combined_features_df.head()}")
    print(f"HGB 예측: {hgb_pred}")
    print(f"LGBM 예측: {lgbm_pred}")
    print(f"CatBoost 예측: {catboost_pred}")
    print(f"앙상블 예측 (BMI): {ensemble_pred}") 
    print("예측 입력 데이터:", features)

    # BMI 값 범위 설정
    min_bmi = 0
    max_bmi = 50
    predicted_bmi = ensemble_pred  # 예측된 BMI 값

    # BMI 기준 범위에 따른 가중치
    underweight_max = 18.5
    normal_max = 23
    overweight_max = 25
    max_bmi = 50  # 최대 BMI (미국 데이터 기준)

    if predicted_bmi <= underweight_max:
        pointer_position = (predicted_bmi / underweight_max) * 30  # 저체중 (0 ~ 30%)
    elif predicted_bmi <= normal_max:
        pointer_position = 30 + ((predicted_bmi - underweight_max) / (normal_max - underweight_max)) * 30  # 정상 (30% ~ 60%)
    elif predicted_bmi <= overweight_max:
        pointer_position = 60 + ((predicted_bmi - normal_max) / (overweight_max - normal_max)) * 20  # 과체중 (60% ~ 80%)
    else:
        pointer_position = 80 + ((predicted_bmi - overweight_max) / (max_bmi - overweight_max)) * 20  # 비만 (80% ~ 100%)

    pointer_position = round(pointer_position, 2)

    print(f"Calculated Pointer Position: {pointer_position}") 

    if ensemble_pred < 18.5:
        bmi_status = "low"
    elif 18.5 <= ensemble_pred < 23:
        bmi_status = "normal"
    elif 23 <= ensemble_pred < 25:
        bmi_status = "overweight"
    else:
        bmi_status = "fat"

    bmi_duration = time.time() - start_time
    print(f"BMI Prediction Duration: {bmi_duration:.4f} seconds")
      
    return render_template(
        "bmi_prediction_result.html",
        hgb_pred=hgb_pred,
        lgbm_pred=lgbm_pred,
        catboost_pred=catboost_pred,
        ensemble_pred=ensemble_pred,
        measurements=measurements,
        pointer_position = pointer_position,
        bmi_status=bmi_status

    )

@app.route('/low_weight')
def low_weight():
    return render_template('low_weight.html')

@app.route('/normal')
def normal():
    return render_template('normal.html')

@app.route('/overweight')
def overweight():
    return render_template('overweight.html')

@app.route('/fat')
def fat():
    return render_template('fat.html')


def compute_white_pixel_area(img_mask, y_start, y_end):
    """지정된 y 좌표 범위 내에서 흰색 픽셀 수 계산"""
    if img_mask is None:
        return 0
    cropped_area = img_mask[y_start:y_end, :]
    white_pixel_count = np.sum(cropped_area == 255)
    return white_pixel_count

def crop_person(image_path):
    """YOLO로 사람 영역 추출 및 자르기"""
    detection_results = detection_model(image_path, task="detect")
    result_image = detection_results[0]
    largest_box = None
    largest_area = 0

    for box in result_image.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        width, height = x2 - x1, y2 - y1
        area = width * height
        if area > largest_area:
            largest_area = area
            largest_box = (x1, y1, x2, y2)

    if largest_box:
        image = Image.open(image_path)
        cropped_image = image.crop(largest_box)
        cropped_path = os.path.join(OUTPUT_FOLDER, f"cropped_{os.path.basename(image_path)}")
        cropped_image.save(cropped_path)
        return cropped_path
    return image_path

def extract_keypoints_and_measurements(image_path, output_csv):
    """Extract keypoints, calculate body measurements, and save results to a CSV."""
    import os
    import math
    import pandas as pd


    # Perform pose estimation
    results = pose_model(image_path, task="pose")
    keypoints = results[0].keypoints.xy[0].cpu().numpy()

    # Define keypoint indices
    indices = {
        "head_length": (3, 4),  # Right ear and Left ear
        "hip_width": (18, 19),  # Right hipline and Left hipline
        "waist_width": (14, 13),  # Right waist and Left waist
        "height": (0, 20, 21)  # Nose and both knees
    }

    def calculate_distance(p1, p2):
        """Calculate the Euclidean distance between two points."""
        if p1 is not None and p2 is not None:
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return None

    def get_keypoint(idx):
        """Retrieve keypoint coordinates by index."""
        if idx < len(keypoints):
            return keypoints[idx]
        return None

    # Calculate distances
    head_length = calculate_distance(
        get_keypoint(indices["head_length"][0]), 
        get_keypoint(indices["head_length"][1])
    )

    hip_width = calculate_distance(
        get_keypoint(indices["hip_width"][0]), 
        get_keypoint(indices["hip_width"][1])
    )

    waist_width = calculate_distance(
        get_keypoint(indices["waist_width"][0]), 
        get_keypoint(indices["waist_width"][1])
    )

    # For height, calculate vertical distance (y-axis only) between Nose and the midpoint of both knees
    height = None
    if all(get_keypoint(idx) is not None for idx in indices["height"]):
        nose_y = get_keypoint(indices["height"][0])[1]
        knees_y = (get_keypoint(indices["height"][1])[1] + get_keypoint(indices["height"][2])[1]) / 2
        height = abs(nose_y - knees_y)

    # Calculate ratios
    def safe_divide(a, b):
        return a / b if a is not None and b is not None and b != 0 else None

    print("Extracted Keypoints Coordinates:")
    for i, (x, y) in enumerate(keypoints):
        print(f"Keypoint {i}: X={x:.2f}, Y={y:.2f}")

    measurements = {
        "Head_Length": head_length,
        "Hip_Width": hip_width,
        "Waist_Width": waist_width,
        "Height": height,
        "Head_Hip_Ratio": safe_divide(head_length, hip_width),
        "Head_Waist_Ratio": safe_divide(head_length, waist_width),
        "Head_Height_Ratio": safe_divide(head_length, height),
        "Hip_Waist_Ratio": safe_divide(hip_width, waist_width),
        "Hip_Height_Ratio": safe_divide(hip_width, height),
        "Waist_Height_Ratio": safe_divide(waist_width, height),
    }

    # Save measurements to a CSV
    df = pd.DataFrame([measurements])
    if not os.path.exists(output_csv):
        df.to_csv(output_csv, index=False)
        print(f"CSV file created: {output_csv}")
    else:
        df.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"Data appended to CSV file: {output_csv}")

    # Save keypoint visualization
    output_folder = os.path.dirname(output_csv)
    keypoint_image_path = os.path.join(output_folder, f"keypoints_{os.path.basename(image_path)}")
    results[0].save(keypoint_image_path)

    return measurements, keypoint_image_path

def process_depth(image_path):
    """깊이 이미지 생성"""
    image = Image.open(image_path)
    depth_result = depth_pipeline(image)["depth"]
    depth_path = os.path.join(DEPTH_OUTPUT_FOLDER, f"depth_{os.path.basename(image_path)}")

    plt.imshow(depth_result, cmap="inferno")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(depth_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return depth_path

def process_sam(image_path):
    """SAM으로 마스크 생성 및 저장"""
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    masks = mask_generator.generate(image_array)

    mask_paths = []
    for idx, mask in enumerate(masks):
        mask_image = Image.fromarray(mask["segmentation"].astype(np.uint8) * 255)
        mask_filename = f"mask_{os.path.splitext(os.path.basename(image_path))[0]}_{idx}.png"
        mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
        mask_image.save(mask_path)
        mask_paths.append(f"outputs/{mask_filename}")

    return mask_paths
 

import pandas as pd

 

if __name__ == "__main__":
    # OUTPUT_FOLDER 초기화
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DEPTH_OUTPUT_FOLDER, exist_ok=True)

    app.run(host="0.0.0.0", port=5000, debug=True)