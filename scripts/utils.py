import torch
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# YOLOv5 허브에서 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/git/img_mk/models/yolov5/best.pt')

# 클래스와 해당 이모지 이미지 경로 매핑 정의
EMOTION_EMOJI_IMAGES = {
    'angry': "D:/git/img_mk/dataset/imoji/angry/angry.png",
    'anxiety': "D:/git/img_mk/dataset/imoji/anxiety/anxiety.jpg",
    'happy': "D:/git/img_mk/dataset/imoji/happy/happy.jpg",
    'neutrality': "D:/git/img_mk/dataset/imoji/neutrality/neutrality.png",
    'panic': "D:/git/img_mk/dataset/imoji/panic/panic.jpg",
    'sad': "D:/git/img_mk/dataset/imoji/sad/sad.jpg",
    'wound': "D:/git/img_mk/dataset/imoji/wound/wound.jpg"
}

# 감정 감지 및 이모지 추가 함수
def detect_and_map_emojis(image_path, model, emoji_dict):
    # 이미지 로드
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # YOLO 감지 수행
    results = model(img_rgb)
    
    # 이모지 오버레이를 위해 PIL로 변환
    img_pil = Image.fromarray(img_rgb)
    detections = results.pandas().xyxy[0]
    
    for _, row in detections.iterrows():
        # 경계 상자와 클래스 레이블 추출
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name'].lower()
        
        # 감지된 레이블이 이모지 사전에 있는지 확인
        if label in emoji_dict:
            # 이모지 이미지 로드
            emoji_img_path = emoji_dict[label]
            emoji_img = Image.open(emoji_img_path).convert("RGBA")
            
            # 경계 상자 크기에 맞게 이모지 크기 조정
            emoji_img = emoji_img.resize((xmax - xmin, ymax - ymin))
            
            # 이모지를 원본 이미지에 붙이기
            img_pil.paste(emoji_img, (xmin, ymin), emoji_img)
    
    # 저장/표시를 위해 배열로 변환
    img_with_emoji = np.array(img_pil)
    return img_with_emoji

# 디렉토리를 순회하며 이미지 처리
INPUT_DIR = 'D:/kor_face_ai/real_t'
OUTPUT_DIR = 'D:/git/img_mk/results/output_images'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for emotion in os.listdir(INPUT_DIR):
    emotion_dir = os.path.join(INPUT_DIR, emotion)
    output_emotion_dir = os.path.join(OUTPUT_DIR, emotion)
    os.makedirs(output_emotion_dir, exist_ok=True)
    
    if os.path.isdir(emotion_dir):
        for idx, img_name in enumerate(os.listdir(emotion_dir)):
            if idx >= 10:
                break
            img_path = os.path.join(emotion_dir, img_name)
            output_path = os.path.join(output_emotion_dir, img_name)
            
            # 감정 감지 및 이모지 매핑
            img_with_emoji = detect_and_map_emojis(img_path, model, EMOTION_EMOJI_IMAGES)
            
            # 출력 이미지 저장
            img_with_emoji_bgr = cv2.cvtColor(img_with_emoji, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_with_emoji_bgr)

print("처리가 완료되었습니다. 이모지가 추가된 이미지가 저장되었습니다.")