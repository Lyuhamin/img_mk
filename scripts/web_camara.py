import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# YOLOv5 허브에서 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/git/img_mk/models/yolov5/best.pt')


# 클래스와 해당 이모지 이미지 경로 매핑 정의
EMOTION_EMOJI_IMAGES = {
    'angry': "D:/git/img_mk/dataset/imoji/angry/angry.png",
    'anxiety': "D:/git/img_mk/dataset/imoji/anxiety/anxiety.png",
    'happy': "D:/git/img_mk/dataset/imoji/happy/happy.png",
    'neutrality': "D:/git/img_mk/dataset/imoji/neutrality/neutrality.png",
    'panic': "D:/git/img_mk/dataset/imoji/panic/panic.png",
    'sad': "D:/git/img_mk/dataset/imoji/sad/sad.png",
    'wound': "D:/git/img_mk/dataset/imoji/wound/wound.png"
}

# 웹캠에서 실시간 감정 감지 및 이모지 매핑 함수
def detect_and_map_emojis_webcam(model, emoji_dict):
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        # RGB로 변환
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO 감지 수행
        results = model(img_rgb)
        
        # 이모지 오버레이를 위해 PIL로 변환
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
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
                
                # 텍스트(감정 레이블) 추가
                font = ImageFont.load_default()
                draw.text((xmin, ymin - 10), label, fill=(255, 0, 0), font=font)
        
        # 배열로 변환 후 출력
        img_with_emoji = np.array(img_pil)
        img_with_emoji_bgr = cv2.cvtColor(img_with_emoji, cv2.COLOR_RGB2BGR)
        
        # 프레임 보여주기
        cv2.imshow('Emotion Detection with Emoji', img_with_emoji_bgr)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 웹캠 및 모든 창 종료
    cap.release()
    cv2.destroyAllWindows()

# 웹캠에서 감정 감지 수행
detect_and_map_emojis_webcam(model, EMOTION_EMOJI_IMAGES)
