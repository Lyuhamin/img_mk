import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import emoji

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\\git\\img_mk\\models\\yolov5\\best.pt')  # 학습된 모델 불러오기

# 감정과 이모지 매핑
emotion_to_emoji = {
    'angry': emoji.emojize(':angry_face:'),
    'Anxiety': emoji.emojize(':worried_face:'),
    'happy': emoji.emojize(':smiling_face_with_smiling_eyes:'),
    'neutrality': emoji.emojize(':neutral_face:'),
    'Panic': emoji.emojize(':fearful_face:'),
    'sad': emoji.emojize(':disappointed_face:'),
    'Wound': emoji.emojize(':face_with_head_bandage:')
}

# 웹캠 설정
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 샤프닝 필터 적용
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

    # YOLOv5로 예측 얻기
    results = model(frame)

    # 결과 파싱q
    detected_objects = results.xyxy[0].cpu().numpy()  # 결과를 NumPy 배열로 변환
    for obj in detected_objects:
        label = model.names[int(obj[5])]
        x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

        # 프레임에 경계 상자와 이모지 그리기
        if label in emotion_to_emoji:
            emoji_text = emotion_to_emoji[label]
            # 경계 상자 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 이모지 그리기
            cv2.putText(frame, emoji_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 감지 결과가 있는 프레임 표시
    cv2.imshow('Emotion Detection', frame)

    # 'q' 키가 눌리면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
