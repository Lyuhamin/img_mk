import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import os
import sys

# YOLOv5 모델 불러오기
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path="D:\\git\\img_mk\\models\\yolov5\\best.pt")

# 감정 분류 모델 정의 및 상호 사전 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# InceptionV3 기반의 모델 정의
emotion_model = models.inception_v3(pretrained=False)
emotion_model.fc = torch.nn.Sequential(
    torch.nn.Linear(emotion_model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 7),  # 감정 클래스 개수 (예시로 7개 클래스 사용)
    torch.nn.LogSoftmax(dim=1)
)
emotion_model.load_state_dict(torch.load("D:\\git\\img_mk\\models\\emotion_model.pth"))
emotion_model = emotion_model.to(device)
emotion_model.eval()

# 이모지 매핑 딕셔너리
emotion_to_emoji = {
    '0': '😠',
    'anxiety': '😰',
    'happy': '😊',
    'neutrality': '😐',
    'panic': '😱',
    'sad': '😢',
    'wound': '🧕'
}

# 이미지 폴더 경로
image_folder = 'D:/kor_face_ai/real_t'

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 입력 크기에 맞춰 조정
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 이미지 출력 폴더 경로 설정
output_folder = "D:\\git\\img_mk\\results\\output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 폴더 내 모든 이미지 처리
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
total_images = len(image_files)

# 이미지 처리 함수
def process_image(image_path, output_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    results = yolo_model(img)
    if len(results.xyxy[0]) == 0:
        print(f"No objects detected in image: {image_path}")
        return

    # 바운딩 박스 얻기
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, det)
        face_img = img[y1:y2, x1:x2]

        # 얼굴 이미지 PIL 형식으로 변환 후 감정 예측
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        emotion_output = emotion_model(face_tensor)
        _, predicted = torch.max(emotion_output, 1)
        emotion_label = predicted.item()

        # 감정 레이블에 따라 이모지 선택
        emotion = list(emotion_to_emoji.keys())[emotion_label]
        emoji = emotion_to_emoji[emotion]
        
        # 디버깅용 콘솔 출력
        print(f"Predicted emotion: {emotion}, Emoji: {emoji}")

        # 이미지에 이모지 매핑
        try:
            cv2.putText(img, emoji, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            # 폰트 문제로 이모지 출력이 안 되는 경우 감정 텍스트 출력
            print(f"Failed to render emoji due to: {e}")
            cv2.putText(img, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 바운딩 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 결과 저장 (JPG 형식으로 저장)
    if not cv2.imwrite(output_path.replace('.png', '.jpg'), img):
        print(f"Error saving image: {output_path}")

# 이미지 포맷 전체 처리
for idx, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, image_file)
    output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.png', '.jpg'))
    
    # 이미지 처리 시작
    process_image(image_path, output_path)
    
    # 진행상태 알림 출력
    progress = (idx / total_images) * 100
    sys.stdout.write(f"\rProgress: {idx}/{total_images} ({progress:.2f}%)")
    sys.stdout.flush()

print("\nAll images have been processed. Results saved to:", output_folder)
