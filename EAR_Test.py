import cv2
import mediapipe as mp
import numpy as np

.# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

.# 눈 좌표 인덱스 (MediaPipe FaceMesh 기준)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    # 좌표 추출
    coords = []
    for i in eye_indices:
        lm = landmarks[i]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    
   . # EAR 공식: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

.# 젯슨 나노 카메라 설정 (CSI 카메라 사용 시 gstreamer 설정 필요, USB면 0)
cap = cv2.VideoCapture(0)

.# 눈 감음 판단 기준값 (테스트 후 조정 필요)
EAR_THRESHOLD = 0.22 

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # 성능을 위해 이미지 쓰기 불가능 설정 후 처리
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            
            avg_ear = (left_ear + right_ear) / 2.0

            # 눈 개폐 판단
            status = "Eyes Open"
            color = (0, 255, 0)
            if avg_ear < EAR_THRESHOLD:
                status = "Eyes Closed"
                color = (0, 0, 255)

            # 화면 표시
            cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, status, (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Jetson Nano - Eye Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27: # ESC로 종료
        break

cap.release()
cv2.destroyAllWindows()
