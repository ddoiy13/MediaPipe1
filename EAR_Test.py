import cv2
import mediapipe as mp
import math

# 두 점 사이의 유클리드 거리를 계산하는 함수
def calculate_distance(p1, p2):
    return math.dist(p1, p2)

# EAR(Eye Aspect Ratio)을 계산하는 함수
def calculate_ear(eye_landmarks):
    # 세로 거리 (눈의 위아래 두 쌍의 점 사이 거리)
    vertical_1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    
    # 가로 거리 (눈의 양 끝점 사이 거리)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    
    # EAR 공식 적용
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# 눈 랜드마크 인덱스 (왼쪽/오른쪽 눈을 구성하는 6개의 점)
# 순서: 양끝점 시작(p1) -> 위쪽(p2, p3) -> 반대끝(p4) -> 아래쪽(p5, p6)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# 눈 감김 판별을 위한 기준값 (환경이나 사람에 따라 조정 필요)
EAR_THRESHOLD = 0.21

# 웹캠 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 좌우 반전 (거울 모드) 및 BGR을 RGB로 변환 (MediaPipe는 RGB를 사용)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 감지
    results = face_mesh.process(rgb_frame)
    
    # 화면의 크기 가져오기
    ih, iw, _ = frame.shape
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # 오른쪽 눈의 픽셀 좌표 추출
            right_eye_points = []
            for index in RIGHT_EYE_INDICES:
                point = face_landmarks.landmark[index]
                right_eye_points.append((int(point.x * iw), int(point.y * ih)))
                
            # 왼쪽 눈의 픽셀 좌표 추출
            left_eye_points = []
            for index in LEFT_EYE_INDICES:
                point = face_landmarks.landmark[index]
                left_eye_points.append((int(point.x * iw), int(point.y * ih)))
            
            # 각 눈의 EAR 계산
            right_ear = calculate_ear(right_eye_points)
            left_ear = calculate_ear(left_eye_points)
            
            # 두 눈의 EAR 평균값 계산
            avg_ear = (right_ear + left_ear) / 2.0
            
            # 화면에 랜드마크 점 그리기 (선택 사항)
            for point in right_eye_points + left_eye_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # EAR 값에 따라 상태 판별
            if avg_ear < EAR_THRESHOLD:
                status = "CLOSED"
                color = (0, 0, 255) # 빨간색
            else:
                status = "OPEN"
                color = (255, 0, 0) # 파란색
                
            # 화면에 텍스트 출력
            cv2.putText(frame, f"Eye Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 결과 화면 출력
    cv2.imshow('Eye Blink Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()