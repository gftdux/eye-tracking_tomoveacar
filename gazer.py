import cv2
import dlib
import numpy as np

# dlib의 얼굴 탐지기와 랜드마크 예측기를 로드합니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/seo_youll/Desktop/shape_predictor_68_face_landmarks.dat')  # 다운받아야 함

def get_eye_region(landmarks, points):
    return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points], np.int32)

def get_eye_center(eye_region):
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    return (min_x + max_x) // 2, (min_y + max_y) // 2

def detect_gaze_direction(current_left_eye_center, current_right_eye_center, fixed_left_eye_center, fixed_right_eye_center):
    # 두 눈의 중심을 평균내어 전체 눈동자 중심을 계산
    current_eye_center = np.array([(current_left_eye_center[0] + current_right_eye_center[0]) // 2, 
                                   (current_left_eye_center[1] + current_right_eye_center[1]) // 2])
    fixed_eye_center = np.array([(fixed_left_eye_center[0] + fixed_right_eye_center[0]) // 2, 
                                 (fixed_left_eye_center[1] + fixed_right_eye_center[1]) // 2])

    # 고정된 점과 현재 눈 중심의 상대 위치 계산
    dx = current_eye_center[0] - fixed_eye_center[0]
    dy = current_eye_center[1] - fixed_eye_center[1]

    # 좌우 및 상하 방향 판별
    if abs(dx) > 2:
        return 4 if dx > 0 else 3  # 오른쪽 or 왼쪽
    elif abs(dy) > 2:
        return 2 if dy > 0 else 1  # 아래 or 위
    else:
        return 0  # 정면

# 웹캠 피드를 캡처합니다.
cap = cv2.VideoCapture(0)

fixed_left_eye_center = None
fixed_right_eye_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])
        
        left_eye_center = get_eye_center(left_eye_region)
        right_eye_center = get_eye_center(right_eye_region)
        
        if left_eye_center and right_eye_center:
            if fixed_left_eye_center is not None and fixed_right_eye_center is not None:
                # 고정된 점 표시
                cv2.circle(frame, fixed_left_eye_center, 4, (0, 255, 255), -1)  # 노란색 점, 반지름 5
                cv2.circle(frame, fixed_right_eye_center, 4, (0, 255, 255), -1)  # 노란색 점, 반지름 5

                # 현재 눈동자 중심 표시
                cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)  # 초록색 점
                cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)  # 초록색 점
                
                # 시선 방향 판별
                gaze_direction = detect_gaze_direction(left_eye_center, right_eye_center, fixed_left_eye_center, fixed_right_eye_center)
                print(gaze_direction)
            else:
                # 첫 고정 상태에서 현재 눈동자 중심 표시
                cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)  # 초록색 점
                cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)  # 초록색 점
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)  # 1ms 대기, 프레임 처리 간격을 줄임
    if key == ord('w'):  # w 키
        fixed_left_eye_center = left_eye_center
        fixed_right_eye_center = right_eye_center
        print("Fixed eye centers:", fixed_left_eye_center, fixed_right_eye_center)
    elif key == ord('q'):  # q 키
        break

cap.release()
cv2.destroyAllWindows()
