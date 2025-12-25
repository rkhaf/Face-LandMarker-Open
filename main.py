import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup detektor
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=".venv/Lib/site-packages/face_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
)

# Koneksi utama untuk wajah (simplified)
FACE_CONNECTIONS = [
    # Outline wajah
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), 
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    
    # Mata kiri (lingkaran)
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), 
    (153, 154), (154, 155), (155, 133), (133, 173), (173, 157),
    (157, 158), (158, 159), (159, 160), (160, 161), (161, 246),
    
    # Mata kanan (lingkaran)
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (362, 398), (398, 384),
    (384, 385), (385, 386), (386, 387), (387, 388), (388, 466),
    
    # Bibir
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), 
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (291, 375), (375, 321), (321, 405), (405, 314), (314, 17),
    (17, 84), (84, 181), (181, 91), (91, 146), (146, 61)
]

# Buka kamera
cap = cv2.VideoCapture(0)

print("Tekan 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Deteksi
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    
    # Gambar titik dan garis
    if result.face_landmarks:
        face = result.face_landmarks[0]
        
        # Gambar garis dulu (background)
        for start_idx, end_idx in FACE_CONNECTIONS:
            if start_idx < len(face) and end_idx < len(face):
                x1 = int(face[start_idx].x * w)
                y1 = int(face[start_idx].y * h)
                x2 = int(face[end_idx].x * w)
                y2 = int(face[end_idx].y * h)
                
                # Warna berdasarkan area
                if 0 <= start_idx <= 16:  # Outline wajah
                    color = (0, 255, 255)  # Kuning
                elif 33 <= start_idx <= 155:  # Mata kiri
                    color = (255, 0, 0)    # Biru
                elif 263 <= start_idx <= 388:  # Mata kanan
                    color = (0, 255, 0)    # Hijau
                else:  # Bibir
                    color = (255, 0, 255)  # Magenta
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)
        
        # Gambar titik-titik (overlay)
        for idx, landmark in enumerate(face):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Titik lebih kecil
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    
    # Tampilkan
    cv2.imshow('Face Landmarker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()