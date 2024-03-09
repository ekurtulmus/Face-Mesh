import cv2
import time
import mediapipe as mp

# Video dosyasını aç
cap = cv2.VideoCapture("video ismi")

# Yüz tespiti için mediapipe kütüphanesini kullan
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1) # Sadece bir yüzü tespit etmek için
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1) # Çizim spesifikasyonları

# Zaman ölçümleri için başlangıç zamanı
pTime = 0
while True:
    # Video karesini oku
    success, img = cap.read()
    # Karesini RGB'ye dönüştür
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Yüz tespiti yap
    results = faceMesh.process(imgRGB)
    print(results.multi_face_landmarks) # Tespit edilen yüzlerin koordinatlarını yazdır
    
    # Tespit edilen yüz varsa
    if results.multi_face_landmarks:
        # Her bir yüz için
        for faceLms in results.multi_face_landmarks:
            # Yüzün çizgilerini çiz
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec) # FACEMESH_CONTOURS
    
        # Her bir yüzdeki noktaları işle
        for id, lm in enumerate(faceLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print([id, cx, cy]) # Noktaların koordinatlarını yazdır
    
    # FPS hesapla
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # FPS'i ekrana yazdır
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  
    
    # İşlenmiş kareyi göster
    cv2.imshow("img", img)
    # Bir tuşa basılmasını bekle (50ms)
    cv2.waitKey(50)