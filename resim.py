import cv2
from deepface import DeepFace

# Resim dosyası yolu
image_path = "sadman.png"  # Resminizin tam yolu buraya girilmeli

try:
    # Resmi yükle
    frame = cv2.imread(image_path)

    if frame is None:
        print("Resim yüklenemedi! Lütfen geçerli bir yol girin.")
        exit()

    # DeepFace ile duygu analizi yap
    try:
        analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)

        # Birden fazla yüz için döngü
        if isinstance(analysis, list):
            for face in analysis:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                dominant_emotion = face['dominant_emotion']

                # Yüzün etrafına dikdörtgen çiz
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Duyguyu yüzün üzerine yaz
                font_scale = 1
                font_color = (255, 0, 0)  # Kırmızı renk
                font_thickness = 2
                position = (x, y - 10)  # Duygu analizi yüzün üst kısmına yazılır
                cv2.putText(frame, f"{dominant_emotion}", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        else:
            # Tek yüz durumunda
            x, y, w, h = analysis['region']['x'], analysis['region']['y'], analysis['region']['w'], analysis['region']['h']
            dominant_emotion = analysis['dominant_emotion']

            # Yüzün etrafına dikdörtgen çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Duyguyu yüzün üzerine yaz
            font_scale = 1
            font_color = (255, 0, 0)  # Kırmızı renk
            font_thickness = 2
            position = (x, y - 10)  # Duygu analizi yüzün üst kısmına yazılır
            cv2.putText(frame, f"{dominant_emotion}", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    except Exception as e:
        print(f"Analiz hatası: {e}")

    # Görüntüyü göster
    cv2.imshow("Duygu Analizi", frame)

    # Pencereyi kapatmak için bir tuşa basmayı bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Program hatası: {e}")
