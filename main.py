

#****************************************************************************
# **                              DÜZCE ÜNİVERSİTESİ
# **                          LİSANSÜSTÜ EĞİTİM ENSTİTÜSÜ
# **                       BİLGİSAYAR MÜHENDİLİĞİ ANABİLİM DALI
# **                       ÖĞRENCİ ADI :          ARDA ÖZYAMAN
# **                       ÖĞRENCİ NUMARASI :     2345007016
# **
# ****************************************************************************/

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import cv2
from deepface import DeepFace

# Yüz tanıma için Haar Cascade sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video yakalamayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Her kareyi yakala
    ret, frame = cap.read()

    # Kareyi gri tonlamaya dönüştür
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gri tonlamalı kareyi RGB formata dönüştür
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Karede yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Yüz ROI (İlgi Alanı) çıkar
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Yüz ROI üzerinde duygu analizi yap
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Baskın duyguyu belirle
        emotion = result[0]['dominant_emotion']

        # Yüzün etrafına dikdörtgen çiz ve tahmin edilen duyguyu etiketle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Sonuç kareyi göster
    cv2.imshow('Gerçek Zamanlı Duygu Tespiti', frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Yakalamayı serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()
