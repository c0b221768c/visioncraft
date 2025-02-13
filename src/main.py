import cv2
import numpy as np
from src.camera import Camera
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from src.sender import FaceDataSender

# カメラ設定
camera = Camera()

# モデルのロード
face_detector = FaceDetector()
face_recognizer = FaceRecognizer(model_path="/app/models/face_recognition.onnx", db_path="/app/data/database.faiss")

# TCP
face_sender = FaceDataSender(server_ip="127.0.0.1", server_port=5002)

def main():
    while True:
        # 1️⃣ カメラ映像を取得
        frame = camera.get_frame()
        if frame is None:
            continue

        # 2️⃣ 顔検出（最大の顔のみ）
        face_boxes = face_detector.detect_faces(frame)
        if len(face_boxes) == 0:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == 27:  # ESCキーで終了
                break
            continue

        x1, y1, x2, y2 = face_boxes[0]  # 最大の顔を取得
        face_crop = frame[y1:y2, x1:x2]

        # 3️⃣ 顔識別（Faissで類似検索）
        identity, score = face_recognizer.identify(face_crop)

        if identity:
            label = f"{identity} ({score:.2f})"
            color = (0, 255, 0)  # 既存客: 緑
            face_sender.send_face_data(identity, is_new=False)
        else:
            # 新規客の場合は UUID で登録
            new_uuid = face_recognizer.register_face(face_crop)
            label = f"{new_uuid} (New)"
            color = (0, 0, 255)  # 新規客: 赤
            face_sender.send_face_data(new_uuid, is_new=True)

        # 5️⃣ 結果を描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
