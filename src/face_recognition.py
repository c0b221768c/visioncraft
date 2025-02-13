import cv2
import numpy as np
import onnxruntime as ort
import faiss
import os
import uuid

class FaceRecognizer:
    def __init__(self, model_path="models/face_recognition.onnx", db_path="data/database.faiss", threshold=0.6):
        """
        顔識別モデルの初期化
        :param model_path: 顔識別用 ONNX モデルのパス
        :param db_path: Faiss データベースのパス
        :param threshold: 類似度しきい値（デフォルト: 0.6）
        """
        self.threshold = threshold
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # ONNX モデルの入力情報取得
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # (1, 3, 112, 112)
        self.feature_dim = self.session.get_outputs()[0].shape[1]  # 特徴量の次元数（通常 512）

        # Faiss データベースのロード
        self.db_path = db_path
        self.index = self.load_faiss_db()

        # UUID リストのロード
        self.uuid_list = self.load_uuid_list()

    def load_faiss_db(self):
        """
        Faiss データベースをロード
        """
        if os.path.exists(self.db_path):
            return faiss.read_index(self.db_path)
        else:
            return faiss.IndexFlatL2(self.feature_dim)  # 新規作成（L2距離）

    def load_uuid_list(self):
        """
        UUID リストをロード（特徴量と紐づくIDリスト）
        """
        uuid_file = self.db_path.replace(".faiss", ".txt")
        if os.path.exists(uuid_file):
            with open(uuid_file, "r") as f:
                return [line.strip() for line in f.readlines()]
        return []

    def save_faiss_db(self):
        """
        Faiss データベースを保存
        """
        faiss.write_index(self.index, self.db_path)

        # UUID リストを保存
        uuid_file = self.db_path.replace(".faiss", ".txt")
        with open(uuid_file, "w") as f:
            f.writelines("\n".join(self.uuid_list))

    def preprocess(self, face):
        """
        顔画像の前処理（112x112 にリサイズし、ONNX 用のフォーマットに変換）
        """
        face = cv2.resize(face, (self.input_shape[2], self.input_shape[3]))
        face = face.astype(np.float32) / 255.0  # 0-1 に正規化
        face = np.transpose(face, (2, 0, 1))  # (H, W, C) → (C, H, W)
        face = np.expand_dims(face, axis=0)  # バッチ次元追加
        return face

    def extract_feature(self, face):
        """
        顔特徴量を ONNX モデルで抽出
        """
        input_tensor = self.preprocess(face)
        feature = self.session.run(None, {self.input_name: input_tensor})[0]
        return feature / np.linalg.norm(feature)  # 正規化

    def identify(self, face):
        """
        顔識別を行い、データベース内の最も類似する人物を検索
        """
        feature = self.extract_feature(face)

        if self.index.ntotal == 0:
            return None, 0.0  # データベースが空の場合

        # 類似検索（Faiss）
        distances, indices = self.index.search(feature, 1)
        best_match_idx = indices[0][0]
        best_match_dist = distances[0][0]

        if best_match_dist < self.threshold:  # しきい値以下なら認識成功
            return self.uuid_list[best_match_idx], best_match_dist
        return None, best_match_dist

    def register_face(self, face):
        """
        新しい顔をデータベースに登録（UUID で管理）
        """
        feature = self.extract_feature(face)
        new_uuid = str(uuid.uuid4())  # UUID 生成
        self.index.add(feature)
        self.uuid_list.append(new_uuid)
        self.save_faiss_db()
        return new_uuid  # 新規登録した UUID を返す
