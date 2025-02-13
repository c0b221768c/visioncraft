import socket
import json

class FaceDataSender:
    def __init__(self, server_ip="192.168.1.100", server_port=5002):
        """
        TCP で JSON を送信するクラス
        :param server_ip: メインコンテナの IP
        :param server_port: メインコンテナのポート番号
        """
        self.server_ip = server_ip
        self.server_port = server_port

    def send_face_data(self, identity, is_new):
        """
        顔識別の結果を TCP で送信
        """
        data = {
            "identity": identity if identity else "new",
            "is_new": is_new
        }
        json_data = json.dumps(data)

        try:
            # TCP ソケットを作成し、データを送信
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_ip, self.server_port))
                sock.sendall(json_data.encode('utf-8'))
                print(f"✅ 送信成功: {json_data}")
        except Exception as e:
            print(f"❌ 送信失敗: {str(e)}")
