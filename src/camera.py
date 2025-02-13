import cv2

class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Init
        :param camera_index:
        :param width:
        :param height:
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError('...Can not open camera')
    
    def get_frame(self):
        """
        Get frame
        :return:
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        """
        Release camera
        """
        self.cap.release()