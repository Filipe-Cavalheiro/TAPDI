import cv2 as cv

class PoseEstimator:
    def __init__(self):
        ### Settings #######################
        self.img_width = 1280
        self.img_height = 720

        self.protoFile = ".\\pose_models\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = ".\\pose_models\\mpi\\pose_iter_160000.caffemodel"

        self.POSE_PAIRS = [
            ("Head", "Neck"),
            ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
            ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
            ("Neck", "Chest"),
            ("Chest", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
            ("Chest", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle")
        ]

        self.BODY_PARTS = {
            "Head": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RHip": 8,
            "RKnee": 9,
            "RAnkle": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "Chest": 14
        }

        self.THRESHOLD = 0.05

        self.haar = cv.CascadeClassifier(
            ".\\haar_models\\haarcascade_frontalface_default.xml"
        )
        ### End Settings #######################

        self._init_camera()
        self._init_network()

    def _init_camera(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.img_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.img_height)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def _init_network(self):
        self.net = cv.dnn.readNetFromCaffe(
            self.protoFile,
            self.weightsFile
        )