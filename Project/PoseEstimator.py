import cv2 as cv
import pyopencl as cl

class PoseEstimator:
    def __init__(self):
        ### Settings #######################
        self.img_width = 1280
        self.img_height = 720

        # open pose #

        self.protoFile = ".\\Project\\pose_models\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = ".\\Project\\pose_models\\mpi\\pose_iter_160000.caffemodel"

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

        self.THRESHOLD = 0.05 # this is the openpose THRESHOLD

        # haar #

        self.haar = cv.CascadeClassifier(
            ".\\Project\\haar_models\\haarcascade_frontalface_default.xml"
        )

        # template matching #

        self.face_template_location = ".\\Project\\face_template.png"
        
        ### End Settings #######################

        self._init_camera()
        self._init_network()
        self._init_opencl()
        self._init_kernels()

    def _init_opencl(self):
        try:
            plaforms = cl.get_platforms()
            global plaform
            plaform = plaforms[0]
            devices = plaform.get_devices()
            global device
            device = devices[0]
            global ctx
            self.ctx = cl.Context(devices) # or dev_type=cl.device_type.ALL)
            global commQ
            self.commQ = cl.CommandQueue(self.ctx,device)
            file = open(".\\Project\\prog.c","r")
            global prog
            try:
                self.prog  = cl.Program(self.ctx, file.read()).build()
            except Exception as e:
                print("Error:", e)
        except Exception as e:
            print(e)
            return False
        
    def _init_kernels(self):
        self.kernel_subtract_val_to_img = cl.Kernel(self.prog, "subtract_val_to_img")
        self.kernel_img_mult = cl.Kernel(self.prog, "img_mult_pix2pix")
        self.kernel_ncc_tiled = cl.Kernel(self.prog, "ncc_tiled")

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