import cv2 as cv
import pyopencl as cl

class PoseEstimator:
    def __init__(self):
        ### Settings #######################
        self.img_width = 1280
        self.img_height = 720

        self.crop_size = 368
        self.tutorial = 1
        self.play_prompt = "Do you want to play a game?"
        self.box_shape = [self.img_width//9, self.img_height//9]
        self.wait_between_rounds = 2
        self.time_per_round = 15
        self.time_decay = 0.9
        self.boxes_colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 255, 255]]

        self.game_colors = ["Blue", "Green", "Red", "Yellow"]
        self.face_detection_threshold = 0.7

        # open pose #

        self.protoFile = ".\\Project\\pose_models\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = ".\\Project\\pose_models\\mpi\\pose_iter_160000.caffemodel"

        self.THRESHOLD = 0.05 # this is the openpose THRESHOLD

        # DNN face detection
        self.DNN = cv.dnn.readNetFromTensorflow(".\\Project\\DNN_models\\opencv_face_detector_uint8.pb", ".\\Project\\DNN_models\\opencv_face_detector.pbtxt")

        # template matching #

        self.face_template_location = ".\\Project\\face_template_2.png"
        
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
        self.kernel_linear_transform_img = cl.Kernel(self.prog, "linear_transform_img")

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