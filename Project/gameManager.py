import json
import cv2 as cv
import pyopencl as cl

class GameManager:
    def __init__(self):
        ### Settings #######################

        self.json_file_location = ".\\Project\\settings.json"

        json_file = None

        with open(self.json_file_location, 'r') as f:
            json_file = json.load(f)

        if json_file is None:
            raise FileNotFoundError

        self.tutorial = json_file["display_tutorial"]
        self.img_width = json_file["image_width"]
        self.img_height = json_file["image_height"]

        self.crop_size = json_file["open_pose_image_size"]
        self.wait_between_rounds = json_file["time_between_rounds_s"]
        self.time_per_round = json_file["time_of_round_s"]
        self.time_decay = json_file["round_time_decay"]
        self.play_prompt = json_file["initial_prompt"]

        self.boxes_colors = json_file["box_colors"]
        self.game_colors = json_file["game_colors"]
        self.face_detection_threshold = json_file["face_detection_confidence_level"]

        # open pose #

        self.protoFile = json_file["pose_model_proto_file"]
        self.weightsFile = json_file["pose_model_weights_file"]

        self.THRESHOLD = json_file["open_pose_threshold_confidence_level"] # this is the openpose THRESHOLD

        # DNN face detection
        self.DNN = cv.dnn.readNetFromTensorflow(json_file["DNN_model_pb_file"], json_file["DNN_model_pbtxt_file"])

        # template matching #
        self.face_template_location = json_file["face_template_location"]

        self.box_shape = [self.img_width//9, self.img_height//9]
        
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