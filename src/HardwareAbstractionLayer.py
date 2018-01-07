"""
This module defines the elements supporting the generic interface to the camera controllers.  A camera controller
is a network device that in turn provides two camera interfaces.
"""
import datetime
import json
import math

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from ProcessAbstractionLayer import EDataDirection


class Configuration(object):
    """ Configurable settings for the hardware abstraction layer

    This is used to configure the system for a particular hardware interface.  It is important to note
    the system has no real idea of "hardware" beyond the assumption that, if asked nicely, it will cough
    up imager data. Actual management of the hardware and providing REST and other endpoints is the function
    of the controller executive, the source of which can be found at.....
    """
    with open('../resources/HardwareAbstractionLayer.json') as data_file:
        config_dict = json.load(data_file)
        IMAGER_MANAGER = str(config_dict["IMAGER_MANAGER"])
        CONTROLLERS = str(config_dict["CONTROLLERS"]).split(",")
        NUM_CONTROLLERS = len(CONTROLLERS)
        PORT = config_dict["PORT"]
        NUM_IMAGERS_CONTROLLER = int(config_dict["NUM_IMAGERS_CONTROLLER"])


class Controller:
    """Maps to physical ring hardware controller with n cameras


    This is meant to be subclassed so that specific subclasses may implement the get_imager_data
    function with respect to their particular hardware environment -
    Attributes:
        resource(str): Identifies the IP or domain name of the hardware controller controller
        port(int): Identifies the port uses for communication with the controller

    """

    Controllers = []
    ControllersByName = {}

    @staticmethod
    def initialize_controllers():
        """ Create the controllers based on the configuration file

        This must be called once at startup to initialize the individual controllers for all attached
        imagers
        """
        classfactory = globals()[Configuration.IMAGER_MANAGER]
        for i, controller in enumerate(Configuration.CONTROLLERS):
            controller_manager = classfactory(controller, Configuration.PORT, 0, i)
            Controller.Controllers.append(controller_manager)
            Controller.ControllersByName[controller] = controller_manager

    @staticmethod
    def on_log_message(client, userdata, message):
        print(message.topic)
        print(message.payload)

    @staticmethod
    def on_camera_message(client, userdata, message):
        termz = message.topic.split('/')

        camno = int(termz[1])

        if camno < 0 or camno > 1:
            print("bad camera mqtt mssg for " + message.topic)
            return
        if termz[2] == "raw" or termz[2] == 'rectified':
            client.controller.imagers[camno].process_live_image(bytearray(message.payload), termz[2])
        elif termz[2] == "status":
            client.controller.imagers[camno].status = message.payload.decode()

        if client.controller.imagers[camno].updateNotifier is not None:
            client.controller.imagers[camno].updateNotifier()

    def __init__(self, resource, port, elevation, resource_id) -> None:
        """Initialize a hardware controller and the hardware imagers under it's control

        :param resource: IP or DNS name of the server
        :type resource: str
        :param port: Assigned TCP port number on server
        :type port: int
        :param elevation:  vertical angle to initially assign to imager
        :type elevation: radian
        """
        super().__init__()
        self.resource = resource
        self.resource_id = resource_id
        self.port = port
        # spin up the imagers
        self.imagers = []
        step = (2.0 * math.pi) * (1.0 / Configuration.NUM_IMAGERS_CONTROLLER)  # step is function of cameras
        for i in range(0, Configuration.NUM_IMAGERS_CONTROLLER):
            ang = (i * step) + (step * 0.5)
            a_camera = Imager(self, i, np.array([elevation, ang]), i)
            self.imagers.append(a_camera)

        # initialize the camera image receiver
        self.mqtt_client = mqtt.Client(self.resource)
        self.mqtt_client.controller = self
        self.mqtt_client.connect(self.resource, port=1883, keepalive=240, bind_address="")
        self.mqtt_client.on_message = Controller.on_camera_message
        self.mqtt_client.subscribe("cam/#")
        self.mqtt_client.loop_start()
        # self.log_topic_handler.subscribe("log/trace/#")

    def publish_message(self, cameraId, subtopic, message):
        topic = 'camctl/{}/{}'.format(cameraId, subtopic)
        self.mqtt_client.publish(topic, message)

    def get_imager(self, ordinal):
        return self.imagers[ordinal]

    def get_imager_data(self, imager) -> bytearray:
        return None


class Imager(object):
    """Maps to a physical imager attached to a controller

    Attributes:
        controller(Controller): The hardware controller controller this imager is attached to
        index(int): The ordinal identifier of the camera in its host controller
        rawImageData(bytearray): The raw data read from the imager
        cvImage(Mat): OpenCV format representation of image from raw data
        attitude(np.array([2])): defines the elevation and yaw angles of the imager
        """

    def __init__(self, controller, camIndex, attitude, imager_address) -> None:
        """Initialize an instance of a hardware imager communication interface
        :param controller: The Hardwarecontroller the imager is physically connected to
        :type controller: QuameracontrollerController
        :param camIndex: The ordinal assigned by the hardware controller to identify this camera
        :type camIndex: int
        :param attitude: Defines camera elevation and pan angles respectively
        :type attitude: np.array([2])
        """
        super().__init__()
        self.imager_address = imager_address
        self.display = None
        self.controller = controller
        self.index = camIndex
        self.rawImageData = None
        self.cvImage = None
        self.attitude = attitude
        self.updateNotifier = None

        # this is a clone of the reset_samples() logic so that there are no complaints about where member vars defined
        self.image_counter = 0
        self.last_time = None
        self.framerate_sum = datetime.timedelta(0)

        self.processes = {}
        self.status = "ONLINE"
        self.channel = "UNKNOWN"
        self.channel_images = {}

    def reset_samples(self):
        """
        Make all of the old bad performance sampling disappear
        :return:
        :rtype:
        """
        self.image_counter = 0
        self.last_time = None
        self.framerate_sum = datetime.timedelta(0)

    def get_image(self, channel):
        """
        Get image from a specific channel
        :param channel:
        :type channel:
        :return:
        :rtype:
        """
        if channel in self.channel_images:
            return self.channel_images[channel]
        return None

    def process_live_image(self, rawdata, channel):
        """ Asyncronous update of images received from the controller/imager and triggered dispatch of processes


        :param channel: defines the mqtt channel the image was received on
        :type channel: basestring
        :param rawdata:  byte array representing a legal JPEG
        :type rawdata: bytearray
        :return: nothing
        :rtype: Mone
        """

        self.rawImageData = rawdata
        self.channel = channel
        try:
            # store image for channel
            self.channel_images[channel] = cv2.imdecode(np.asarray(self.rawImageData, dtype="uint8"), cv2.IMREAD_COLOR)
            self.dispatch_processes()
            if channel == 'raw':
                self.image_counter += 1
        except:
            self.rawImageData = None

    def send_control_message(self, message, body):
        """
        Send a control message for this imager
        :param message:
        :type message:
        :param body:
        :type body:
        :return:
        :rtype:
        """
        topic = "camctl/{}/{}".format(self.index, message)
        self.controller.client.publish(topic, body)

    def dispatch_processes(self):
        for name, idef_process in self.processes.items():
            if idef_process.get_container('RAWIMAGE').data_direction == EDataDirection.Output:
                idef_process.get_container('RAWIMAGE').matrix = self.get_image('raw')
                idef_process.get_container('RAWIMAGE').data_direction = EDataDirection.Input

            idef_process.process()
            if idef_process.completed:
                # now what the hell are you going to do ??????
                del self.processes[name]

    def calibration_filename(self):
        """
        Location of precomputed calibration file with camera intrinsics
        :return: Location of precomputed calibration file with camera intrinsics
        :rtype: basestring
        """
        filename = "calibration_{}_{}.json".format(self.controller.resource,
                                                   self.imager_address)
        return "../resources/" + filename
