"""
This module defines the elements supporting the generic interface to the camera controllers.  A camera controller
is a network device that in turn provides two camera interfaces.
"""
import datetime
import json
import math

import cv2
import io
import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image, ExifTags

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
    def initialize_controllers(masterControl):
        """ Create the controllers based on the configuration file

        This must be called once at startup to initialize the individual controllers for all attached
        imagers
        :param masterControl:
        :type masterControl:
        """

        classfactory = globals()[Configuration.IMAGER_MANAGER]
        for i, controller in enumerate(Configuration.CONTROLLERS):
            controller_manager = classfactory(controller, Configuration.PORT, i, masterControl)
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

    def __init__(self, resource, port, resource_id, master_control) -> None:
        """Initialize a hardware controller and the hardware imagers under it's control

        :param resource: IP or DNS name of the server
        :type resource: str
        :param port: Assigned TCP port number on server
        :type port: int
        """
        super().__init__()
        self.master_control = master_control
        self.resource = resource
        self.resource_id = resource_id
        self.port = port
        # spin up the imagers
        self.imagers = []
        step = (2.0 * math.pi) * (1.0 / 2)  # step is function of cameras
        for i in range(0, 2):
            ang = (i * step) + (step * 0.5)
            a_camera = Imager(self, i, np.array([0, ang]), i)
            self.imagers.append(a_camera)

        # initialize the camera image receiver
        self.mqtt_client = mqtt.Client(self.resource)
        self.mqtt_client.controller = self
        self.mqtt_client.connect(self.resource, port=1883, keepalive=240, bind_address="")
        self.mqtt_client.on_message = Controller.on_camera_message
        self.mqtt_client.subscribe("cam/#")
        self.mqtt_client.loop_start()
        # self.log_topic_handler.subscribe("log/trace/#")

    def antipode(self, imager):
        """
        Return the opposite camera imager
        :param imager:
        :type imager:
        :return: the opposite imager
        :rtype: Imager
        """
        if self.imagers[0] == imager:
            return self.imagers[1]
        else:
            return self.imagers[0]

    def publish_message(self, cameraId, subtopic, message):
        topic = 'camctl/{}/{}'.format(cameraId, subtopic)
        self.mqtt_client.publish(topic, message)

    def get_imager(self, ordinal):
        return self.imagers[ordinal]

    def get_imager_data(self, imager) -> bytearray:
        return None


class ImagerImageData(object):
    def __init__(self, host, camera_index, image_number, timestamp):
        self.host = host
        self.camera_index = camera_index
        self.image_number = image_number
        self.timestamp = timestamp


class Imager(object):
    """Maps to a physical imager attached to a controller

    Attributes:
        controller(Controller): The hardware controller controller this imager is attached to
        index(int): The ordinal identifier of the camera in its host controller
        rawImageData(bytearray): The raw data read from the imager
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
        self.updateNotifier = None

        # must define vars prior to calling reset to make analyzers happy
        self.local_image_counter = 0
        self.last_time = None
        self.framerate_sum = datetime.timedelta(0)
        self.reset_fps_samples()

        self.processes = {}
        self.status = "ONLINE"
        self.cv2_image_array = {}
        self.raw_image = {}
        self.image_metadata = {}
        self.resolution_code = 5

    def get_resolution(self):
        if self.resolution_code == 1:
            return 160, 120
        elif self.resolution_code == 2:
            return 176, 144
        elif self.resolution_code == 3:
            return 320, 240
        elif self.resolution_code == 4:
            return 352, 288
        elif self.resolution_code == 5:
            return 640, 480
        elif self.resolution_code == 6:
            return 800, 600
        elif self.resolution_code == 7:
            return 1024, 768
        elif self.resolution_code == 8:
            return 1280, 1024
        elif self.resolution_code == 9:
            return 1600, 1200
        elif self.resolution_code == 10:
            return 1280, 960
        elif self.resolution_code == 11:
            return 2048, 1536
        elif self.resolution_code == 12:
            return 2592, 1944
        else:
            return None

    def set_resolution(self, resolution):
        if resolution == (160, 120):
            self.resolution_code = 1
        elif resolution == (176, 144):
            self.resolution_code = 2
        elif resolution == (320, 240):
            self.resolution_code = 3
        elif resolution == (352, 288):
            self.resolution_code = 4
        elif resolution == (640, 480):
            self.resolution_code = 5
        elif resolution == (800, 600):
            self.resolution_code = 6
        elif resolution == (1024, 768):
            self.resolution_code = 7
        elif resolution == (1280, 1024):
            self.resolution_code = 8
        elif resolution == (1600, 1200):
            self.resolution_code = 9
        elif resolution == (1280, 960):
            self.resolution_code = 10
        elif resolution == (2048, 1536):
            self.resolution_code = 11
        elif resolution == (2592, 1944):
            self.resolution_code = 12
        else:
            return
        self.controller.publish_message(self.imager_address, "resolution", self.resolution_code)

    def reset_fps_samples(self):
        """
        Make all of the old bad performance sampling disappear
        :return:
        :rtype:
        """
        self.local_image_counter = 0
        self.last_time = None
        self.framerate_sum = datetime.timedelta(0)

    def get_image(self, channel, asOpenCVArray):
        """
        Get image from a specific channel
        :param channel:
        :type channel:
        :param asOpenCVArray: True if it's an openCV array, false if its a raw jpeg with exif
        :type asOpenCVArray: bool
        :return:
        :rtype:
        """
        search = self.cv2_image_array if asOpenCVArray else self.raw_image
        if channel in search:
            return search[channel]
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

        try:
            # store image for channel
            self.raw_image[channel] = rawdata
            self.cv2_image_array[channel] = cv2.imdecode(np.asarray(rawdata, dtype="uint8"), cv2.IMREAD_COLOR)

            image = Image.open(io.BytesIO(rawdata))
            exif_data = image._getexif()
            if exif_data is None:
                print('missing exif data')
            else:
                exif = {
                    ExifTags.TAGS[k]: v
                    for k, v in exif_data.items()
                    if k in ExifTags.TAGS
                }

                # parse the baseline datetime
                intime = exif['DateTimeOriginal']
                timestamp = datetime.datetime.strptime(intime, '%Y-%m-%d %H:%M:%S')
                msec = int(exif['SubsecTimeOriginal'])
                timestamp = timestamp + datetime.timedelta(milliseconds=msec)

                image_number = int(exif['ImageNumber'])
                host = exif['HostComputer'].split('.')

                self.image_metadata[channel] = ImagerImageData(host[0], host[1], image_number, timestamp)

            self.dispatch_processes()
            if channel == 'raw':
                self.local_image_counter += 1
            alt_imager = self.controller.antipode(self)
            if channel in self.image_metadata and channel in alt_imager.image_metadata:
                tdelta = (self.image_metadata[channel].timestamp - alt_imager.image_metadata[
                    channel].timestamp).total_seconds()
                tdelta = abs(int(tdelta * 1000))
                self.controller.master_control.jitter_var.set(tdelta)
        except ValueError:
            print('i blew up')
            # self.raw_image[channel] = None
            # self.cv2_image_array[channel] = None

    def reboot(self):
        """
        Reboot the imager hardware
        Note this is the physical imager, not the Dantalion control system
        :return:
        :rtype:
        """
        self.send_control_message('reboot', '1')

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
        self.controller.mqtt_client.publish(topic, body)

    def dispatch_processes(self):
        for name, idef_process in list(self.processes.items()):
            is_mono = idef_process.get_container('RAWIMAGE') is not None
            if is_mono:
                if idef_process.get_container('RAWIMAGE').data_direction == EDataDirection.Output:
                    idef_process.get_container('RAWIMAGE').matrix = self.get_image('raw', True)
                    idef_process.get_container('RAWIMAGE').data_direction = EDataDirection.Input
            else:
                if self.controller.master_control.jitter_var.get() > 220:
                    # if not self.controller.master_control.trigger:
                    continue
                self.controller.master_control.trigger = False;
                if (idef_process.get_container('RAWIMAGELEFT').data_direction == EDataDirection.Output
                        and idef_process.get_container('RAWIMAGERIGHT').data_direction == EDataDirection.Output):
                    idef_process.get_container('RAWIMAGELEFT').matrix = self.get_image('raw', True)
                    idef_process.get_container('RAWIMAGERIGHT').matrix = \
                        self.controller.antipode(self).get_image('raw', True)
                    idef_process.get_container('RAWIMAGELEFT').data_direction = EDataDirection.Input
                    idef_process.get_container('RAWIMAGERIGHT').data_direction = EDataDirection.Input
            idef_process.process()
            if idef_process.completed:
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

    def stereo_filename(self):
        """
        Location of precomputed calibration file with camera intrinsics
        :return: Location of precomputed calibration file with camera intrinsics
        :rtype: basestring
        """
        filename = "stereo_{}_{}.json".format(self.controller.resource,
                                              self.imager_address)
        return "../resources/" + filename

    def get_calibration(self):
        filename = self.calibration_filename()
        with open(filename, "r") as file:
            return file.read()

    def set_calibration(self):
        filename = self.calibration_filename()
        with open(filename, "r") as file:
            cfg = file.read()
            self.controller.publish_message(self.imager_address, "intrinsics", cfg)
