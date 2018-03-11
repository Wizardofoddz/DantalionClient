"""
This module defines the classes and other elements that define the basic implementation of the IDEF0 based process
definition system.  The goal is to allow for the easy construction of image processing tasks from standardized
building blocks.

"""
import os
import threading
import uuid
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from threading import Lock

import cv2
import numpy as np


class EScope(Enum):
    Undefined = 0
    Local = 1
    Global = 2


class EExecutionFlow(Enum):
    Undefined = 0
    Syncronous = 1
    Asyncronous = 2


class IDEFProcess(ABC):
    """
    Defines the generic process that operates on data containers via a collection of stages
    """
    ActiveProcesses = {}
    ImageQueues = {}
    Run = True

    DataReady = threading.Event()

    @staticmethod
    def initialize_image_intake(stereo_controllers):
        for acontroller in stereo_controllers:
            IDEFProcess.ImageQueues[acontroller + '_LEFT'] = deque()
            IDEFProcess.ImageQueues[acontroller + '_RIGHT'] = deque()

    @staticmethod
    def post_image(controller_name, isleft, channel, imageinfo):
        """
        Add an image to the end of it's appropriate intake queue and pulse data input signal
        :param channel:
        :type channel:
        :param isleft:
        :type isleft:
        :param controller_name:
        :type controller_name:
        :param imageinfo:
        :type imageinfo: tuple of (opencv image matrix,image metadata)
        :return:
        :rtype:
        """
        if channel != 'raw':
            return
        name = controller_name
        if isleft:
            name += "_LEFT"
        else:
            name += "_RIGHT"

        cloned = (imageinfo[0].copy(), imageinfo[1])
        IDEFProcess.ImageQueues[name].append(cloned)

    @staticmethod
    def bind_imager_input_functions(imager):
        """
        The
        :param imager:
        :type imager:
        :return:
        :rtype:
        """
        imager.submit_image_to_processes = lambda ctlr_name, isleft, channel, imageinfo: IDEFProcess.post_image(
            ctlr_name,
            isleft,
            channel,
            imageinfo)

    @staticmethod
    def add_process(idef_process):
        IDEFProcess.ActiveProcesses[idef_process.name] = idef_process
        IDEFProcess.DataReady.set()

    @staticmethod
    def process_dispatcher():
        while IDEFProcess.Run:
            if len(IDEFProcess.ActiveProcesses) == 0:
                IDEFProcess.DataReady.wait()
            for name, idef_process in list(IDEFProcess.ActiveProcesses.items()):
                # print('Dispatching process '.format(name))
                idef_process.process()
                if idef_process.completed:
                    print('deleting process {}'.format(name))
                    del IDEFProcess.ActiveProcesses[name]
                    break

    @staticmethod
    def serialize_matrix_to_json(m):
        cm = []
        cm.append(m.shape[0])
        cm.append(m.shape[1])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                cm.append(m.item((i, j)))
        return cm

    def __init__(self, name):
        self.name = name

        self.controller_name = 'cancer'
        self.global_variables = {}
        self.children = {}
        self.locals = {}
        self.parent = None
        self.stages = {}
        self.tape = []
        self.completed = False
        self.status_message = None
        self.targetImager = None
        self.processLock = Lock()
        self.live_image_transfer = True

    def initialize(self, kvp):
        self.create_stages()
        self.allocate_containers()
        self.bind_containers()
        self.initialize_containers()
        for n, v in kvp.items():
            c = self.get_container(n)
            if c.get_container_type() == EContainerType.SCALARCONTAINER:
                c.value = v
            elif c.get_container_type() == EContainerType.MATRIXCONTAINER:
                c.matrix = v

    def add_stages(self, stages):
        for i, s in enumerate(stages):
            self.stages[s.container_name] = s
            self.tape.append(s)
            s.process_index = i

    def send_status_message(self, text):
        """
        Use status callback to send a status message
        :param text: status text
        :type text: string
        :return:
        :rtype:
        """
        if self.status_message is not None:
            self.status_message(text)

    @abstractmethod
    def output_ready(self):
        """
        Test for a process output
        :return: True if there is data available
        :rtype:
        """
        pass

    @abstractmethod
    def initialize_containers(self):
        """
        After containers are created and bound, this routine can fill the ones owned by the process
        with data
        """
        pass

    @abstractmethod
    def create_stages(self):
        """
        Create all of the named stages in this process and place them in the stages dictionary

        This routine will create new instances of every stage in the process, assign them specific
        names, and add them to the stage dictionary.
        """
        pass

    @abstractmethod
    def get_required_containers(self):
        pass

    def on_completion(self):
        """
        Called when the internal stage logic has marked the process complete

        This may automatically restart the process, as with the differential view
        :return:
        :rtype:
        """
        pass

    def get_container(self, name):
        """
        Search global and local container names for the given container_name and return the associated container
        :param name: Name of desired container
        :type name: basestring
        :return: Returns the located container or None
        :rtype: DataContainer
        """
        result = self.global_variables.get(name)
        if result is None:
            result = self.locals.get(name)
        return result

    def has_container(self, name):
        if name in self.global_variables:
            return True
        return name in self.locals

    def get_stage(self, name):
        return self.stages[name]

    def run_stages(self):
        """
        Search through the stages for any reporting that they are ready to run and run their process
        On completion, this will release the process lock
        :return:
        :rtype:
        """
        try:
            if not self.completed:
                for aStage in self.tape:
                    # print('testing stage {}'.format(aStage.container_name))
                    if aStage.is_ready_to_run():
                        # print('dispatching stage {}'.format(aStage.container_name))
                        aStage.process()
                        if self.completed:
                            self.on_completion()
                            break
        finally:
            self.processLock.release()

    def process(self):
        if not self.processLock.acquire(False):
            return
        else:
            t = threading.Thread(target=self.run_stages)
            t.daemon = True
            t.start()
        # self.run_stages()

    def allocate_containers(self):
        for key, aStage in self.stages.items():
            bindings = aStage.get_required_containers()
            bindings.extend(self.get_required_containers())
            for aBinding in bindings:
                if aBinding.name in self.global_variables:
                    if self.global_variables[aBinding.name].get_container_type() != aBinding.container_type:
                        raise ValueError('Global container type mismatch.')
                else:
                    if aBinding.name in self.locals:
                        if self.locals[aBinding.name].get_container_type() != aBinding.container_type:
                            raise ValueError('Local container type mismatch.')
                    else:
                        self.locals[aBinding.name] = DataContainer.container_factory(aBinding.container_type,
                                                                                     aBinding.name)

    def bind_containers(self):
        for key, aStage in self.stages.items():
            bindings = aStage.get_required_containers()
            bindings.extend(self.get_required_containers())
            for aBinding in bindings:
                dc = self.locals[aBinding.name]
                if dc is None:
                    dc = self.global_variables[aBinding.name]
                if dc is None:
                    raise ValueError('Cannot bind variable.')
                aStage.bind_container(aBinding.connection, dc)
            aStage.initialize_container_impedance()

    def record_stereo_images(self, controller_name, iid, left, right):
        if left is not None:
            self.record_stereo_image(controller_name, left, iid, 'L')
        if right is not None:
            self.record_stereo_image(controller_name, right, iid, 'R')
        return iid

    @staticmethod
    def stereo_session_image_folder():
        return "../CalibrationRecords/StereoImages"

    @staticmethod
    def extract_training_dictionary():
        td = {}
        files = [i for i in os.listdir("../CalibrationRecords/StereoImages") if i.endswith("jpg")]
        for f in files:
            fnc = f.split('_')
            key = fnc[0]
            ctlr = fnc[1]
            cam = fnc[2][0]
            camindex = 0 if cam == 'L' else 1
            if key in td:
                ia = td[key]
                if ia[camindex] is True:
                    print('{} has a duplicate'.format(key))
                else:
                    ia[camindex] = True
            else:
                ia = [False, False]
                ia[camindex] = True
                td[key] = ia
        return td

    @staticmethod
    def extract_training_stereo_pairs(td):
        result = []
        for k, v in td.items():
            if v[0] and v[1]:
                result.append(k)
        return result

    @staticmethod
    def extract_training_singles(td, isLeft):
        result = []
        tidx = 0 if isLeft else 1
        fidx = 1 if isLeft else 0
        for k, v in td.items():
            if v[tidx] and not v[fidx]:
                result.append(k)
        return result

    @staticmethod
    def stereo_sessionfolder():
        return "../CalibrationRecords/StereoSessions"

    @staticmethod
    def stereo_session_filename(fileid, controller_name, stereoSide):
        return IDEFProcess.stereo_session_image_folder() + "/{}_{}_{}.jpg".format(fileid,
                                                                                  controller_name,
                                                                                  stereoSide)

    @staticmethod
    def record_stereo_image(controller_name, image, fileid, stereoSide):
        filename = IDEFProcess.stereo_session_filename(fileid, controller_name, stereoSide)
        # save the image to a file
        cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    @staticmethod
    def fetch_stereo_image(fileid, controller_name, stereoSide):
        filename = IDEFProcess.stereo_session_filename(fileid, controller_name, stereoSide)
        # read the image as a raw byte array
        with open(filename, 'rb') as f:
            jpegdata = f.read()
        return bytearray(jpegdata)

    @staticmethod
    def fetch_stereo_image_matrix(fileid, controller_name, stereoSide):
        image = IDEFProcess.fetch_stereo_image(fileid, controller_name, stereoSide)
        ra = np.asarray(image, dtype="uint8")
        return cv2.imdecode(ra, cv2.IMREAD_COLOR)


class EConnection(Enum):
    Undefined = 0
    Input = 1
    Output = 2
    Control = 3
    Environment = 4


class IDEFStageBinding:
    def __init__(self, connection, container_type, data_direction, name):
        self.connection = connection
        self.container_type = container_type
        self.data_direction = data_direction
        self.name = name


class IDEFStage(ABC):
    def __init__(self, host_process, name):
        """
        nb:  The get_required_containers should not be perceived to imply allocation - it is called whenever the
        stage specific container list is required
        :param host_process: The process that contains this stage instance
        :type host_process: IDEFProcess
        :param name: The container_name of this stage within the host process scope
        :type name: basestring
        """
        self.host_process = host_process
        self.container_name = name
        self.process_index = None
        self.input_containers = {}
        self.output_containers = {}
        self.control_containers = {}
        self.environment_containers = {}

    @abstractmethod
    def get_required_containers(self):
        pass

    @abstractmethod
    def initialize_container_impedance(self):
        pass

    @abstractmethod
    def is_ready_to_run(self):
        pass

    @abstractmethod
    def is_output_ready(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def bind_container(self, connection, container):
        cs = self.get_connection_set(connection)
        cs[container.name] = container

    def get_connection_set(self, connection):
        """
        Abstracts the difference between connection types and lets consuming logic simply operate on the list
        result
        :param connection: Connection surface idenfitier
        :type connection: EConnection
        :return: the data containers bound to this connection surface
        :rtype: [DataContainer]
        """
        if connection == EConnection.Input:
            return self.input_containers
        if connection == EConnection.Output:
            return self.output_containers
        if connection == EConnection.Control:
            return self.control_containers
        if connection == EConnection.Environment:
            return self.environment_containers
        return None

    def is_container_valid(self, connection, name, datadir):
        """
        Test a container for data operations - this ensures that container directionality is maintained
        :param connection: identifies connection surface
        :type connection: EConnection
        :param name: Name of the container on the connection surface
        :type name: basestring
        :param datadir: Required data direction for the container
        :type datadir: EDataDirection
        :return: true if the named container on the given surface can be used in this data direction
        :rtype: bool
        """
        connections = self.get_connection_set(connection)
        binding = connections[name]
        if binding is not None:
            return binding.data_direction == datadir or binding.data_direction == EDataDirection.InputOutput
        return False


class EContainerType(Enum):
    """
    Type definition for specific container type
    """
    UNKNOWN = 1
    MATRIXCONTAINER = 2
    CAMERA_INTRINSICS = 5
    SCALARCONTAINER = 6


class EDataDirection(Enum):
    """
    Defines direction of data flow where outputs are writes and inputs are reads
    """
    Unknown = 0
    Input = 1
    Output = 2
    InputOutput = 3


class DataContainer(ABC):
    """
    Abstract based class for all data containers
    """

    def __init__(self, name):
        """
        Set container_name to given value and initialize data direction to unknown
        :param name:
        :type name:
        """
        self.name = name
        self.data_direction = EDataDirection.Unknown
        self.identifier = uuid.uuid4()

    @abstractmethod
    def get_container_type(self):
        return EContainerType.UNKNOWN

    @abstractmethod
    def set_output_for_subsequent_input(self, data):
        self.data_direction = EDataDirection.Input

    def has_valid_input(self):
        return self.data_direction == EDataDirection.Input

    @staticmethod
    def container_factory(container_type, name):
        result = None
        if container_type == EContainerType.MATRIXCONTAINER:
            result = MatrixContainer(name)
        elif container_type == EContainerType.SCALARCONTAINER:
            result = ScalarContainer(name)
        return result


class EScalarType(Enum):
    Unknown = 0
    Boolean = 1
    Integer = 2
    LongInteger = 3
    Double = 4
    Text = 5
    CvSize = 6


class MatrixContainer(DataContainer):
    def __init__(self, name):
        super().__init__(name)
        self.matrix = None

    def get_container_type(self):
        return EContainerType.MATRIXCONTAINER

    def set_output_for_subsequent_input(self, data):
        self.matrix = data
        super(MatrixContainer, self).set_output_for_subsequent_input(data)


class ScalarContainer(DataContainer):
    def __init__(self, name):
        super().__init__(name)
        self.scalar_type = EScalarType.Unknown
        self.value = None

    def get_container_type(self):
        return EContainerType.SCALARCONTAINER

    def set_output_for_subsequent_input(self, data):
        self.value = data
        super(ScalarContainer, self).set_output_for_subsequent_input(data)

    def set_boolean(self, v):
        self.scalar_type = EScalarType.Boolean
        self.value = v

    def get_boolean(self):
        if self.scalar_type != EScalarType.Boolean:
            raise ValueError('Not a boolean scalar container.')
        return self.value
