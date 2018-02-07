"""
This module defines the classes and other elements that define the basic implementation of the IDEF0 based process
definition system.  The goal is to allow for the easy construction of image processing tasks from standardized
building blocks.

"""
import json
import threading
import uuid
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from threading import Lock

import cv2
import os

import datetime


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
    Jitter = 0
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
        :param controller:
        :type controller:
        :param side: -1 for left, 0 for mono, and 1 for right
        :type side:
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
        if len(IDEFProcess.ActiveProcesses) > 0:
            IDEFProcess.DataReady.set()

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
            for name, idef_process in list(IDEFProcess.ActiveProcesses.items()):
                is_mono = idef_process.has_container('RAWIMAGE')
                if is_mono:
                    if idef_process.get_container('RAWIMAGE').data_direction == EDataDirection.Output:
                        qn = idef_process.controller_name + '_LEFT' if idef_process.targetImager.index == 0 else idef_process.controller_name + "_RIGHT"
                        ql = len(IDEFProcess.ImageQueues[qn])
                        if ql >= 30:
                            print('resync q {}={}'.format(qn, len(IDEFProcess.ImageQueues[qn])))
                            IDEFProcess.ImageQueues[qn].clear()
                            ql = len(IDEFProcess.ImageQueues[qn])  # this better be zero
                        if ql == 0:
                            continue;
                        # single image data always consumed
                        image, imagametadata = IDEFProcess.ImageQueues[qn].pop()
                        idef_process.get_container('RAWIMAGE').matrix = image
                        idef_process.get_container('RAWIMAGE').data_direction = EDataDirection.Input
                        idef_process.get_container('IMAGER').value = imagametadata.imager
                        idef_process.get_container('IMAGER').data_direction = EDataDirection.Input
                elif (idef_process.get_container('RAWIMAGELEFT').data_direction == EDataDirection.Output
                      and idef_process.get_container('RAWIMAGERIGHT').data_direction == EDataDirection.Output):

                    lqn = idef_process.controller_name + '_LEFT'
                    rqn = idef_process.controller_name + '_RIGHT'

                    "if either queue is empty NOTHING HAPPENS"

                    if len(IDEFProcess.ImageQueues[lqn]) == 0 or len(IDEFProcess.ImageQueues[rqn]) == 0:
                        continue;
                    leftimage, leftimagametadata = IDEFProcess.ImageQueues[lqn][0]  # peek at leftmost item
                    rightimage, rightimagemetadata = IDEFProcess.ImageQueues[rqn][0]  # peek at leftmost item

                    "the oldest image WILL ALWAYS be consumed"
                    leftyoungest = leftimagametadata.timestamp >= rightimagemetadata.timestamp
                    if leftyoungest:
                        rightimage, rightimagemetadata = IDEFProcess.ImageQueues[rqn].pop()
                    else:
                        leftimage, leftimagametadata = IDEFProcess.ImageQueues[lqn].pop()

                    tdelta = (leftimagametadata.timestamp - rightimagemetadata.timestamp).total_seconds()
                    IDEFProcess.Jitter = abs(int(tdelta * 1000))
                    if IDEFProcess.Jitter > 1000:
                        print('resync L={},R={}'.format(len(IDEFProcess.ImageQueues[lqn]),
                                                        len(IDEFProcess.ImageQueues[rqn])))
                        IDEFProcess.ImageQueues[lqn].clear()
                        IDEFProcess.ImageQueues[rqn].clear()
                        continue

                    if IDEFProcess.Jitter > 800:
                        # if not self.controller.master_control.trigger:
                        continue
                    "the youngest image WILL ONLY be consumed if this is under the jitter limit"
                    if leftyoungest:
                        leftimage, leftimagametadata = IDEFProcess.ImageQueues[lqn].pop()
                    else:
                        rightimage, rightimagemetadata = IDEFProcess.ImageQueues[rqn].pop()

                    if True:
                        idef_process.get_container('RAWIMAGELEFT').matrix = leftimage
                        idef_process.get_container('RAWIMAGERIGHT').matrix = rightimage
                        idef_process.get_container('RAWIMAGELEFT').data_direction = EDataDirection.Input
                        idef_process.get_container('RAWIMAGERIGHT').data_direction = EDataDirection.Input
                        idef_process.get_container('LEFTIMAGER').value = leftimagametadata.imager
                        idef_process.get_container('RIGHTIMAGER').value = rightimagemetadata.imager
                        idef_process.get_container('LEFTIMAGER').data_direction = EDataDirection.Input
                        idef_process.get_container('RIGHTIMAGER').data_direction = EDataDirection.Input

                idef_process.process()
                if idef_process.completed:
                    print('deleting process {}'.format(name))
                    del IDEFProcess.ActiveProcesses[name]
                break
            IDEFProcess.DataReady.clear()
            IDEFProcess.DataReady.wait()
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
        self.completed = False
        self.status_message = None
        self.targetImager = None
        self.processLock = Lock()

    def initialize(self,kvp):
        self.create_stages()
        self.allocate_containers()
        self.bind_containers()
        self.initialize_containers()
        for n,v in kvp.items():
            c = self.get_container(n)
            if c.get_container_type() == EContainerType.SCALARCONTAINER:
                c.value = v;
            elif c.get_container_type() == EContainerType.MATRIXCONTAINER:
                c.matrix = v;


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

    def run_stages(self):
        """
        Search through the stages for any reporting that they are ready to run and run their process
        On completion, this will release the process lock
        :return:
        :rtype:
        """
        try:
            if not self.completed:
                for key, aStage in self.stages.items():
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

        :param host_process: The process that contains this stage instance
        :type host_process: IDEFProcess
        :param name: The container_name of this stage within the host process scope
        :type name: basestring
        """
        self.host_process = host_process
        self.container_name = name
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
