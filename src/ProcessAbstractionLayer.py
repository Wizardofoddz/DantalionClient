"""
This module defines the classes and other elements that define the basic implementation of the IDEF0 based process
definition system.  The goal is to allow for the easy construction of image processing tasks from standardized
building blocks.

"""
import json
import threading
from abc import ABC, abstractmethod
from enum import Enum
from threading import Lock


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

    def __init__(self, name):
        self.name = name
        self.global_variables = {}
        self.children = {}
        self.locals = {}
        self.parent = None
        self.stages = {}
        self.completed = False
        self.status_message = None
        self.targetImager = None
        self.processLock = Lock()

    def initialize(self):
        self.create_stages()
        self.allocate_containers()
        self.bind_containers()
        self.initialize_containers()

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
        pass

    @abstractmethod
    def initialize_containers(self):
        pass

    @abstractmethod
    def create_stages(self):
        pass

    def on_completion(self):
        pass

    def get_container(self, name):
        result = self.global_variables.get(name)
        if result is None:
            result = self.locals.get(name)
        return result

    def run_stages(self):
        try:
            for key, aStage in self.stages.items():
                if aStage.is_ready_to_run():
                    # print('dispatching stage {}'.format(aStage.name))
                    aStage.process()
        finally:
            self.processLock.release()

    def process(self):
        if not self.processLock.acquire(False):
            return

        t = threading.Thread(target=self.run_stages)
        t.start()
        # self.run_stages()
        if self.completed:
            self.on_completion()

    def allocate_containers(self):
        for key, aStage in self.stages.items():
            bindings = aStage.get_required_containers()
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
        :param name: The name of this stage within the host process scope
        :type name: basestring
        """
        self.host_process = host_process
        self.name = name
        self.input_containers = {}
        self.output_containers = {}
        self.control_containers = {}
        self.environment_containers = {}

    @abstractmethod
    def get_required_containers(self):
        print('must override')

    @abstractmethod
    def initialize_container_impedance(self):
        print('must override')

    @abstractmethod
    def is_ready_to_run(self):
        print('must override')

    @abstractmethod
    def is_output_ready(self):
        print('must override')

    @abstractmethod
    def process(self):
        print('must override')

    def bind_container(self, connection, container):
        cs = self.get_connection_set(connection)
        cs[container.name] = container

    def get_connection_set(self, connection):
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
        connections = self.get_connection_set(connection)
        binding = connections[name]
        if binding is not None:
            return binding.data_direction == datadir or binding.data_direction == EDataDirection.InputOutput
        return False


class EContainerType(Enum):
    UNKNOWN = 1
    MATRIXCONTAINER = 2
    CAMERA_INTRINSICS = 5
    SCALARCONTAINER = 6


class EDataDirection(Enum):
    Unknown = 0
    Input = 1
    Output = 2
    InputOutput = 3


class DataContainer(ABC):
    def __init__(self, name):
        self.name = name
        self.data_direction = EDataDirection.Unknown

    @abstractmethod
    def get_container_type(self):
        return EContainerType.UNKNOWN

    def has_valid_input(self):
        return self.data_direction == EDataDirection.Input

    @staticmethod
    def container_factory(container_type, name):
        result = None
        if container_type == EContainerType.MATRIXCONTAINER:
            result = MatrixContainer(name)
        elif container_type == EContainerType.CAMERA_INTRINSICS:
            result = CameraIntrinsicsContainer(name)
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


class CameraIntrinsicsContainer(DataContainer):
    def __init__(self, name):
        super().__init__(name)
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.rotation_vectors = None
        self.translation_vectors = None

    def get_container_type(self):
        return EContainerType.CAMERA_INTRINSICS

    def serialize_intrinsics(self, rectify_alpha):
        """
        Serialize the data that represents the camera intrinsics
        :param rectify_alpha: 0 for only good pixels, 1 for all pixels where outer black pixels are invalid
        :type rectify_alpha: float
        :return: serialized JSON representation
        :rtype: string
        """
        dd = {}
        dd['rectify_alpha'] = rectify_alpha
        dd['camera_matrix'] = self.flatten_matrix(self.camera_matrix)
        dd['distortion_coefficients'] = self.flatten_matrix(self.distortion_coefficients)

        serialized = json.dumps(dd, ensure_ascii=False)
        return serialized

    def flatten_matrix(self, m):
        cm = []
        cm.append(m.shape[0])
        cm.append(m.shape[1])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                cm.append(m.item((i, j)))
        return cm

    def print_matrix(self, m):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                print(m.item(i, j), end='')
                print(',', end='')
            print()


class MatrixContainer(DataContainer):
    def __init__(self, name):
        super().__init__(name)
        self.matrix = None

    def get_container_type(self):
        return EContainerType.MATRIXCONTAINER


class ScalarContainer(DataContainer):
    def __init__(self, name):
        super().__init__(name)
        self.scalar_type = EScalarType.Unknown
        self.value = None

    def get_container_type(self):
        return EContainerType.SCALARCONTAINER

    def set_boolean(self, v):
        self.scalar_type = EScalarType.Boolean
        self.value = v

    def get_boolean(self):
        if self.scalar_type != EScalarType.Boolean:
            raise ValueError('Not a boolean scalar container.')
        return self.value
