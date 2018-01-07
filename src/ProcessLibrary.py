from Notification import Notifier
from ProcessAbstractionLayer import IDEFProcess, CameraIntrinsicsContainer, \
    EDataDirection
from StageLibrary import CalibrationStage, DistortionCalculatorStage, DifferentialStage


class ImageDifferentialCalculatorProcess(IDEFProcess):
    def __init__(self, name, targetImager):
        IDEFProcess.__init__(self, name)
        self.targetImager = targetImager

    def output_ready(self):
        return self.stages["DifferentialStage"].is_output_ready()

    def initialize_containers(self):
        sc = self.get_container('RAWIMAGE')
        sc.matrix = None
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

    def create_stages(self):
        x = [DifferentialStage(self, 'DifferentialStage')]
        for aStage in x:
            self.stages[aStage.name] = aStage

    def on_completion(self):
        ds = self.stages['DifferentialStage']
        output = ds.output_containers['DIFFERENTIALIMAGE'].matrix  # type: MatrixContainer
        self.targetImager.channel_images['diff'] = output

        ds.output_containers['MEANSQUAREDERROR'].data_direction = EDataDirection.Output
        ds.output_containers['DIFFERENTIALIMAGE'].data_direction = EDataDirection.Output
        ds.input_containers['RAWIMAGE'].data_direction = EDataDirection.Output
        self.completed = False


class IntrinsicsCalculatorProcess(IDEFProcess):

    def __init__(self, name, targetImager):
        IDEFProcess.__init__(self, name)
        self.targetImager = targetImager

    def output_ready(self):
        return self.stages["DistortionCalculatorStage"].is_output_ready()

    def initialize_containers(self):
        sc = self.get_container('BOARDSIZE')
        sc.value = (9, 7)
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('IMAGESIZE')
        sc.value = (640, 480)
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = 24
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 35
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('RAWIMAGE')
        sc.matrix = None
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('CAMERAINTRINSICS')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

    def create_stages(self):
        x = [CalibrationStage(self, 'CalibrationStage'), DistortionCalculatorStage(self, 'DistortionCalculatorStage')]
        for aStage in x:
            self.stages[aStage.name] = aStage

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['DistortionCalculatorStage']
        intrinsics = ds.output_containers['CAMERAINTRINSICS']  # type: CameraIntrinsicsContainer
        cfg = intrinsics.serialize_intrinsics(1.0)
        self.targetImager.controller.publish_message(self.targetImager.imager_address, "intrinsics", cfg)
        filename = self.targetImager.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()
