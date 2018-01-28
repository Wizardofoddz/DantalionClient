import json

import numpy

from Notification import Notifier
from ProcessAbstractionLayer import IDEFProcess, CameraIntrinsicsContainer, \
    EDataDirection
from StageLibrary import CalibrationStage, DistortionCalculatorStage, DifferentialStage, \
    StereoCalibratorCalculatorStage, StereoCalibrationStage


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
        sc.value = self.targetImager.get_resolution()
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = 48
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 5
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('RAWIMAGE')
        sc.matrix = None
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('CAMERAINTRINSICS')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('REPROJECTIONERROR')
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
        reperr = ds.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
        self.status_message("Reprojection error is {}".format(reperr.value))
        cfg = intrinsics.serialize_intrinsics(1.0)
        self.targetImager.controller.publish_message(self.targetImager.imager_address, "intrinsics", cfg)
        filename = self.targetImager.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()


class StereoCalculatorProcess(IDEFProcess):

    def __init__(self, name, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight

    def output_ready(self):
        return self.stages["StereoCalibrationCalculatorStage"].is_output_ready()

    def buildmat(self, dv):
        X = numpy.empty(shape=[dv[0], dv[1]])
        dvi = 2
        for i in range(dv[0]):
            for j in range(dv[1]):
                X[i, j] = dv[dvi]
                dvi += 1
        return X

    def initialize_containers(self):

        sc = self.get_container('RAWIMAGELEFT')
        sc.matrix = None
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('RAWIMAGERIGHT')
        sc.matrix = None
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        left_calib = json.loads(self.targetImagerLeft.get_calibration())
        right_calib = json.loads(self.targetImagerRight.get_calibration())
        sc = self.get_container('CAMERAMATRIXLEFT')
        sc.matrix = self.buildmat(left_calib["camera_matrix"])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('CAMERAMATRIXRIGHT')
        sc.matrix = self.buildmat(right_calib["camera_matrix"])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('DISTORTIONCOEFFSLEFT')
        sc.matrix = self.buildmat(left_calib['distortion_coefficients'])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('DISTORTIONCOEFFSRIGHT')
        sc.matrix = self.buildmat(right_calib['distortion_coefficients'])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('BOARDSIZE')
        sc.value = (9, 7)
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('IMAGESIZE')
        # todo This is assuming both imagers have equal resolution
        sc.value = self.targetImagerLeft.get_resolution()
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = 40
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('STEREOROTATION')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('STEREOTRANSLATION')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('ESSENTIAL')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('FUNDAMENTAL')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

    def create_stages(self):
        x = [StereoCalibrationStage(self, 'StereoCalibrationStage'),
             StereoCalibratorCalculatorStage(self, 'StereoCalibratorCalculatorStage')]
        for aStage in x:
            self.stages[aStage.name] = aStage

    def flatten_matrix(self, m):
        cm = []
        cm.append(m.shape[0])
        cm.append(m.shape[1])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                cm.append(m.item((i, j)))
        return cm

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['StereoCalibratorCalculatorStage']
        dd = {}
        dd['Q'] = str(ds.output_containers['RMS_STEREO'].value)
        dd['R'] = self.flatten_matrix(ds.output_containers['STEREOROTATION'].matrix)  # type: MatrixContainer
        dd['T'] = self.flatten_matrix(ds.output_containers['STEREOTRANSLATION'].matrix)  # type: MatrixContainer
        dd['E'] = self.flatten_matrix(ds.output_containers['ESSENTIAL'].matrix)  # type: MatrixContainer
        dd['F'] = self.flatten_matrix(ds.output_containers['FUNDAMENTAL'].matrix)  # type: MatrixContainer
        dd['CML'] = self.flatten_matrix(ds.input_containers['CAMERAMATRIXLEFT'].matrix)
        dd['CMR'] = self.flatten_matrix(ds.input_containers['CAMERAMATRIXRIGHT'].matrix)
        dd['DCL'] = self.flatten_matrix(ds.input_containers['DISTORTIONCOEFFSLEFT'].matrix)
        dd['DCR'] = self.flatten_matrix(ds.input_containers['DISTORTIONCOEFFSRIGHT'].matrix)
        self.status_message("RMS Stereo error is {}".format(dd['Q']))

        serialized = json.dumps(dd, ensure_ascii=False)
        self.targetImagerLeft.controller.publish_message(self.targetImagerLeft.imager_address, "stereoextrinsics",
                                                         serialized)
        filename = self.targetImagerLeft.stereo_filename()
        file = open(filename, "w")
        file.write(serialized)
        file.close()
