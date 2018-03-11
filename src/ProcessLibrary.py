import datetime
import json
import os
import uuid

import numpy

from Notification import Notifier
from ProcessAbstractionLayer import IDEFProcess, \
    EDataDirection, IDEFStageBinding, EConnection, EContainerType
from StageLibrary import CalibrationStage, DistortionCalculatorStage, DifferentialStage, \
    StereoCalibratorCalculatorStage, StereoCalibrationStage, ChessboardCaptureStage, \
    StoredImageInterfaceStage, RandomSolutionSearchStage, GeneticSolutionSearchStage, StereoSolutionSearchStage, \
    ImageInjector
from TrainingSupport import StereoTrainingSet


class ImageDifferentialCalculatorProcess(IDEFProcess):
    def __init__(self, name, targetImager):
        IDEFProcess.__init__(self, name)
        self.targetImager = targetImager

    def get_required_containers(self):
        return []

    def output_ready(self):
        return self.stages["DifferentialStage"].is_output_ready()

    def initialize_containers(self):
        sc = self.get_container('RAWIMAGE')
        sc.matrix = None
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

    def create_stages(self):
        x = [DifferentialStage(self, 'DifferentialStage')]
        self.add_stages(x)

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

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "IMAGER")]
        return container_list

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
        sc.value = 4
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 2
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHMOVE')
        sc.value = 125
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

        # clear out the composite chessboard stream
        self.targetImager.set_image('compositechess', None, None)

    def create_stages(self):
        x = [CalibrationStage(self, 'CalibrationStage'), DistortionCalculatorStage(self, 'DistortionCalculatorStage')]
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['DistortionCalculatorStage']
        intrinsics = ds.output_containers['CAMERAINTRINSICS'].value  # type: ScalarContainer
        reperr = ds.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
        self.status_message("Reprojection error is {}".format(reperr.value))
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, numpy.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = uuid.uuid4().hex
        imager = self.locals['IMAGER'].value
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = imager.imager_address
        translated['MATCHCOUNT'] = self.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        self.targetImager.controller.publish_message(self.targetImager.imager_address, "intrinsics", cfg)
        filename = self.targetImager.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()

        filename = "../CalibrationRecords/IntrinsicsHistory.txt"
        if os.path.isfile(filename):
            file = open(filename, "a")
            file.write(",\n")
        else:
            file = open(filename, "w")

        file.write(cfg)
        file.close()


class StereoCalculatorProcess(IDEFProcess):

    def __init__(self, name, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RECORDING")]
        return container_list

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
        sc.matrix = self.buildmat(left_calib["CAMERAMATRIX"])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('CAMERAMATRIXRIGHT')
        sc.matrix = self.buildmat(right_calib["CAMERAMATRIX"])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('DISTORTIONCOEFFSLEFT')
        sc.matrix = self.buildmat(left_calib['DISTORTIONCOEFFICIENTS'])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('DISTORTIONCOEFFSRIGHT')
        sc.matrix = self.buildmat(right_calib['DISTORTIONCOEFFICIENTS'])
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
        sc.value = 20
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
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['StereoCalibratorCalculatorStage']
        dd = {}
        dd['Q'] = str(ds.output_containers['REPROJECTIONERROR'].value)
        dd['R'] = StereoTrainingSet.flatten_matrix(
            ds.output_containers['STEREOROTATION'].matrix)  # type: MatrixContainer
        dd['T'] = StereoTrainingSet.flatten_matrix(
            ds.output_containers['STEREOTRANSLATION'].matrix)  # type: MatrixContainer
        dd['E'] = StereoTrainingSet.flatten_matrix(ds.output_containers['ESSENTIAL'].matrix)  # type: MatrixContainer
        dd['F'] = StereoTrainingSet.flatten_matrix(ds.output_containers['FUNDAMENTAL'].matrix)  # type: MatrixContainer
        dd['CML'] = StereoTrainingSet.flatten_matrix(ds.input_containers['CAMERAMATRIXLEFT'].matrix)
        dd['CMR'] = StereoTrainingSet.flatten_matrix(ds.input_containers['CAMERAMATRIXRIGHT'].matrix)
        dd['DCL'] = StereoTrainingSet.flatten_matrix(ds.input_containers['DISTORTIONCOEFFSLEFT'].matrix)
        dd['DCR'] = StereoTrainingSet.flatten_matrix(ds.input_containers['DISTORTIONCOEFFSRIGHT'].matrix)
        self.status_message("RMS Stereo error is {}".format(dd['Q']))

        serialized = json.dumps(dd, ensure_ascii=False)
        self.targetImagerLeft.controller.publish_message(self.targetImagerLeft.imager_address, "stereoextrinsics",
                                                         serialized)
        filename = self.targetImagerLeft.stereo_filename()
        file = open(filename, "w")
        file.write(serialized)
        file.close()

        # save the recorded session data
        if self.get_container('RECORDING').value:
            dd['CALIBRATIONIMAGEHISTORY'] = ds.environment_containers['CALIBRATIONIMAGEHISTORY'].value
            dd['CONTROLLER'] = 'unknownctlr'
            dd['timestamp'] = datetime.datetime.now().isoformat()
            path = StereoCalculatorProcess.stereo_sessionfolder()
            num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))

            filename = "../CalibrationRecords/StereoSessions/session.{}.json".format(num_files + 1)
            with open(filename, 'w') as f:
                json.dump(dd, f, ensure_ascii=True)


class ChessboardCaptureProcess(IDEFProcess):

    def __init__(self, name, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight

    def get_required_containers(self):
        container_list = []
        return container_list

    def output_ready(self):
        return self.stages["ChessboardCaptureStage"].is_output_ready()

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

        sc = self.get_container('LEFTIMAGER')
        sc.value = self.targetImagerLeft
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('RIGHTIMAGER')
        sc.value = self.targetImagerRight
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc
        sc = self.get_container('BOARDSIZE')
        sc.value = (9, 7)
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

    def create_stages(self):
        x = [ImageInjector(self, 'ImageInjector', 'raw'),
             ChessboardCaptureStage(self, 'ChessboardCaptureStage')]
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['ChessboardCaptureStage']


class CalibratedStereoCalculatorProcess(IDEFProcess):

    def __init__(self, name, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RECORDING")]
        return container_list

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
        sc.matrix = self.buildmat(left_calib["CAMERAMATRIX"])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('CAMERAMATRIXRIGHT')
        sc.matrix = self.buildmat(right_calib["CAMERAMATRIX"])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('DISTORTIONCOEFFSLEFT')
        sc.matrix = self.buildmat(left_calib['DISTORTIONCOEFFICIENTS'])
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc
        sc = self.get_container('DISTORTIONCOEFFSRIGHT')
        sc.matrix = self.buildmat(right_calib['DISTORTIONCOEFFICIENTS'])
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

        sc = self.get_container('NUMPAIRS')
        sc.value = 10
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = 999
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
        x = [RandomSolutionSearchStage(self, 'TestImageIngestorStage'),
             StereoCalibrationStage(self, 'StereoCalibrationStage'),
             StereoCalibratorCalculatorStage(self, 'StereoCalibratorCalculatorStage')]
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['StereoCalibratorCalculatorStage']
        dd = {}
        dd['Q'] = str(ds.output_containers['REPROJECTIONERROR'].value)
        dd['R'] = StereoTrainingSet.flatten_matrix(
            ds.output_containers['STEREOROTATION'].matrix)  # type: MatrixContainer
        dd['T'] = StereoTrainingSet.flatten_matrix(
            ds.output_containers['STEREOTRANSLATION'].matrix)  # type: MatrixContainer
        dd['E'] = StereoTrainingSet.flatten_matrix(ds.output_containers['ESSENTIAL'].matrix)  # type: MatrixContainer
        dd['F'] = StereoTrainingSet.flatten_matrix(ds.output_containers['FUNDAMENTAL'].matrix)  # type: MatrixContainer
        dd['CML'] = StereoTrainingSet.flatten_matrix(ds.input_containers['CAMERAMATRIXLEFT'].matrix)
        dd['CMR'] = StereoTrainingSet.flatten_matrix(ds.input_containers['CAMERAMATRIXRIGHT'].matrix)
        dd['DCL'] = StereoTrainingSet.flatten_matrix(ds.input_containers['DISTORTIONCOEFFSLEFT'].matrix)
        dd['DCR'] = StereoTrainingSet.flatten_matrix(ds.input_containers['DISTORTIONCOEFFSRIGHT'].matrix)
        self.status_message("RMS Stereo error is {}".format(dd['Q']))

        serialized = json.dumps(dd, ensure_ascii=False)
        self.targetImagerLeft.controller.publish_message(self.targetImagerLeft.imager_address, "stereoextrinsics",
                                                         serialized)
        filename = self.targetImagerLeft.stereo_filename()
        file = open(filename, "w")
        file.write(serialized)
        file.close()

        # save the recorded session data
        if self.get_container('RECORDING').value:
            dd['CALIBRATIONIMAGEHISTORY'] = ds.environment_containers['CALIBRATIONIMAGEHISTORY'].value
            dd['CONTROLLER'] = 'unknownctlr'
            dd['timestamp'] = datetime.datetime.now().isoformat()
            path = StereoCalculatorProcess.stereo_sessionfolder()
            num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))

            filename = "../CalibrationRecords/StereoSessions/session.{}.json".format(num_files + 1)
            with open(filename, 'w') as f:
                json.dump(dd, f, ensure_ascii=True)


class CalibratedIntrinsicsCalculatorProcess(IDEFProcess):

    def __init__(self, name, isLeftCamera, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.cameraCode = 'L' if isLeftCamera else 'R'
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight
        self.live_image_transfer = False

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RECORDING")]
        return container_list

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

        sc = self.get_container('INGESTMODE')
        sc.value = self.cameraCode
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('RAWIMAGE')
        sc.matrix = None
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

        numsamples = 24

        sc = self.get_container('NUMPAIRS')
        sc.value = numsamples
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = numsamples
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('IMAGER')
        sc.value = self.targetImagerLeft
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHMOVE')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('REPROJECTIONERROR')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        self.targetImagerLeft.set_image('compositechess', None, None)

    def create_stages(self):
        x = [RandomSolutionSearchStage(self, 'RandomSolutionSearchStage'),
             StoredImageInterfaceStage(self, 'StoredImageInterfaceStage'),
             CalibrationStage(self, 'CalibrationStage'),
             DistortionCalculatorStage(self, 'DistortionCalculatorStage')]
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['DistortionCalculatorStage']
        intrinsics = ds.output_containers['CAMERAINTRINSICS'].value  # type: ScalarContainer
        reperr = ds.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
        self.status_message("Reprojection error is {}".format(reperr.value))
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, numpy.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = uuid.uuid4().hex
        imager = self.locals['IMAGER'].value
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = imager.imager_address
        translated['MATCHCOUNT'] = self.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        sc = self.get_container('IMAGER').value
        sc.controller.publish_message(sc.imager_address, "intrinsics", cfg)
        filename = sc.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()

        filename = "../CalibrationRecords/IntrinsicsHistory.txt"
        if os.path.isfile(filename):
            file = open(filename, "a")
            file.write(",\n")
        else:
            file = open(filename, "w")

        file.write(cfg)
        file.close()


class GeneticIntrinsicsCalculatorProcess(IDEFProcess):

    def __init__(self, name, isLeftCamera, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.cameraCode = 'L' if isLeftCamera else 'R'
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight
        self.live_image_transfer = False

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RECORDING")]
        return container_list

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

        sc = self.get_container('INGESTMODE')
        sc.value = self.cameraCode
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('POPULATIONLIMIT')
        sc.value = 100
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        if StereoTrainingSet.data_ready(True):
            sts = StereoTrainingSet(True)
            sts.load_training_results()
            td = sts.extract_population(self.get_container('POPULATIONLIMIT').value)
            sc = self.get_container('POPULATIONSEED')
            sc.value = td
            sc.data_direction = EDataDirection.Input
            self.locals[sc.name] = sc

        sc = self.get_container('RAWIMAGE')
        sc.matrix = None
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

        numsamples = 24

        sc = self.get_container('NUMPAIRS')
        sc.value = numsamples
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = numsamples
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('IMAGER')
        sc.value = self.targetImagerLeft
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHMOVE')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('REPROJECTIONERROR')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        # load initialization data if it's avaiable
        self.targetImagerLeft.set_image('compositechess', None, None)

    def create_stages(self):
        x = [GeneticSolutionSearchStage(self, 'GeneticSolutionSearchStage'),
             StoredImageInterfaceStage(self, 'StoredImageInterfaceStage'),
             CalibrationStage(self, 'CalibrationStage'),
             DistortionCalculatorStage(self, 'DistortionCalculatorStage')]
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['DistortionCalculatorStage']
        intrinsics = ds.output_containers['CAMERAINTRINSICS'].value  # type: ScalarContainer
        reperr = ds.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
        self.status_message("Reprojection error is {}".format(reperr.value))
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, numpy.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = uuid.uuid4().hex
        imager = self.locals['IMAGER'].value
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = imager.imager_address
        translated['MATCHCOUNT'] = self.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        sc = self.get_container('IMAGER').value
        sc.controller.publish_message(sc.imager_address, "intrinsics", cfg)
        filename = sc.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()

        filename = "../CalibrationRecords/IntrinsicsHistory.txt"
        if os.path.isfile(filename):
            file = open(filename, "a")
            file.write(",\n")
        else:
            file = open(filename, "w")

        file.write(cfg)
        file.close()


class StereoIntrinsicsCalculatorProcess(IDEFProcess):

    def __init__(self, name, targetImagerLeft, targetImagerRight):
        IDEFProcess.__init__(self, name)
        self.targetImagerLeft = targetImagerLeft
        self.targetImagerRight = targetImagerRight

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RECORDING")]
        return container_list

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
        if StereoTrainingSet.data_ready(False):
            sts = StereoTrainingSet(True)
            sts.load_training_results()
            td = sts.extract_population(250)
            sc = self.get_container('STEREOTRAININGSET')
            sc.value = td
            sc.data_direction = EDataDirection.Input
            self.locals[sc.name] = sc
        else:
            raise ValueError('no stereo data available')

        sc = self.get_container('BOARDSIZE')
        sc.value = (9, 7)
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('IMAGESIZE')
        # todo This is assuming both imagers have equal resolution
        sc.value = self.targetImagerLeft.get_resolution()
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        numsamples = 24

        sc = self.get_container('NUMPAIRS')
        sc.value = numsamples
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHCOUNT')
        sc.value = numsamples
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHSEPARATION')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('LEFTIMAGER')
        sc.value = self.targetImagerLeft
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('RIGHTIMAGER')
        sc.value = self.targetImagerRight
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('MATCHMOVE')
        sc.value = 0
        sc.data_direction = EDataDirection.Input
        self.locals[sc.name] = sc

        sc = self.get_container('REPROJECTIONERROR')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('RAWIMAGELEFT')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        sc = self.get_container('RAWIMAGERIGHT')
        sc.data_direction = EDataDirection.Output
        self.locals[sc.name] = sc

        # load initialization data if it's avaiable
        self.targetImagerLeft.set_image('compositechess', None, None)

    def create_stages(self):
        x = [StereoSolutionSearchStage(self, 'StereoSolutionSearchStage'),
             StereoCalibrationStage(self, 'StereoCalibrationStage'),
             StereoCalibratorCalculatorStage(self, 'StereoCalibratorCalculatorStage')]
        self.add_stages(x)

    def on_completion(self):

        if Notifier.activeNotifier is not None:
            Notifier.activeNotifier.speak_message("completed")
        ds = self.stages['DistortionCalculatorStage']
        intrinsics = ds.output_containers['CAMERAINTRINSICS'].value  # type: ScalarContainer
        reperr = ds.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
        self.status_message("Reprojection error is {}".format(reperr.value))
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, numpy.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = uuid.uuid4().hex
        imager = self.locals['IMAGER'].value
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = imager.imager_address
        translated['MATCHCOUNT'] = self.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        sc = self.get_container('IMAGER').value
        sc.controller.publish_message(sc.imager_address, "intrinsics", cfg)
        filename = sc.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()

        filename = "../CalibrationRecords/IntrinsicsHistory.txt"
        if os.path.isfile(filename):
            file = open(filename, "a")
            file.write(",\n")
        else:
            file = open(filename, "w")

        file.write(cfg)
        file.close()
