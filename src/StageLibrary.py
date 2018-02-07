"""
Defines a number of useful predefined stages that can be composed into new processes
"""
import cv2
import numpy as np

from Notification import Notifier
from ProcessAbstractionLayer import IDEFStage, IDEFStageBinding, EConnection, EContainerType, EDataDirection, \
    ScalarContainer


class DifferentialStage(IDEFStage):
    """
    Provides a stream of differential images from the input
    """

    def __init__(self, host, name):
        super().__init__(host, name)

    def get_required_containers(self):
        " Need an input image (RAWIMAGE), a last image to compute difference from (LASTIMAGE),\
        and an output image (DIFFERENTIALIMAGE)"
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGE"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "LASTIMAGE"),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "MEANSQUAREDERROR"),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, "DIFFERENTIALIMAGE")]
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['LASTIMAGE'].matrix = None
        self.environment_containers['LASTIMAGE'].data_direction = EDataDirection.InputOutput

        self.output_containers['MEANSQUAREDERROR'].value = None
        self.output_containers['MEANSQUAREDERROR'].data_direction = EDataDirection.Output

        self.output_containers['DIFFERENTIALIMAGE'].matrix = None
        self.output_containers['DIFFERENTIALIMAGE'].data_direction = EDataDirection.Output

    def is_ready_to_run(self):
        return self.is_container_valid(EConnection.Input, 'RAWIMAGE', EDataDirection.Input)

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'DIFFERENTIALIMAGE', EDataDirection.Input)

    def process(self):
        image = self.input_containers['RAWIMAGE'].matrix
        lastimage = self.environment_containers['LASTIMAGE'].matrix
        if lastimage is not None:
            mse_container = self.output_containers['MEANSQUAREDERROR']
            mse = self.mean_squared_error(image, lastimage)
            mse_container.value = mse
            diff = cv2.absdiff(image, lastimage)
            self.output_containers['DIFFERENTIALIMAGE'].matrix = diff
            self.output_containers['DIFFERENTIALIMAGE'].data_direction = EDataDirection.Input
            self.output_containers['MEANSQUAREDERROR'].data_direction = EDataDirection.Input
            self.host_process.completed = True
        self.environment_containers['LASTIMAGE'].matrix = image.copy()

    "Structural Similarity Image Metric (SSIM)"

    def mean_squared_error(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err


class CalibrationStage(IDEFStage):
    """
    Hunts for checkerboard patterns and passes them off to consumers when found
    """

    def __init__(self, host, name):
        super().__init__(host, name)

    def get_required_containers(self):
        """
        Working image in RAWIMAGE, # squares on UV axes in BOARDSIZE, and an output listing located
        corners in CURRENTCHESSBOARDCORNERS
        :return:
        :rtype:
        """
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGE"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "BOARDSIZE"),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "CURRENTCHESSBOARDCORNERS"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "IMAGESIZE")]
        return container_list

    def initialize_container_impedance(self):
        self.output_containers['CURRENTCHESSBOARDCORNERS'].value = None
        self.output_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Output
        self.host_process.targetImager.raw_image['compositechess'] = None

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERS', EDataDirection.Output)
                and self.is_container_valid(EConnection.Control, 'BOARDSIZE', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGE', EDataDirection.Input))

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERS', EDataDirection.Input)

    def process(self):
        image = self.input_containers['RAWIMAGE'].matrix
        self.environment_containers['IMAGESIZE'].value = (image.shape[1], image.shape[0])
        boardsize = self.control_containers['BOARDSIZE'].value
        corner_container = self.output_containers['CURRENTCHESSBOARDCORNERS']

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, boardsize)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001))
            corner_container.value = corners2
            # mark the output valid
            self.output_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Input
            img = cv2.drawChessboardCorners(gray, (9, 7), corner_container.value, True)
            self.host_process.targetImager.set_image('chess', None, img)

            # generate the composite image
            if not 'compositechess' in self.host_process.targetImager.cv2_image_array:
                # create an empty black image
                blank_image = np.zeros(image.shape, np.uint8)
                img = cv2.drawChessboardCorners(blank_image, (9, 7), corner_container.value, True)
                self.host_process.targetImager.set_image('compositechess', None, img)
            else:
                img = cv2.drawChessboardCorners(self.host_process.targetImager.cv2_image_array['compositechess'], (9, 7),
                                                corner_container.value, True)
                self.host_process.targetImager.set_image('compositechess', None, img)
        else:
            self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Output


class StereoCalibrationStage(IDEFStage):
    """
    Hunts for checkerboard patterns and passes them off to consumers when found
    """

    def __init__(self, host, name):
        super().__init__(host, name)

    def get_required_containers(self):
        """
        Working image in RAWIMAGE, # squares on UV axes in BOARDSIZE, and an output listing located
        corners in CURRENTCHESSBOARDCORNERS
        :return:
        :rtype:
        """
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGELEFT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGERIGHT"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "BOARDSIZE"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "LEFTIMAGER"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RIGHTIMAGER"),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "CURRENTCHESSBOARDCORNERSLEFT"),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "CURRENTCHESSBOARDCORNERSRIGHT"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, "CALIBRATIONIMAGEHISTORY")]
        return container_list

    def initialize_container_impedance(self):
        self.output_containers['CURRENTCHESSBOARDCORNERSLEFT'].value = None
        self.output_containers['CURRENTCHESSBOARDCORNERSLEFT'].data_direction = EDataDirection.Output

        self.output_containers['CURRENTCHESSBOARDCORNERSRIGHT'].value = None
        self.output_containers['CURRENTCHESSBOARDCORNERSRIGHT'].data_direction = EDataDirection.Output

        self.environment_containers['CALIBRATIONIMAGEHISTORY'].value = []
        self.environment_containers['CALIBRATIONIMAGEHISTORY'].data_direction = EDataDirection.InputOutput

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERSLEFT', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERSRIGHT', EDataDirection.Output)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGELEFT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGERIGHT', EDataDirection.Input))

    def is_output_ready(self):
        return (self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERSLEFT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERSRIGHT', EDataDirection.Input))

    def process(self):
        # a base assumption is that candidate images are correlated
        left_image = self.input_containers['RAWIMAGELEFT'].matrix
        right_image = self.input_containers['RAWIMAGERIGHT'].matrix
        boardsize = self.control_containers['BOARDSIZE'].value
        left_corner_container = self.output_containers['CURRENTCHESSBOARDCORNERSLEFT']
        right_corner_container = self.output_containers['CURRENTCHESSBOARDCORNERSRIGHT']

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
        leftimager = self.host_process.get_container('LEFTIMAGER').value
        rightimager = self.host_process.get_container('RIGHTIMAGER').value
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = None
        leftimager.set_image('findchess', None, left_image)
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, boardsize)
        right_ret = False
        right_corners = None
        if left_ret:
            print('found left')
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            right_ret, right_corners = cv2.findChessboardCorners(right_gray, boardsize)

        if left_ret and right_ret:
            print('found both')
            left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (5, 5), (-1, -1),
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001))
            right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (5, 5), (-1, -1),
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001))
            left_corner_container.value = left_corners2
            right_corner_container.value = right_corners2
            # mark the output valid
            left_corner_container.data_direction = EDataDirection.Input
            right_corner_container.data_direction = EDataDirection.Input

            img = cv2.drawChessboardCorners(left_gray, (9, 7), left_corner_container.value, True)
            leftimager.set_image('chess', None, img)

            img = cv2.drawChessboardCorners(right_gray, (9, 7), right_corner_container.value, True)
            rightimager.set_image('chess', None, img)

            # record for posterity
            if self.control_containers['RECORDING'].value:
                filekey = self.host_process.record_stereo_images("unknownctlr", left_image, right_image)
                self.environment_containers['CALIBRATIONIMAGEHISTORY'].value.append(filekey)
        else:
            self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Output
            self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Output


class ChessboardProcessorStage(IDEFStage):
    def initialize_container_impedance(self):
        pass

    def is_ready_to_run(self):
        pass

    def is_output_ready(self):
        pass

    def process(self):
        pass

    def get_required_containers(self):
        container_list = [IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'BOARDSIZE'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'MATCHCOUNT'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'MATCHMOVE'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'MATCHSEPARATION'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'IMAGESIZE'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'RECTIFY_ALPHA'),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, 'REPROJECTIONERROR')]
        return container_list

    @staticmethod
    def compute_reprojection_error(imgpoints, objpoints, k, dist, rvecs, tvecs):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], k, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error / len(objpoints)

    @staticmethod
    def average_corner(corners):
        mean = abs(np.mean(corners, axis=0))
        return mean


class DistortionCalculatorStage(ChessboardProcessorStage):
    """Processes located checkerboard patterns to compute lens intrinsics"""

    def __init__(self, host, name):
        super().__init__(host, name)

    def get_required_containers(self):
        """
        RAWIMAGE is used to show accepted checkerboard, CURRENTCHESSBOARDCORNERS is the new
        set of corners, BOARDSIZE defines uv dimensions of checkerboard, MATCHCOUNT defines
        the number of samples required, MATCHSEPARATION defines the minimum distance between
        samples, IMAGESIZE defines source image size,RECTIFY_ALPHA defines alpha rectification
        coefficient where 0 is all points, and 1 is only valid points, CHESSBOARDCORNERLIST for
        storing the collected chessboard corners,and CAMERAINSTRINCS for reported calculated camera
        matrix and distortion coefficients
        :return:
        :rtype:
        """
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGE"),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'CURRENTCHESSBOARDCORNERS'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, 'CHESSBOARDCORNERLIST'),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, 'CAMERAINTRINSICS'),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, 'REPROJECTIONERROR')]
        container_list.extend(super().get_required_containers())
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['CHESSBOARDCORNERLIST'].data_direction = EDataDirection.InputOutput
        self.environment_containers['CHESSBOARDCORNERLIST'].value = []
        self.output_containers['CAMERAINTRINSICS'].data_direction = EDataDirection.Output

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Input, 'CURRENTCHESSBOARDCORNERS', EDataDirection.Input)
                and self.is_container_valid(EConnection.Control, 'BOARDSIZE', EDataDirection.Input)
                and self.is_container_valid(EConnection.Control, 'MATCHCOUNT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Control, 'MATCHSEPARATION', EDataDirection.Input)
                and self.is_container_valid(EConnection.Control, 'MATCHMOVE', EDataDirection.Input)
                and self.is_container_valid(EConnection.Control, 'IMAGESIZE', EDataDirection.Input)
                and self.is_container_valid(EConnection.Environment, 'CHESSBOARDCORNERLIST', EDataDirection.InputOutput)
                and self.is_container_valid(EConnection.Output, 'CAMERAINTRINSICS', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'REPROJECTIONERROR', EDataDirection.Output))

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'CAMERAINTRINSICS', EDataDirection.Input)

    def process(self):
        print('analyse')
        boardsize = self.control_containers['BOARDSIZE'].value
        imagesize = self.control_containers['IMAGESIZE'].value
        mindistance = self.control_containers['MATCHSEPARATION'].value
        matchmove = self.control_containers['MATCHMOVE'].value
        matchcount = self.control_containers['MATCHCOUNT'].value
        corner_container = self.input_containers['CURRENTCHESSBOARDCORNERS']
        cornerlist = self.environment_containers['CHESSBOARDCORNERLIST'].value

        self.host_process.status_message("Processing {} of {} calibration tests".format(len(cornerlist), matchcount))
        avgpt = self.average_corner(corner_container.value)
        if len(cornerlist) == 0:
            cornerlist.append(corner_container.value)
            self.input_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Output
        else:
            reject = False
            # make sure they've moved far enough away from the previous point
            lastavg = self.average_corner(cornerlist[-1])
            dist = cv2.norm(lastavg - avgpt)
            if dist <= matchmove:
                reject = True
            if not reject:
                for corners in cornerlist:
                    t_avg = self.average_corner(corners)
                    dist = cv2.norm(t_avg - avgpt)
                    if dist <= mindistance:
                        reject = True
                        break
            if not reject:
                image = self.input_containers['RAWIMAGE'].matrix

                if len(cornerlist) < matchcount:
                    cornerlist.append(corner_container.value)
                    if len(cornerlist) % (matchcount / 4) == 0 and Notifier.activeNotifier is not None:
                        Notifier.activeNotifier.play_soundfile_async('bityes.wav')
                else:
                    if Notifier.activeNotifier is not None:
                        Notifier.activeNotifier.speak_message("Calculating")
                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((boardsize[0] * boardsize[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1, 2)
                    oblocs = []
                    for i in range(0, len(cornerlist)):
                        oblocs.append(objp)

                    ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(oblocs, cornerlist,
                                                                     (imagesize[0],
                                                                      imagesize[1]), None, None)
                    if ret:
                        intrinsics = self.output_containers['CAMERAINTRINSICS']  # type: ScalarContainer
                        reprojerr = self.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
                        re = self.compute_reprojection_error(cornerlist, oblocs, k, dist, rvecs, tvecs)
                        reprojerr.value = re
                        # todo  ROTATIONVECTORS snd TRANSLATIONVECTORS sre not propagated  - they may be ephemeral

                        dd = {
                            'CAMERAMATRIX': k, 'DISTORTIONCOEFFICIENTS': dist,
                            "REPROJECTIONERROR": re
                        }

                        intrinsics.value = dd
                        # output is ready for consumer
                        intrinsics.data_direction = EDataDirection.Input
                        reprojerr.data_direction = EDataDirection.Input
                        # and we're done
                        self.host_process.completed = True
                        print("intrinsics computed")
                    else:
                        print("could not calculate intrinsics from sample set")
        self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Output
        self.input_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Output


class StereoCalibratorCalculatorStage(ChessboardProcessorStage):
    """Processes located checkerboard patterns to compute interlens stereo transform"""

    def __init__(self, host, name):
        super().__init__(host, name)

    def get_required_containers(self):
        """
        RAWIMAGE is used to show accepted checkerboard, CURRENTCHESSBOARDCORNERS is the new
        set of corners, BOARDSIZE defines uv dimensions of checkerboard, MATCHCOUNT defines
        the number of samples required, MATCHSEPARATION defines the minimum distance between
        samples, IMAGESIZE defines source image size,RECTIFY_ALPHA defines alpha rectification
        coefficient where 0 is all points, and 1 is only valid points, CHESSBOARDCORNERLIST for
        storing the collected chessboard corners,and CAMERAINSTRINCS for reported calculated camera
        matrix and distortion coefficients
        :return:
        :rtype:
        """
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGELEFT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGERIGHT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "CAMERAMATRIXLEFT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "CAMERAMATRIXRIGHT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "DISTORTIONCOEFFSLEFT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "DISTORTIONCOEFFSRIGHT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'CURRENTCHESSBOARDCORNERSLEFT'),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'CURRENTCHESSBOARDCORNERSRIGHT'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, 'CHESSBOARDCORNERLISTLEFT'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, 'CHESSBOARDCORNERLISTRIGHT'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, 'CALIBRATIONIMAGEHISTORY'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, 'STEREOROTATION'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, 'STEREOTRANSLATION'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, 'ESSENTIAL'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, 'FUNDAMENTAL')]
        container_list.extend(super().get_required_containers())
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['CHESSBOARDCORNERLISTLEFT'].data_direction = EDataDirection.InputOutput
        self.environment_containers['CHESSBOARDCORNERLISTLEFT'].value = []
        self.environment_containers['CHESSBOARDCORNERLISTRIGHT'].data_direction = EDataDirection.InputOutput
        self.environment_containers['CHESSBOARDCORNERLISTRIGHT'].value = []
        self.output_containers['STEREOROTATION'].data_direction = EDataDirection.Output
        self.output_containers['STEREOTRANSLATION'].data_direction = EDataDirection.Output
        self.output_containers['ESSENTIAL'].data_direction = EDataDirection.Output
        self.output_containers['FUNDAMENTAL'].data_direction = EDataDirection.Output
        self.output_containers['REPROJECTIONERROR'].data_direction = EDataDirection.Output

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Input, 'CURRENTCHESSBOARDCORNERSLEFT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'CURRENTCHESSBOARDCORNERSRIGHT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Environment, 'CHESSBOARDCORNERLISTLEFT',
                                            EDataDirection.InputOutput)
                and self.is_container_valid(EConnection.Environment, 'CHESSBOARDCORNERLISTRIGHT',
                                            EDataDirection.InputOutput)
                and self.is_container_valid(EConnection.Output, 'STEREOROTATION', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'STEREOTRANSLATION', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'ESSENTIAL', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'FUNDAMENTAL', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'REPROJECTIONERROR', EDataDirection.Output))

    def is_output_ready(self):
        return (self.is_container_valid(EConnection.Output, 'STEREOROTATION', EDataDirection.Input)
                and self.is_container_valid(EConnection.Output, 'STEREOTRANSLATION', EDataDirection.Input)
                and self.is_container_valid(EConnection.Output, 'ESSENTIAL', EDataDirection.Input)
                and self.is_container_valid(EConnection.Output, 'FUNDAMENTAL', EDataDirection.Input)
                and self.is_container_valid(EConnection.Output, 'REPROJECTIONERROR', EDataDirection.Input))

    def process(self):
        print('analyse')
        boardsize = self.control_containers['BOARDSIZE'].value
        imagesize = self.control_containers['IMAGESIZE'].value
        mindistance = self.control_containers['MATCHSEPARATION'].value
        matchcount = self.control_containers['MATCHCOUNT'].value
        left_corner_container = self.input_containers['CURRENTCHESSBOARDCORNERSLEFT']
        right_corner_container = self.input_containers['CURRENTCHESSBOARDCORNERSRIGHT']
        left_cornerlist = self.environment_containers['CHESSBOARDCORNERLISTLEFT'].value
        right_cornerlist = self.environment_containers['CHESSBOARDCORNERLISTRIGHT'].value

        left_cm = self.input_containers['CAMERAMATRIXLEFT'].matrix
        right_cm = self.input_containers['CAMERAMATRIXRIGHT'].matrix
        left_dc = self.input_containers['DISTORTIONCOEFFSLEFT'].matrix
        right_dc = self.input_containers['DISTORTIONCOEFFSRIGHT'].matrix
        self.host_process.status_message(
            "Processing {} of {} stereo calibration tests".format(len(left_cornerlist), matchcount))
        left_avgpt = self.average_corner(left_corner_container.value)
        right_avgpt = self.average_corner(right_corner_container.value)
        if len(left_cornerlist) == 0:
            left_cornerlist.append(left_corner_container.value)
            right_cornerlist.append(right_corner_container.value)
            self.input_containers['CURRENTCHESSBOARDCORNERSLEFT'].data_direction = EDataDirection.Output
            self.input_containers['CURRENTCHESSBOARDCORNERSRIGHT'].data_direction = EDataDirection.Output
        else:
            reject = False
            for corners in left_cornerlist:
                t_avg = self.average_corner(corners)
                dist = cv2.norm(t_avg - left_avgpt)
                if dist <= mindistance:
                    reject = True
                    break
            if not reject:
                for corners in right_cornerlist:
                    t_avg = self.average_corner(corners)
                    dist = cv2.norm(t_avg - right_avgpt)
                    if dist <= mindistance:
                        reject = True
                        break
            # has to be distinct in both environments
            if not reject:
                if len(left_cornerlist) < matchcount:
                    left_cornerlist.append(left_corner_container.value)
                    right_cornerlist.append(right_corner_container.value)
                    if len(left_cornerlist) % (matchcount / 4) == 0 and Notifier.activeNotifier is not None:
                        Notifier.activeNotifier.play_soundfile_async('bityes.wav')
                else:
                    if Notifier.activeNotifier is not None:
                        Notifier.activeNotifier.speak_message("Calculating")
                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((boardsize[0] * boardsize[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1, 2)
                    oblocs = []
                    for i in range(0, len(left_cornerlist)):
                        oblocs.append(objp)
                    flags = 0
                    flags |= cv2.CALIB_FIX_INTRINSIC
                    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
                    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
                    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
                    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
                    flags |= cv2.CALIB_ZERO_TANGENT_DIST
                    # flags |= cv2.CALIB_RATIONAL_MODEL
                    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
                    # flags |= cv2.CALIB_FIX_K3
                    # flags |= cv2.CALIB_FIX_K4
                    # flags |= cv2.CALIB_FIX_K5

                    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)

                    (rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F) = \
                        cv2.stereoCalibrate(oblocs, left_cornerlist, right_cornerlist, left_cm, left_dc, right_cm,
                                            right_dc,
                                            imagesize, criteria=stereocalib_criteria, flags=flags)

                    self.output_containers['REPROJECTIONERROR'].set_output_for_subsequent_input(rms_stereo)
                    self.output_containers['STEREOROTATION'].set_output_for_subsequent_input(R)
                    self.output_containers['STEREOTRANSLATION'].set_output_for_subsequent_input(T)
                    self.output_containers['ESSENTIAL'].set_output_for_subsequent_input(E)
                    self.output_containers['FUNDAMENTAL'].set_output_for_subsequent_input(F)

                    # in case things got recomputed
                    self.input_containers['CAMERAMATRIXLEFT'].matrix = camera_matrix_l
                    self.input_containers['CAMERAMATRIXRIGHT'].matrix = camera_matrix_r
                    self.input_containers['DISTORTIONCOEFFSLEFT'].matrix = dist_coeffs_l
                    self.input_containers['DISTORTIONCOEFFSRIGHT'].matrix = dist_coeffs_r

                    # mark process complete
                    self.host_process.completed = True
                    print("stereo computed")
        self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Output
        self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Output
        self.input_containers['CURRENTCHESSBOARDCORNERSLEFT'].data_direction = EDataDirection.Output
        self.input_containers['CURRENTCHESSBOARDCORNERSRIGHT'].data_direction = EDataDirection.Output
