"""
Defines a number of useful predefined stages that can be composed into new processes
"""
import datetime
import json
import os
import random
import uuid

import cv2
import numpy as np

from HardwareAbstractionLayer import Imager, Controller
from Notification import Notifier
from ProcessAbstractionLayer import IDEFStage, IDEFStageBinding, EConnection, EContainerType, EDataDirection, \
    ScalarContainer, IDEFProcess
from TrainingSupport import StereoTrainingSet


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
                                           EDataDirection.Input, "IMAGESIZE"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "IMAGER")]
        return container_list

    def initialize_container_impedance(self):
        self.output_containers['CURRENTCHESSBOARDCORNERS'].value = None
        self.output_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Output

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERS', EDataDirection.Output)
                and self.is_container_valid(EConnection.Control, 'BOARDSIZE', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGE', EDataDirection.Input))

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'CURRENTCHESSBOARDCORNERS', EDataDirection.Input)

    def process(self):
        image = self.input_containers['RAWIMAGE'].matrix
        self.environment_containers['IMAGESIZE'].value = (image.shape[0], image.shape[1])
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
            self.environment_containers['IMAGER'].value.set_image('chess', None, img)

        else:
            self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Output


class ChessboardCaptureStage(IDEFStage):
    """
    Hunts for checkerboard patterns and passes them off to consumers when found
    """

    def __init__(self, host, name):
        super().__init__(host, name)
        self.counter = 0

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
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "IMAGEKEY"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "BOARDSIZE"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "LEFTIMAGER"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RIGHTIMAGER"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, "CALIBRATIONIMAGEHISTORY")]
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['CALIBRATIONIMAGEHISTORY'].value = []
        self.environment_containers['CALIBRATIONIMAGEHISTORY'].data_direction = EDataDirection.InputOutput

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Input, 'RAWIMAGELEFT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGERIGHT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'IMAGEKEY', EDataDirection.Input))

    def is_output_ready(self):
        return False

    def process(self):
        # a base assumption is that candidate images are correlated
        iid = self.input_containers['IMAGEKEY'].value
        left_image = self.input_containers['RAWIMAGELEFT'].matrix
        right_image = self.input_containers['RAWIMAGERIGHT'].matrix
        boardsize = self.control_containers['BOARDSIZE'].value

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
        leftimager = self.host_process.get_container('LEFTIMAGER').value
        rightimager = self.host_process.get_container('RIGHTIMAGER').value
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, boardsize)
        if left_ret:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            right_ret, right_corners = cv2.findChessboardCorners(right_gray, boardsize)
            if right_ret:
                left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (5, 5), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001))
                # generate the composite image
                # if 'compositechess' not in leftimager.cv2_image_array:
                #     # create an empty black image
                #     blank_image = np.zeros(left_image.shape[:2], np.uint8)
                #     img = cv2.drawChessboardCorners(blank_image, (9, 7), left_corners2, True)
                #     leftimager.set_image('compositechess', None, img)
                # else:
                #     img = cv2.drawChessboardCorners(leftimager.cv2_image_array['compositechess'],
                #                                     (9, 7),
                #                                     left_corners2, True)
                #     leftimager.set_image('compositechess', None, img)
                #
                # right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (5, 5), (-1, -1),
                #                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001))
                # # generate the composite image
                # if 'compositechess' not in rightimager.cv2_image_array:
                #     # create an empty black image
                #     blank_image = np.zeros(right_image.shape[:2], np.uint8)
                #     img = cv2.drawChessboardCorners(blank_image, (9, 7), right_corners2, True)
                #     rightimager.set_image('compositechess', None, img)
                # else:
                #     img = cv2.drawChessboardCorners(rightimager.cv2_image_array['compositechess'],
                #                                     (9, 7),
                #                                     right_corners2, True)
                #     rightimager.set_image('compositechess', None, img)
                # record for posterity
                self.host_process.record_stereo_images("unknownctlr", iid, left_image, right_image)
                self.counter += 1
                if self.counter % 10 == 0:
                    s = str(self.counter)
                    Notifier.activeNotifier.speak_message(s)

        self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Output
        self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Output
        self.input_containers['IMAGEKEY'].data_direction = EDataDirection.Output


class StereoCalibrationStage(IDEFStage):
    """
   Handles batch stereo calibration intake cycle
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
                                           EDataDirection.InputOutput, "CALIBRATIONIMAGEHISTORY"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "IMAGESIZE")]
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
        self.environment_containers['IMAGESIZE'].value = (left_image.shape[0], left_image.shape[1])
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
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            right_ret, right_corners = cv2.findChessboardCorners(right_gray, boardsize)

        if left_ret and right_ret:
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
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, 'IMAGER'),
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
        self.output_containers['REPROJECTIONERROR'].data_direction = EDataDirection.Output

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

    def draw_matched_corners(self, corner_container):
        """
        Update the composite chess image by adding the given set of corners - used to track where chessboard
        coordinates have been accepted
        :param corner_container:
        :type corner_container:
        :return:
        :rtype:
        """
        # generate the composite image
        if 'compositechess' not in self.environment_containers['IMAGER'].value.cv2_image_array:
            # create an empty black image
            blank_image = np.zeros(self.control_containers['IMAGESIZE'].value, np.uint8)
            img = cv2.drawChessboardCorners(blank_image, (9, 7), corner_container.value, True)
            self.environment_containers['IMAGER'].value.set_image('compositechess', None, img)
        else:
            img = cv2.drawChessboardCorners(
                self.environment_containers['IMAGER'].value.cv2_image_array['compositechess'],
                (9, 7),
                corner_container.value, True)
            self.environment_containers['IMAGER'].value.set_image('compositechess', None, img)

    def process(self):
        boardsize = self.control_containers['BOARDSIZE'].value
        imagesize = self.control_containers['IMAGESIZE'].value
        mindistance = self.control_containers['MATCHSEPARATION'].value
        matchmove = self.control_containers['MATCHMOVE'].value
        matchcount = self.control_containers['MATCHCOUNT'].value
        corner_container = self.input_containers['CURRENTCHESSBOARDCORNERS']
        cornerlist = self.environment_containers['CHESSBOARDCORNERLIST'].value

        self.host_process.status_message(
            "Processing {} of {} calibration tests".format(len(cornerlist) + 1, matchcount))
        avgpt = self.average_corner(corner_container.value)
        if len(cornerlist) == 0:
            cornerlist.append(corner_container.value)
            self.input_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Output
            self.draw_matched_corners(corner_container)
        else:
            reject = False
            if matchmove > 0 and mindistance > 0:
                # make sure they've moved far enough away from the previous point
                lastavg = self.average_corner(cornerlist[-1])
                dist = cv2.norm(lastavg - avgpt)
                if dist < matchmove:
                    reject = True
                if not reject:
                    for corners in cornerlist:
                        t_avg = self.average_corner(corners)
                        dist = cv2.norm(t_avg - avgpt)
                        if dist < mindistance:
                            reject = True
                            break
            if not reject:
                # self.draw_matched_corners(corner_container)
                cornerlist.append(corner_container.value)
                if len(cornerlist) == matchcount:
                    self.host_process.status_message("Calculating...")
                    # if Notifier.activeNotifier is not None:
                    #     Notifier.activeNotifier.speak_message("Calculating")
                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((boardsize[0] * boardsize[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1, 2)
                    oblocs = []
                    for i in range(0, len(cornerlist)):
                        oblocs.append(objp)

                    ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(oblocs, cornerlist,
                                                                     imagesize,
                                                                     Imager.perfect_camera_matrix(imagesize), None,
                                                                     flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO)
                    if ret:
                        self.host_process.status_message("Accepted...")
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
                        self.host_process.status_message("Completed")
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
        boardsize = self.control_containers['BOARDSIZE'].value
        imagesize = self.control_containers['IMAGESIZE'].value
        mindistance = self.control_containers['MATCHSEPARATION'].value
        matchcount = self.control_containers['MATCHCOUNT'].value
        left_corner_container = self.input_containers['CURRENTCHESSBOARDCORNERSLEFT']
        right_corner_container = self.input_containers['CURRENTCHESSBOARDCORNERSRIGHT']
        left_cornerlist = self.environment_containers['CHESSBOARDCORNERLISTLEFT'].value
        right_cornerlist = self.environment_containers['CHESSBOARDCORNERLISTRIGHT'].value

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
                    # if len(left_cornerlist) % (matchcount / 4) == 0 and Notifier.activeNotifier is not None:
                    #    Notifier.activeNotifier.play_soundfile_async('bityes.wav')
                if len(left_cornerlist) == matchcount:
                    # if Notifier.activeNotifier is not None:
                    #    Notifier.activeNotifier.speak_message("Calculating")
                    left_cm = self.input_containers['CAMERAMATRIXLEFT'].matrix
                    right_cm = self.input_containers['CAMERAMATRIXRIGHT'].matrix
                    left_dc = self.input_containers['DISTORTIONCOEFFSLEFT'].matrix
                    right_dc = self.input_containers['DISTORTIONCOEFFSRIGHT'].matrix
                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((boardsize[0] * boardsize[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1, 2)
                    oblocs = []
                    for i in range(0, len(left_cornerlist)):
                        oblocs.append(objp)
                    # flags = 0
                    # flags |= cv2.CALIB_FIX_INTRINSIC
                    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
                    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
                    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
                    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
                    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
                    # flags |= cv2.CALIB_RATIONAL_MODEL
                    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
                    # flags |= cv2.CALIB_FIX_K3
                    # flags |= cv2.CALIB_FIX_K4
                    # flags |= cv2.CALIB_FIX_K5
                    # flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
                    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
                    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)
                    T = np.zeros((3, 1), dtype=np.float64)
                    R = np.eye(3, dtype=np.float64)
                    # (rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R1, T1, E, F) = \
                    #     cv2.stereoCalibrate(oblocs, left_cornerlist, right_cornerlist, left_cm, left_dc, right_cm,
                    #                         right_dc,
                    #                         imagesize, R, T, criteria=stereocalib_criteria, flags=flags)
                    (rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R1, T1, E, F) = \
                        cv2.stereoCalibrate(oblocs, left_cornerlist, right_cornerlist, left_cm, left_dc, right_cm,
                                            right_dc,
                                            imagesize, None, None, None, None, criteria=stereocalib_criteria,
                                            flags=flags)
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
        self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Output
        self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Output
        self.input_containers['CURRENTCHESSBOARDCORNERSLEFT'].data_direction = EDataDirection.Output
        self.input_containers['CURRENTCHESSBOARDCORNERSRIGHT'].data_direction = EDataDirection.Output


class StereoSolutionSearchStage(IDEFStage):
    """
        Direct Crossover Ratio -  percentage of genes that will swap at same index
        Perturbed Crossover Ratio - percentage of genes that will swap at different indices
        Mutation Ratio - percentage of genes that will mutate to different images
        Offspring - number of new chromosomes produced from the two parent chromosomes


    """

    def __init__(self, host, name):
        super().__init__(host, name)

        self.stereo_pairs = None
        self.training_dictionary = None
        self.left_intrinsics = None
        self.right_intrinsics = None

    def get_required_containers(self):
        " Need an input image (RAWIMAGE), a last image to compute difference from (LASTIMAGE),\
        and an output image (DIFFERENTIALIMAGE)"
        container_list = [IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "PAIRINDEX"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "NUMPAIRS"),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'STEREOTRAININGSET'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'STEREOTRAININGCHROMOSOME'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'STEREOTRAININGCHROMOSOMEINDEX'),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'STEREOPAIR'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, 'CAMERAMATRIXLEFT'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, 'CAMERAMATRIXRIGHT'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, 'DISTORTIONCOEFFSLEFT'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, 'DISTORTIONCOEFFSRIGHT'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, 'RAWIMAGELEFT'),
                          IDEFStageBinding(EConnection.Output, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, 'RAWIMAGERIGHT')]
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['PAIRINDEX'].value = 0
        self.environment_containers['PAIRINDEX'].data_direction = EDataDirection.Input

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Input, 'STEREOTRAININGSET', EDataDirection.Input)
                and self.is_container_valid(EConnection.Output, 'RAWIMAGELEFT', EDataDirection.Output)
                and self.is_container_valid(EConnection.Output, 'RAWIMAGERIGHT', EDataDirection.Output))

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'STEREOPAIR', EDataDirection.Input)

    def buildmat(self, dv):
        X = np.empty(shape=[dv[0], dv[1]])
        dvi = 2
        for i in range(dv[0]):
            for j in range(dv[1]):
                X[i, j] = dv[dvi]
                dvi += 1
        return X

    def process(self):
        """For each StereoTrainingChromosom in the StereoTrainingSet
            Load up the left and right camera intrinsics
            Load the chromosome
            For each gene in the chromosome
                Feed the images identified by the gene into the left and right intakes
                Allow the downstream processing to run
            On completion, the stereo calibration should be calculate for this StereoTrainingChromosome
            Record the output information
        """
        if self.environment_containers['STEREOTRAININGCHROMOSOMEINDEX'].value is None:
            "first entry - set up for iteration across the training set"
            self.environment_containers['STEREOTRAININGCHROMOSOMEINDEX'].value = 0
        if self.environment_containers['STEREOTRAININGCHROMOSOME'].value is None:
            "have we eaten all of the chromosomes"
            if (self.environment_containers['STEREOTRAININGCHROMOSOMEINDEX'].value ==
                    len(self.input_containers['STEREOTRAININGSET'].value)):
                "COMPLETE PROCESSING - ALL DONE"
                self.host_process.completed = True
            "cycling to the next training chromosome"
            self.environment_containers['STEREOTRAININGCHROMOSOME'].value = \
                self.input_containers['STEREOTRAININGSET'].value[
                    self.environment_containers['STEREOTRAININGCHROMOSOMEINDEX'].value]
            self.environment_containers['STEREOTRAININGCHROMOSOMEINDEX'].value += 1
            " camera intrinsics are constant across the genes in the chromosome"
            self.output_containers['CAMERAMATRIXLEFT'].matrix = \
                self.buildmat(self.environment_containers['STEREOTRAININGCHROMOSOME'].value.left_camera_matrix)
            self.output_containers['DISTORTIONCOEFFSLEFT'].matrix = \
                self.buildmat(
                    self.environment_containers['STEREOTRAININGCHROMOSOME'].value.left_distortion_coefficients)
            self.output_containers['CAMERAMATRIXLEFT'].matrix = \
                self.buildmat(self.environment_containers['STEREOTRAININGCHROMOSOME'].value.left_camera_matrix)
            self.output_containers['CAMERAMATRIXLEFT'].data_direction = EDataDirection.Output
            self.output_containers['DISTORTIONCOEFFSLEFT'].matrix = \
                self.buildmat(
                    self.environment_containers['STEREOTRAININGCHROMOSOME'].value.left_distortion_coefficients)
            self.output_containers['DISTORTIONCOEFFSLEFT'].data_direction = EDataDirection.Output
            self.output_containers['CAMERAMATRIXRIGHT'].matrix = \
                self.buildmat(self.environment_containers['STEREOTRAININGCHROMOSOME'].value.right_camera_matrix)
            self.output_containers['CAMERAMATRIXRIGHT'].data_direction = EDataDirection.Output
            self.output_containers['DISTORTIONCOEFFSRIGHT'].matrix = \
                self.buildmat(
                    self.environment_containers['STEREOTRAININGCHROMOSOME'].value.right_distortion_coefficients)
            self.output_containers['DISTORTIONCOEFFSRIGHT'].data_direction = EDataDirection.Output
            self.environment_containers['PAIRINDEX'].value = 0
        pi = self.environment_containers['PAIRINDEX'].value
        if pi == 0:
            # reset the downstream processors
            self.host_process.stages['StereoCalibrationStage'].initialize_container_impedance()
            self.host_process.stages['StereoCalibratorCalculatorStage'].initialize_container_impedance()

        if pi >= self.control_containers['NUMPAIRS'].value:
            " cycle completed"
            self.cycle_completed()

            return

        fileid = self.environment_containers['STEREOTRAININGCHROMOSOME'].value.chromosome[pi]
        self.output_containers['RAWIMAGELEFT'].matrix = IDEFProcess.fetch_stereo_image_matrix(fileid, "unknownctlr",
                                                                                              'L')
        self.output_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Input
        self.output_containers['RAWIMAGERIGHT'].matrix = IDEFProcess.fetch_stereo_image_matrix(fileid, "unknownctlr",
                                                                                               'R')
        self.output_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Input

        self.environment_containers['PAIRINDEX'].value += 1

    def cycle_completed(self):
        self.environment_containers['STEREOTRAININGCHROMOSOME'].value = None
        ds = self.host_process.stages['StereoCalibratorCalculatorStage']
        if ds.output_containers['REPROJECTIONERROR'].value is None:
            print('stereo failed')
            return
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
        msg = "RMS Stereo error is {}".format(dd['Q'])
        print(msg)
        self.host_process.status_message(msg)
        filename = "../CalibrationRecords/StereoIntrinsicCalcs.txt"
        if os.path.isfile(filename):
            file = open(filename, "a")
            file.write(",\n")
        else:
            file = open(filename, "w")

        file.write(json.dumps(dd, ensure_ascii=False))
        file.close()

    def log_data(self, identifier, chromosome, intrinsics, camera_index):
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, np.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = identifier
        imager = self.host_process.locals['IMAGER'].value
        translated['DATASET'] = chromosome
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = camera_index
        translated['MATCHCOUNT'] = self.host_process.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.host_process.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        sc = self.host_process.get_container('IMAGER').value
        # sc.controller.publish_message(sc.imager_address, "intrinsics", cfg)

        filename = sc.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()


class GeneticSolutionSearchStage(IDEFStage):
    """
        Direct Crossover Ratio -  percentage of genes that will swap at same index
        Perturbed Crossover Ratio - percentage of genes that will swap at different indices
        Mutation Ratio - percentage of genes that will mutate to different images
        Offspring - number of new chromosomes produced from the two parent chromosomes


    """

    def __init__(self, host, name):
        super().__init__(host, name)

        self.num_passes = 2500
        self.stereo_pairs = None
        self.training_dictionary = None
        self.parent_chromosomes = None
        self.cossover_perturb_ratio = 0.05
        self.mutation_ratio = 0.05
        self.wins = 0
        self.losses = 0
        self.breeding_pool = []
        self.min_viable_population = 1
        self.minimum_fitness = 0
        self.leftside = True
        self.left_fitness = None
        self.left_intrinsics = None
        self.right_fitness = None
        self.right_intrinsics = None

        self.wait_on_computation = False
        self.training_dictionary = IDEFProcess.extract_training_dictionary()
        self.stereo_pairs = IDEFProcess.extract_training_stereo_pairs(self.training_dictionary)

    def get_required_containers(self):
        " Need an input image (RAWIMAGE), a last image to compute difference from (LASTIMAGE),\
        and an output image (DIFFERENTIALIMAGE)"
        container_list = [IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "SELECTED_STEREO_PAIRS"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "PAIRINDEX"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "NUMPAIRS"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "INGESTMODE"),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "STEREOPAIR"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'CAMERAINTRINSICS'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'REPROJECTIONERROR'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'POPULATIONLIMIT'),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'POPULATIONSEED')]
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['PAIRINDEX'].value = 0
        self.environment_containers['PAIRINDEX'].data_direction = EDataDirection.Input

    def is_ready_to_run(self):
        has_sp = self.is_container_valid(EConnection.Output, 'STEREOPAIR', EDataDirection.Output)
        if not has_sp:
            return False
        return True

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'STEREOPAIR', EDataDirection.Input)

    def process(self):
        if self.input_containers['POPULATIONSEED'].value is not None:
            for r in self.input_containers['POPULATIONSEED'].value:
                fm = (r.left_fitness(), r.right_fitness())
                c = r.chromosome
                self.add_breeder(c, fm)
            self.input_containers['POPULATIONSEED'].value = None
        pi = self.environment_containers['PAIRINDEX'].value
        if pi == 0:
            # reset the downstream processors
            self.host_process.stages['CalibrationStage'].initialize_container_impedance()
            self.host_process.stages['DistortionCalculatorStage'].initialize_container_impedance()

        if self.parent_chromosomes is None:
            self.update_parent_chromosomes()

        if self.environment_containers['SELECTED_STEREO_PAIRS'].value is None:
            spi = self.generate_child(self.cossover_perturb_ratio, self.mutation_ratio)
            self.environment_containers['SELECTED_STEREO_PAIRS'].value = spi
            self.environment_containers['PAIRINDEX'].value = 0
            self.leftside = True
            self.control_containers['INGESTMODE'].value = 'L'
            self.wait_on_computation = False

        if pi >= self.control_containers['NUMPAIRS'].value:
            if self.leftside:
                " figure out if you want to bail out early because the left child is not viable"

                " flip over to the right side"
                ds = self.host_process.stages['DistortionCalculatorStage']
                self.left_fitness = 1.0 - ds.output_containers['REPROJECTIONERROR'].value
                self.left_intrinsics = ds.output_containers['CAMERAINTRINSICS'].value
                # reset the pair counter
                self.environment_containers['PAIRINDEX'].value = 0
                self.leftside = False
                self.control_containers['INGESTMODE'].value = 'R'
                # restart feeding chromosome from other imager
                self.wait_on_computation = True
                return
            ds = self.host_process.stages['DistortionCalculatorStage']
            self.right_fitness = 1.0 - ds.output_containers['REPROJECTIONERROR'].value
            self.right_intrinsics = ds.output_containers['CAMERAINTRINSICS'].value
            " cycle completed"
            self.cycle_completed(self.environment_containers['SELECTED_STEREO_PAIRS'].value)
            self.wait_on_computation = False
            return

        fileid = self.environment_containers['SELECTED_STEREO_PAIRS'].value[pi]
        self.output_containers['STEREOPAIR'].value = fileid
        self.output_containers['STEREOPAIR'].data_direction = EDataDirection.Input
        self.environment_containers['PAIRINDEX'].value += 1

    def fitness_distance(self, fitness):
        return np.math.sqrt((fitness[0] * fitness[0]) + (fitness[1] * fitness[1]))

    def fitness_minimum(self, fitness):
        return min(fitness[0], fitness[1])

    def add_breeder(self, chromosome, fitness):
        if len(self.breeding_pool) >= self.control_containers['POPULATIONLIMIT'].value:
            self.breeding_pool[0] = (chromosome, fitness)
        else:
            self.breeding_pool.append((chromosome, fitness))
        self.breeding_pool.sort(key=lambda x: self.fitness_distance(x[1]))
        if len(self.breeding_pool) >= self.control_containers['POPULATIONLIMIT'].value:
            self.minimum_fitness = self.fitness_minimum(self.breeding_pool[0][1])
            print('Required: {}'.format(self.minimum_fitness))
        else:
            worstfit = self.fitness_minimum(self.breeding_pool[-1][1]) / 2
            if worstfit > self.minimum_fitness:
                self.minimum_fitness = worstfit
                print('Required: {}'.format(self.minimum_fitness))
            else:
                print('+', end='', flush=True)

    def update_parent_chromosomes(self):
        if len(self.breeding_pool) < self.control_containers['POPULATIONLIMIT'].value:
            self.parent_chromosomes = [random.sample(self.stereo_pairs, self.control_containers['NUMPAIRS'].value),
                                       random.sample(self.stereo_pairs, self.control_containers['NUMPAIRS'].value)]
        else:
            mc = random.sample(self.breeding_pool, 2)
            self.parent_chromosomes = [mc[0][0], mc[1][0]]
        self.wins = 0
        self.losses = 0

    def generate_child(self, cossover_perturb_ratio, mutation_ratio):
        p1 = self.parent_chromosomes[0]
        p2 = self.parent_chromosomes[1]
        c = [None] * len(p1)

        # seed the child with a set of direct crossovers
        for i in range(0, len(p1)):
            if random.random() <= 0.5:
                c[i] = p1[i]
            else:
                c[i] = p2[i]

        # handle the perturbed crossovers
        npc = int(len(p1) * cossover_perturb_ratio)
        cpl1 = random.sample(range(0, len(p1)), npc)
        cpl2 = random.sample(range(0, len(p2)), npc)

        for i1, i2 in zip(cpl1, cpl2):
            if random.random() <= 0.5:
                c[i1] = p1[i2]
            else:
                c[i1] = p2[i2]
        # handle gene duplications
        dupes = [n for n, x in enumerate(c) if x in c[:n]]
        for i in dupes:
            c[i] = random.sample(self.stereo_pairs, 1)[0]
        # handle the mutuations if none were required for gene duplication
        if len(dupes) == 0:
            for i in range(0, len(p1)):
                if random.random() <= mutation_ratio:
                    c[i] = random.sample(self.stereo_pairs, 1)[0]
        return c

    def bailout(self, isSuccess):
        print('.', end='', flush=True)
        self.environment_containers['SELECTED_STEREO_PAIRS'].value = None

        self.num_passes -= 1
        if self.num_passes <= 0:
            self.host_process.completed = True
        if isSuccess:
            self.wins += 1
        else:
            self.losses += 1
        gc = self.wins + self.losses
        if gc > self.min_viable_population / 4 and (self.losses / gc) > 0.677:
            print('regenerating')
            self.update_parent_chromosomes()

    def cycle_completed(self, chromosome):
        success = False
        if self.left_fitness >= self.minimum_fitness and self.right_fitness >= self.minimum_fitness:
            self.add_breeder(chromosome, (self.left_fitness, self.right_fitness))
            self.log_data(chromosome)
            self.update_parent_chromosomes()
            success = True
        self.bailout(success)

    def log_data(self, chromosome):
        id = uuid.uuid4().hex
        self.log_data_item(id, chromosome, self.left_intrinsics, 0)
        self.log_data_item(id, chromosome, self.right_intrinsics, 1)

    def log_data_item(self, identifier, chromosome, intrinsics, camera_index):
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, np.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = identifier
        imager = self.host_process.locals['IMAGER'].value
        translated['DATASET'] = chromosome
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = camera_index
        translated['MATCHCOUNT'] = self.host_process.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.host_process.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        sc = self.host_process.get_container('IMAGER').value
        # sc.controller.publish_message(sc.imager_address, "intrinsics", cfg)

        filename = sc.calibration_filename()
        file = open(filename, "w")
        file.write(cfg)
        file.close()

        filename = "../CalibrationRecords/GeneticIntrinsicsHistory.txt"
        if os.path.isfile(filename):
            file = open(filename, "a")
            file.write(",\n")
        else:
            file = open(filename, "w")

        file.write(cfg)
        file.close()


class RandomSolutionSearchStage(IDEFStage):
    """
    Provides a stream of differential images from the input
    """

    def __init__(self, host, name):
        super().__init__(host, name)
        self.num_passes = 1000
        self.stereo_pairs = None
        self.training_dictionary = None

    def get_required_containers(self):
        " Need an input image (RAWIMAGE), a last image to compute difference from (LASTIMAGE),\
        and an output image (DIFFERENTIALIMAGE)"
        container_list = [IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "SELECTED_STEREO_PAIRS"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "PAIRINDEX"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "NUMPAIRS"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "INGESTMODE"),
                          IDEFStageBinding(EConnection.Output, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "STEREOPAIR")]
        return container_list

    def initialize_container_impedance(self):
        self.environment_containers['PAIRINDEX'].value = 9999
        self.environment_containers['PAIRINDEX'].data_direction = EDataDirection.Input

    def is_ready_to_run(self):
        return self.is_container_valid(EConnection.Output, 'STEREOPAIR', EDataDirection.Output)

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'STEREOPAIR', EDataDirection.Input)

    def process(self):
        if self.stereo_pairs is None:
            self.training_dictionary = IDEFProcess.extract_training_dictionary()
            self.stereo_pairs = IDEFProcess.extract_training_stereo_pairs(self.training_dictionary)
        if self.environment_containers['SELECTED_STEREO_PAIRS'].value is None:
            spi = random.sample(self.stereo_pairs, self.control_containers['NUMPAIRS'].value)
            self.environment_containers['SELECTED_STEREO_PAIRS'].value = spi
            self.environment_containers['PAIRINDEX'].value = 0

        pi = self.environment_containers['PAIRINDEX'].value
        if pi >= self.control_containers['NUMPAIRS'].value:
            " cycle completed"
            self.cycle_completed(self.environment_containers['SELECTED_STEREO_PAIRS'].value)
            self.environment_containers['SELECTED_STEREO_PAIRS'].value = None
            self.num_passes -= 1
            self.host_process.stages['CalibrationStage'].initialize_container_impedance()
            self.host_process.stages['DistortionCalculatorStage'].initialize_container_impedance()
            if self.num_passes <= 0:
                self.host_process.completed = True
            return

        fileid = self.environment_containers['SELECTED_STEREO_PAIRS'].value[pi]
        self.output_containers['STEREOPAIR'].value = fileid
        self.output_containers['STEREOPAIR'].data_direction = EDataDirection.Input
        self.environment_containers['PAIRINDEX'].value += 1

    def cycle_completed(self, ssi):
        ds = self.host_process.stages['DistortionCalculatorStage']
        intrinsics = ds.output_containers['CAMERAINTRINSICS'].value  # type: ScalarContainer
        reperr = ds.output_containers['REPROJECTIONERROR']  # type: ScalarContainer
        self.host_process.status_message("Reprojection error is {}".format(reperr.value))
        translated = {}
        for n, v in intrinsics.items():
            if isinstance(v, np.ndarray):
                translated[n] = IDEFProcess.serialize_matrix_to_json(v)
            else:
                translated[n] = v;
        translated['TIMESTAMP'] = datetime.datetime.now().isoformat()
        translated['ID'] = uuid.uuid4().hex
        imager = self.host_process.locals['IMAGER'].value
        translated['DATASET'] = ssi
        translated['CONTROLLER'] = imager.controller.resource
        translated['CAMERAINDEX'] = imager.imager_address
        translated['MATCHCOUNT'] = self.host_process.locals['MATCHCOUNT'].value
        translated['MATCHSEPARATION'] = self.host_process.locals['MATCHSEPARATION'].value
        cfg = json.dumps(translated, ensure_ascii=False)
        sc = self.host_process.get_container('IMAGER').value
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


class StoredImageInterfaceStage(IDEFStage):
    """
    Given a valid image key and ingestion mode, this will deliver an image to downstream processing units. Its not
    interested in what those downstream units do with the images or where the upstream images came from

    The image is identified by its file uuid and must be a stereo pair, i.e. there is a left and right image associated
    with that file id
    """

    def __init__(self, host, name):
        super().__init__(host, name)

    def get_required_containers(self):
        " Need an input image (RAWIMAGE), a last image to compute difference from (LASTIMAGE),\
        and an output image (DIFFERENTIALIMAGE)"
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGE"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGELEFT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Input, "RAWIMAGERIGHT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "STEREOPAIR"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "LEFTIMAGER"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "RIGHTIMAGER"),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "IMAGER"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "NUMPAIRS"),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, "INGESTMODE")]
        return container_list

    def initialize_container_impedance(self):
        self.input_containers['STEREOPAIR'].value = None
        self.input_containers['STEREOPAIR'].data_direction = EDataDirection.Output

        self.input_containers['RAWIMAGELEFT'].matrix = None
        self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Output

        self.input_containers['RAWIMAGERIGHT'].matrix = None
        self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Output

    def is_ready_to_run(self):
        if not self.is_container_valid(EConnection.Input, 'STEREOPAIR', EDataDirection.Input):
            return False
        im = self.control_containers['INGESTMODE'].value
        if im == 'S':
            return (self.is_container_valid(EConnection.Input, 'RAWIMAGELEFT', EDataDirection.Output) and
                    self.is_container_valid(EConnection.Input, 'RAWIMAGERIGHT', EDataDirection.Output))
        elif im == 'L':
            return self.is_container_valid(EConnection.Input, 'RAWIMAGE', EDataDirection.Output)
        elif im == 'R':
            return self.is_container_valid(EConnection.Input, 'RAWIMAGE', EDataDirection.Output)

        return False

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'DIFFERENTIALIMAGE', EDataDirection.Input)

    def process(self):
        fileid = self.input_containers['STEREOPAIR'].value
        im = self.control_containers['INGESTMODE'].value

        if im == 'S' or im == 'L':
            leftimagematrix = IDEFProcess.fetch_stereo_image_matrix(fileid, "unknownctlr", 'L')
        if im == 'S' or im == 'R':
            rightimagematrix = IDEFProcess.fetch_stereo_image_matrix(fileid, "unknownctlr", 'R')

        if im == 'S':
            self.input_containers['RAWIMAGELEFT'].matrix = leftimagematrix
            self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Input
            self.input_containers['RAWIMAGERIGHT'].matrix = rightimagematrix
            self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Input
        elif im == 'L':
            self.input_containers['RAWIMAGE'].matrix = leftimagematrix
            self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Input
        elif im == 'R':
            self.input_containers['RAWIMAGE'].matrix = rightimagematrix
            self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Input

        self.input_containers['STEREOPAIR'].data_direction = EDataDirection.Output


class ImageInjector(IDEFStage):
    def __init__(self, host, name, channel):
        super().__init__(host, name)
        self.channel = channel

    def get_required_containers(self):
        " Need an input image (RAWIMAGE), a last image to compute difference from (LASTIMAGE),\
        and an output image (DIFFERENTIALIMAGE)"
        container_list = [IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, "RAWIMAGELEFT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.MATRIXCONTAINER,
                                           EDataDirection.Output, "RAWIMAGERIGHT"),
                          IDEFStageBinding(EConnection.Input, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Output, "IMAGEKEY")
                          ]
        return container_list

    def initialize_container_impedance(self):
        self.input_containers['IMAGEKEY'].value = None
        self.input_containers['IMAGEKEY'].data_direction = EDataDirection.Output

        self.input_containers['RAWIMAGELEFT'].matrix = None
        self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Output

        self.input_containers['RAWIMAGERIGHT'].matrix = None
        self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Output

    def is_ready_to_run(self):
        return (self.is_container_valid(EConnection.Input, 'IMAGEKEY', EDataDirection.Output)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGELEFT', EDataDirection.Output)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGERIGHT', EDataDirection.Output))

    def is_output_ready(self):
        return (self.is_container_valid(EConnection.Input, 'IMAGEKEY', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGELEFT', EDataDirection.Input)
                and self.is_container_valid(EConnection.Input, 'RAWIMAGERIGHT', EDataDirection.Input))

    def process(self):
        if self.input_containers['IMAGEKEY'].value == Controller.GatedKey:
            print('delay')
            IDEFProcess.DataReady.clear()
            IDEFProcess.DataReady.wait()
        imagedata = Controller.Controllers[0].get_stereo_image_data(self.channel)
        self.input_containers['IMAGEKEY'].value = imagedata[0]
        self.input_containers['IMAGEKEY'].data_direction = EDataDirection.Input
        self.input_containers['RAWIMAGELEFT'].matrix = imagedata[1][1]
        self.input_containers['RAWIMAGELEFT'].data_direction = EDataDirection.Input
        self.input_containers['RAWIMAGERIGHT'].matrix = imagedata[2][1]
        self.input_containers['RAWIMAGERIGHT'].data_direction = EDataDirection.Input
