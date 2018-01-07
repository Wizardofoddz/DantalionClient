"""
Defines a number of useful predefined stages that can be composed into new processes
"""
import cv2
import numpy as np

from Notification import Notifier
from ProcessAbstractionLayer import IDEFStage, IDEFStageBinding, EConnection, EContainerType, EDataDirection, \
    CameraIntrinsicsContainer


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
                                           EDataDirection.Output, "CURRENTCHESSBOARDCORNERS")]
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
        boardsize = self.control_containers['BOARDSIZE'].value
        corner_container = self.output_containers['CURRENTCHESSBOARDCORNERS']

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, boardsize)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (9, 7), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            corner_container.value = corners2
            # mark the output valid
            self.output_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Input
        else:
            self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Output


class DistortionCalculatorStage(IDEFStage):
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
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'BOARDSIZE'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'MATCHCOUNT'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'MATCHSEPARATION'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'IMAGESIZE'),
                          IDEFStageBinding(EConnection.Control, EContainerType.SCALARCONTAINER,
                                           EDataDirection.Input, 'RECTIFY_ALPHA'),
                          IDEFStageBinding(EConnection.Environment, EContainerType.SCALARCONTAINER,
                                           EDataDirection.InputOutput, 'CHESSBOARDCORNERLIST'),
                          IDEFStageBinding(EConnection.Output, EContainerType.CAMERA_INTRINSICS,
                                           EDataDirection.Output, 'CAMERAINTRINSICS')]
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
                and self.is_container_valid(EConnection.Control, 'IMAGESIZE', EDataDirection.Input)
                and self.is_container_valid(EConnection.Environment, 'CHESSBOARDCORNERLIST', EDataDirection.InputOutput)
                and self.is_container_valid(EConnection.Output, 'CAMERAINTRINSICS', EDataDirection.Output))

    def is_output_ready(self):
        return self.is_container_valid(EConnection.Output, 'CAMERAINTRINSICS', EDataDirection.Input)

    def process(self):
        print('analyse')
        boardsize = self.control_containers['BOARDSIZE'].value
        imagesize = self.control_containers['IMAGESIZE'].value
        mindistance = self.control_containers['MATCHSEPARATION'].value
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
            for corners in cornerlist:
                t_avg = self.average_corner(corners)
                dist = cv2.norm(t_avg - avgpt)
                if dist <= mindistance:
                    reject = True
                    break
            if not reject:
                image = self.input_containers['RAWIMAGE'].matrix
                img = cv2.drawChessboardCorners(image, (9, 7), corner_container.value, True)
                self.input_containers['RAWIMAGE'].matrix = img
                if len(cornerlist) < matchcount:
                    cornerlist.append(corner_container.value)
                    if len(cornerlist) % (matchcount / 4) == 0 and Notifier.activeNotifier is not None:
                        Notifier.activeNotifier.play_soundfile_async('bityes.wav')
                else:
                    if Notifier.activeNotifier is not None:
                        Notifier.activeNotifier.speak_message("Calculating")
                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((boardsize[0] * boardsize[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:boardsize[1], 0:boardsize[0]].T.reshape(-1, 2)
                    oblocs = []
                    for i in range(0, len(cornerlist)):
                        oblocs.append(objp)
                    ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(oblocs, cornerlist,
                                                                     (imagesize[0],
                                                                      imagesize[1]),
                                                                     None, None)
                    if ret:
                        intrinsics = self.output_containers['CAMERAINTRINSICS']  # type: CameraIntrinsicsContainer
                        re = self.compute_reprojection_error(cornerlist, oblocs, k, dist, rvecs, tvecs)
                        print("Reprojection error is : {}".format(re))
                        intrinsics.camera_matrix = k
                        intrinsics.distortion_coefficients = dist
                        intrinsics.rotation_vectors = rvecs
                        intrinsics.translation_vectors = tvecs
                        # output is ready for consumer
                        intrinsics.data_direction = EDataDirection.Input
                        # and we're done
                        self.host_process.completed = True
                        print("intrinsics computed")
                    else:
                        print("could not calculate intrinsics from sample set")
        self.input_containers['RAWIMAGE'].data_direction = EDataDirection.Output
        self.input_containers['CURRENTCHESSBOARDCORNERS'].data_direction = EDataDirection.Output

    def compute_reprojection_error(self, imgpoints, objpoints, k, dist, rvecs, tvecs):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], k, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error / len(objpoints)

    def average_corner(self, corners):
        mean = abs(np.mean(corners, axis=0))
        return mean
