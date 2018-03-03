import json
import math
import os

import numpy as np


class StereoTrainingChromosome:

    def __init__(self, id):
        self.id = id
        self.chromosome = []
        self.left_camera_matrix = None
        self.left_distortion_coefficients = None
        self.left_reprojerr = 0
        self.right_camera_matrix = None
        self.right_distortion_coefficients = None
        self.right_reprojerr = 0

    def combined_reprojection_error(self):
        return math.sqrt((self.left_reprojerr ** 2) + (self.right_reprojerr ** 2))

    def left_fitness(self):
        return 1.0 - self.left_reprojerr

    def right_fitness(self):
        return 1.0 - self.right_reprojerr


class StereoTrainingSet:
    @staticmethod
    def flatten_matrix(m):
        cm = []
        cm.append(m.shape[0])
        cm.append(m.shape[1])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                cm.append(m.item((i, j)))
        return cm

    @staticmethod
    def resurrect_matrix(dv):
        X = np.empty(shape=[dv[0], dv[1]])
        dvi = 2
        for i in range(dv[0]):
            for j in range(dv[1]):
                X[i, j] = dv[dvi]
                dvi += 1
        return X

    @staticmethod
    def training_file(isMono):
        if isMono:
            return '../CalibrationRecords/GeneticIntrinsicsHistory.txt'
        else:
            return '../CalibrationRecords/GeneticIntrinsicsHistory.txt'

    @staticmethod
    def data_ready(isMono):
        fn = StereoTrainingSet.training_file(isMono)
        return os.path.isfile(fn)

    def __init__(self, isMono):
        self.mono_mode = isMono
        self.training_data = None

    def load_training_results(self):
        training_set = {}
        trainingfile = StereoTrainingSet.training_file(self.mono_mode)
        with open(trainingfile, 'r') as myfile:
            data = '[' + myfile.read() + ']'
        jd = json.loads(data)

        # load into two arrays, one for each camera, sorted by ascending error
        for x in jd:
            id = x['ID']
            if id not in training_set:
                d = StereoTrainingChromosome(id)
                for td in x['DATASET']:
                    d.chromosome.append(td)
                training_set[id] = d
            d = training_set[id]
            if x['CAMERAINDEX'] == 0:
                if d.left_camera_matrix is not None:
                    raise ValueError("Something wrong with the left training data")
                d.left_camera_matrix = x['CAMERAMATRIX']
                d.left_distortion_coefficients = x['DISTORTIONCOEFFICIENTS']
                d.left_reprojerr = x['REPROJECTIONERROR']
                for i, td in enumerate(x['DATASET']):
                    if d.chromosome[i] != td:
                        raise ValueError('Left hromosome mismatch error')
            else:
                if d.right_camera_matrix is not None:
                    raise ValueError("Something wrong with the right training data")
                d.right_camera_matrix = x['CAMERAMATRIX']
                d.right_distortion_coefficients = x['DISTORTIONCOEFFICIENTS']
                d.right_reprojerr = x['REPROJECTIONERROR']
                for i, td in enumerate(x['DATASET']):
                    if d.chromosome[i] != td:
                        raise ValueError('Right hromosome mismatch error')
        # order by inreasing 2d reprojection error
        vals = list(training_set.values())
        vals.sort(key=lambda a: a.combined_reprojection_error())
        self.training_data = vals

    def extract_population(self, popsize):
        result = []
        for i in range(0, min(popsize, len(self.training_data))):
            v = (self.training_data[i].left_fitness(), self.training_data[i].right_fitness(),
                 self.training_data[i].chromosome)
            result.append(self.training_data[i])
        return result

    def load_intrinsics_data(self):
        with open('../CalibrationRecords/GeneticIntrinsicsHistory.txt', 'r') as myfile:
            data = '[' + myfile.read() + ']'
        jd = json.loads(data)

        # load into two arrays, one for each camera, sorted by ascending error
        self.calibrationdata = [[], []]
        for x in jd:
            if x['CAMERAINDEX'] == 0:
                self.calibrationdata[0].append(x)
            else:
                self.calibrationdata[1].append(x)

    def pivot_image_value(self):
        self.calibrationdata[0].sort(key=lambda x: x['REPROJECTIONERROR'])
        self.calibrationdata[1].sort(key=lambda x: x['REPROJECTIONERROR'])
        # compute cross correlation of selected images
        l0 = len(self.calibrationdata[0])
        l1 = len(self.calibrationdata[1])
        for i in range(0, min(l0, l1)):
            ds1 = self.calibrationdata[0][i]['DATASET']
            ds2 = self.calibrationdata[1][i]['DATASET']
            ordscore = 0
            setscore = 0
            for j in range(0, len(ds1)):
                if ds1[j] == ds2[j]:
                    ordscore += 1
                if ds2[j] in ds1:
                    setscore += 1
            print('ordinal score[{}] = {}, set score[{}] = {}'.format(i, ordscore / len(ds1), i, setscore / len(ds1)))

    def create_image_scores(self):
        with open('../CalibrationRecords/intrinsicsHistory.txt', 'r') as myfile:
            data = '[' + myfile.read() + ']'
        jd = json.loads(data)
        scoredict = {}
        countdict = {}
        for x in jd:
            re = x['REPROJECTIONERROR']
            if 'DATASET' in x:
                for iid in x['DATASET']:
                    if iid in scoredict:
                        v = scoredict[iid] + re
                        c = countdict[iid] + 1
                        countdict[iid] = c
                        scoredict[iid] = v
                    else:
                        scoredict[iid] = re
                        countdict[iid] = 1
        with open('../CalibrationRecords/imagescores.csv', 'w') as myfile:
            for key, value in scoredict.items():
                count = countdict[key]
                score = scoredict[key] / count
                pi = '{},{}\n'.format(key, score)
                myfile.write(pi)
