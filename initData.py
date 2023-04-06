import numpy as np
import pandas as pd
import os
import subprocess
from configParser import Parser

NROWS = None


class initData:
    def __init__(self, args=None):

        if not args:
            parser = Parser()
            args = parser.get_args()

        self.args = args

        self.acc = {
            1: 'Acc_x',
            2: 'Acc_y',
            3: 'Acc_z'
        }

        self.loc = {
            1: 'acc',
            2: 'Lat',
            3: 'Long',
            4: 'Alt'
        }

        self.labels = {
            0: 'null',
            1: 'still',
            2: 'walk',
            3: 'run',
            4: 'bike',
            5: 'car',
            6: 'bus',
            7: 'train',
            8: 'subway',
        }

        self.completeData = (self.args.data_args['dataset'] == 'CompleteUser1')

        if self.completeData:

            self.pos = ['Hips']

            self.files = {
                '1': ['010317','010617','020317','020517','020617','030317',
                      '030517','030617','030717','040517','040717','050517',
                      '050617','050717','060317','060617','070317','070617',
                      '080317','080517','080617','090317','090517','090617',
                      '100317','100517','110517','120517','120617','130317',
                      '130617','140317','140617','150317','150517','150617',
                      '160317','170317','170517','190417',
                      '190517','200317','200417','200517','200617','210317',
                      '220317','220517','220617','230317','230517','230617',
                      '240317','240417','240517','250317','250417','250517',
                      '260417','260517','260617','270317','270417','270617',
                      '280317','280417','280617','290317','290517','290617',
                      '300317','300517','300617','310517']
            }

        else:
            self.pos = [
                'Torso',
                'Hips',
                'Bag',
                'Hand'
            ]

            self.files = {
                '1': ['220617', '260617', '270617'],
                '2': ['140617', '140717', '180717'],
                '3': ['030717', '070717', '140617']
            }

        path = self.args.data_args['path']
        if self.completeData:
            path = os.path.join(
                path,
                'completeData'
            )

        srcPath = os.path.join(
            path,
            'srcData'
        )

        path = os.path.join(
            path,
            'dstData'
        )

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.srcPath = srcPath

    def __call__(self, verbose=False, *args, **kwargs):
        self.verbose = verbose

        if self.verbose:
            print(self.files)
            print(self.pos)

        loc, acc = self.extract_features()
        lbs = self.extract_labels()
        return loc, acc, lbs

    def extract_features(self):

        location = {}
        acceleration = {}
        filename = 'SHLDataset_preview_v1'

        self.n_acc = {}
        self.n_loc = {}

        for user, days in self.files.items():
            location[user] = {}
            acceleration[user] = {}

            self.n_acc[user] = {}
            self.n_loc[user] = {}

            for day in days:
                location[user][day] = {}
                acceleration[user][day] = {}

                self.n_acc[user][day] = {}
                self.n_loc[user][day] = {}

                for position in self.pos:

                    self.src_loc = os.path.join(
                        self.srcPath,
                        filename + '_part' + user,
                        filename,
                        'User' + user,
                        day,
                        position + '_Location.txt',
                    )

                    self.src_mot = os.path.join(
                        self.srcPath,
                        filename + '_part' + user,
                        filename,
                        'User' + user,
                        day,
                        position + '_Motion.txt'
                    )

                    stdout = subprocess.Popen(
                        "wc -l < " + self.src_loc,
                        shell=True,
                        stdout=subprocess.PIPE)

                    lines_loc, _ = stdout.communicate()
                    lines_loc = int(lines_loc)
                    self.n_loc[user][day][position] = lines_loc

                    stdout = subprocess.Popen(
                        "wc -l < " + self.src_mot,
                        shell=True,
                        stdout=subprocess.PIPE)

                    lines_mot, _ = stdout.communicate()
                    lines_mot = int(lines_mot)
                    self.n_acc[user][day][position] = lines_mot

                    dst_filename = 'user' + user + '_' + day + '_' + position

                    self.dst_loc = os.path.join(
                        self.path,
                        dst_filename + '_location' + '.mmap'
                    )

                    self.dst_mot = os.path.join(
                        self.path,
                        dst_filename + '_motion' + '.mmap'
                    )

                    if os.path.exists(self.dst_loc) and os.path.exists(self.dst_mot):
                        location[user][day][position], \
                        acceleration[user][day][position] = self.to_mmap_features(
                            shape_loc=(lines_loc, 5),
                            shape_mot=(lines_mot, 4),
                            exists=True
                        )

                        continue

                    location[user][day][position], \
                    acceleration[user][day][position] = self.to_mmap_features(
                        shape_loc=(lines_loc, 5),
                        shape_mot=(lines_mot, 4)
                    )

        return location, acceleration

    def extract_labels(self):
        labels = {}

        filename = 'SHLDataset_preview_v1'

        for user, days in self.files.items():
            labels[user] = {}

            for day in days:

                self.src_lbs = os.path.join(
                    self.srcPath,
                    filename + '_part' + user,
                    filename,
                    'User' + user,
                    day,
                    'Label.txt',
                )

                stdout = subprocess.Popen(
                    "wc -l < " + self.src_lbs,
                    shell=True,
                    stdout=subprocess.PIPE)

                lines_lbs, _ = stdout.communicate()
                lines_lbs = int(lines_lbs)

                dst_filename = 'user' + user + '_' + \
                               day

                self.dst_lbs = os.path.join(
                    self.path,
                    dst_filename + '_labels' + '.mmap'
                )

                if os.path.exists(self.dst_lbs):
                    labels[user][day] = self.to_mmap_labels(
                        shape_lbs=(int(lines_lbs), 2),
                        exists=True
                    )

                    continue

                labels[user][day] = self.to_mmap_labels(
                    shape_lbs=(lines_lbs, 2)
                )

        return labels

    def to_mmap_labels(self, shape_lbs, dtype_lbs=np.int64, exists=False):

        if self.verbose:
            print(self.dst_lbs)

        if exists:
            # print(self.dst_lbs)

            mmap_lbs = np.memmap(
                self.dst_lbs,
                mode='r+',
                shape=shape_lbs,
                dtype=dtype_lbs
            )

            return mmap_lbs

        else:
            mmap_lbs = np.memmap(
                self.dst_lbs,
                mode='w+',
                shape=shape_lbs,
                dtype=dtype_lbs
            )

            offset = 0
            for batch in pd.read_csv(self.src_lbs, delimiter=' ',
                                     chunksize=5000, header=None, nrows=NROWS):
                mmap_lbs[offset: offset + batch.shape[0]] = batch.iloc[:, [0, 1]]
                offset += batch.shape[0]

            return mmap_lbs

    def to_mmap_features(self,
                         shape_loc,
                         shape_mot,
                         dtype_loc=np.float64,
                         dtype_mot=np.float64,
                         exists=False):

        if self.verbose:
            print(self.dst_loc)
            print(self.dst_mot)

        if exists:

            mmap_loc = np.memmap(
                self.dst_loc,
                mode='r+',
                dtype=dtype_loc,
                shape=shape_loc
            )

            mmap_acc = np.memmap(
                self.dst_mot,
                mode='r+',
                dtype=dtype_mot,
                shape=shape_mot
            )

        else:

            mmap_loc = np.memmap(
                self.dst_loc,
                mode='w+',
                dtype=dtype_loc,
                shape=shape_loc
            )

            mmap_acc = np.memmap(
                self.dst_mot,
                mode='w+',
                dtype=dtype_mot,
                shape=shape_mot
            )

            offset = 0
            for batch in pd.read_csv(self.src_loc, delimiter=' ',
                                     chunksize=5000, header=None, nrows=NROWS):
                mmap_loc[offset: offset + batch.shape[0]] = batch.iloc[:, [0, 3, 4, 5, 6]]
                offset += batch.shape[0]

            offset = 0
            for batch in pd.read_csv(self.src_mot, delimiter=' ',
                                     chunksize=5000, header=None, nrows=NROWS):
                mmap_acc[offset: offset + batch.shape[0]] = batch.values[:, [0, 1, 2, 3]]
                offset += batch.shape[0]

        return mmap_loc, mmap_acc
