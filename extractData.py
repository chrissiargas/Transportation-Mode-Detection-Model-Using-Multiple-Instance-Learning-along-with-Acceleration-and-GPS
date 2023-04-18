import shutil

import numpy as np
import os
from configParser import Parser
from dataParser import dataParser

class extractData:
    def __init__(self ,args = None):

        if not args:
            parser = Parser()
            self.shl_args = parser.get_args()

        else:
            self.shl_args = args

        self.path = self.shl_args.data_args['path']
        if self.shl_args.data_args['dataset'] == 'CompleteUser1':
            self.path = os.path.join(
                self.path,
                'completeData'
            )

        self.path_data = os.path.join(
            self.path,
            'filteredData'
        )

        path_config = os.path.join(
            self.path,
            'data_config.yaml'
        )

        if os.path.exists(self.path_data) and os.path.exists(path_config):
            dp = dataParser()
            self.args = dp(path_config)
            print('Found Data')
            self.found = True

        else:
            print('No filteredData folder or config file')
            self.found = False


        if self.shl_args.data_args['dataset'] == 'CompleteUser1':

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

    def __call__(self,
                 delete_dst = False,
                 delete_tmp = False,
                 delete_final = False):


        if not self.found:
            return


        if delete_dst:

            z = os.path.join(self.path , 'dstData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

        if delete_tmp:

            z = os.path.join(self.path , 'tmpFolder')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

        if delete_final:

            z = os.path.join(self.path , 'finalData')

            try:
                shutil.rmtree(z)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))



        location = []
        for position in self.pos:


            loc_mmap_path = os.path.join(
                self.path_data,
                'location_' + position +'.mmap'
            )

            pos_loc = np.memmap(
                filename=loc_mmap_path,
                mode='r+',
                shape=self.get_shape(self.args.shapes['location'][position]),
                dtype=np.float64
            )

            location.append(pos_loc)

        acc_mmap_path = os.path.join(
            self.path_data,
            'acceleration.mmap'
        )

        acceleration = np.memmap(
            filename=acc_mmap_path,
            mode='r+',
            shape=self.get_shape(self.args.shapes['acceleration']),
            dtype=np.float64
        )

        lbs_mmap_path = os.path.join(
            self.path_data,
            'labels.mmap'
        )

        labels = np.memmap(
            filename=lbs_mmap_path,
            mode='r+',
            shape=self.get_shape(self.args.shapes['labels'], is_lbs=True),
            dtype=np.int64
        )

        return acceleration , labels , location


    def take_user_day(self, x, u, d):
        ret = []
        stop = False
        for el in x:

            if el[0, -3] == u and el[0, -2] == d:
                stop = True
                ret.append(el)

            elif stop:
                break

        return np.array(ret)

    def get_shape(self ,x ,is_lbs=False):
        if not is_lbs:
            return (x['samples'] ,x['duration'] ,x['channels'])

        else:
            return (x['samples'] ,x['channels'])