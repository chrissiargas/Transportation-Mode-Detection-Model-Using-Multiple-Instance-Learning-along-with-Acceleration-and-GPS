

# def free_gpu_cache():
#
#
#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)
#
#
# free_gpu_cache()

import tensorflow as tf

import os
import pprint
import shutil
import time



logdir = os.path.join("results","results_" + time.strftime("%Y%m%d-%H%M%S") + ".txt")



class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(logdir, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


import random
import warnings
import os

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



import numpy as np
import simCLR

SEED = 0



def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


from dataset import SignalsDataset
from MILclassifier import MIL_fit
import sys
import ruamel.yaml



def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:

            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)



def main(logger = False,
         regenerate = True,
         all_users = True,
         randomness = True):
    
    if not randomness:
        set_global_determinism(seed=SEED)


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


    if logger:
        sys.stdout = Logger()

    if all_users:
        for j in range(5):
            for i,test_user in enumerate([1,2,3]):

                print('USER: ' + str(test_user))

                config_edit('train_args', 'test_user', test_user)
                if i == 0 and j == 0 and regenerate:
                    regenerate = True

                else:
                    regenerate = False

                SD = SignalsDataset(regenerate=regenerate)

                if logger:
                    pprint.pprint(SD.shl_args.data_args)
                    print()
                    pprint.pprint(SD.shl_args.train_args)
                    print()

                MIL_fit(SD,
                        summary=True,
                        verbose=SD.shl_args.train_args['verbose'],
                        evaluation=False,
                        load=False)

    else:

        SD = SignalsDataset(regenerate=regenerate)

        if logger:
            pprint.pprint(SD.shl_args.data_args)
            print()
            pprint.pprint(SD.shl_args.train_args)
            print()

        MIL_fit(SD,
                summary=True,
                verbose=SD.shl_args.train_args['verbose'],
                evaluation=False,
                load=False)

        # for _ in range(2):
        #     for specto_augment in [[],['frequencyMask', 'timeMask']]:
        #
        #         config_edit('train_args', 'specto_augment', specto_augment)
        #
        #         SD = SignalsDataset(regenerate=False)
        #         regenerate = False
        #
        #         if logger:
        #             pprint.pprint(SD.shl_args.data_args)
        #             print()
        #             pprint.pprint(SD.shl_args.train_args)
        #             print()
        #
        #         MIL_fit(SD,
        #                 summary=True,
        #                 verbose=SD.shl_args.train_args['verbose'])
        #
        #         del SD
        #

        # for _ in range(2):
        #     for transfer_learning in [['train','none'], ['train','train']]:
        #
        #
        #         config_edit('train_args', 'transfer_learning_acc', transfer_learning[0])
        #         config_edit('train_args', 'transfer_learning_loc', transfer_learning[1])
        #         config_edit('train_args', 'heads', 1)
        #
        #         SD = SignalsDataset(regenerate=False)
        #
        #         if logger:
        #             pprint.pprint(SD.shl_args.data_args)
        #             print()
        #             pprint.pprint(SD.shl_args.train_args)
        #             print()
        #
        #         MIL_fit(SD,
        #                 summary=True,
        #                 verbose=SD.shl_args.train_args['verbose'])
        #
        #         del SD

        # for _ in range(2):
        #     for seperate_MIL in [True,False]:
        #         for transfer_learning in [['train','none']]:
        #
        #
        #             config_edit('train_args', 'transfer_learning_acc', transfer_learning[0])
        #             config_edit('train_args', 'transfer_learning_loc', transfer_learning[1])
        #             config_edit('train_args', 'heads', 1)
        #             config_edit('train_args', 'seperate_MIL', seperate_MIL)


        #
        # for test_user in [1, 2, 3]:
        #     for GPSNet in ['LSTM', 'FCLSTM']:
        #         for noise_std_factor in [0.3, 0.6, 0.9,1.2]:
        #
        #             print(noise_std_factor)
        #             print(GPSNet)
        #             print(test_user)
        #
        #             config_edit('train_args', 'pair_threshold', 60000)
        #             config_edit('train_args', 'noise_std_factor', noise_std_factor)
        #             config_edit('train_args', 'GPSNet', GPSNet)
        #             config_edit('train_args', 'test_user', test_user)
        #             SD = SignalsDataset(regenerate=False)
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD
        #
        # for test_user in [1, 2, 3]:
        #     for point_features in [[],['Velocity'],
        #                             ['Acceleration'],
        #                             ['BearingRate'],
        #                             ['Movability'],
        #                             ['Accuracy'],
        #                             ['Jerk']]:
        #         for _ in range(3):
        #             print(point_features)
        #             print(test_user)
        #
        #             config_edit('train_args', 'pair_threshold', 60000)
        #             config_edit('train_args', 'noise_std_factor', 1.0)
        #             config_edit('train_args', 'GPSNet', 'LSTM')
        #             config_edit('train_args', 'test_user', test_user)
        #             config_edit('train_args', 'point_features', point_features)
        #             SD = SignalsDataset(regenerate=False)
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD

        # SD = SignalsDataset(regenerate=False)
        #
        # if logger:
        #     pprint.pprint(SD.shl_args.data_args)
        #     print()
        #     pprint.pprint(SD.shl_args.train_args)
        #     print()
        #
        # MIL_fit(SD,
        #         summary=True,
        #         verbose=SD.shl_args.train_args['verbose'])
        #
        # del SD

        # SD = SignalsDataset(regenerate=False)
        #
        # if logger:
        #     pprint.pprint(SD.shl_args.data_args)
        #     print()
        #     pprint.pprint(SD.shl_args.train_args)
        #     print()
        #
        # MIL_fit(SD,
        #         summary=True,
        #         verbose=SD.shl_args.train_args['verbose'])
        #
        # del SD

        # for heads in [2,3]:
        #     for seperate_MIL in [False, True]:
        #         for _ in range(2):
        #
        #             config_edit('train_args', 'seperate_MIL', seperate_MIL)
        #             config_edit('train_args', 'heads', heads)
        #
        #             SD = SignalsDataset(regenerate=False)
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD
        #
        # config_edit('train_args', 'heads', 1)
        #
        # config_edit('train_args','accDuration',1200)
        # config_edit('data_args','accDuration',1200)
        #
        # config_edit('train_args','accBagStride',800)
        # config_edit('data_args','accBagStride',800)
        #
        # config_edit('train_args','specto_window',5)
        # config_edit('train_args','specto_overlap',4.9)
        #
        # regenerate = True
        #
        # for transfer_learning in [['train','train']]:
        #
        #     config_edit('train_args', 'transfer_learning_acc', transfer_learning[0])
        #     config_edit('train_args', 'transfer_learning_loc', transfer_learning[1])
        #
        #     SD = SignalsDataset(regenerate=regenerate)
        #     regenerate = False
        #
        #     if logger:
        #         pprint.pprint(SD.shl_args.data_args)
        #         print()
        #         pprint.pprint(SD.shl_args.train_args)
        #         print()
        #
        #     MIL_fit(SD,
        #             summary=True,
        #             verbose=SD.shl_args.train_args['verbose'])
        #
        #     del SD
        #
        # config_edit('train_args', 'accDuration', 1800)
        # config_edit('data_args', 'accDuration', 1800)
        #
        # config_edit('train_args', 'accBagStride', 800)
        # config_edit('data_args', 'accBagStride', 800)
        #
        # config_edit('train_args', 'specto_window', 10)
        # config_edit('train_args', 'specto_overlap', 9.5)
        #
        # regenerate = True
        # for transfer_learning in [['train', 'none'], ['train','train']]:
        #
        #     config_edit('train_args', 'transfer_learning_acc', transfer_learning[0])
        #     config_edit('train_args', 'transfer_learning_loc', transfer_learning[1])
        #
        #     SD = SignalsDataset(regenerate=regenerate)
        #     regenerate = False
        #
        #     if logger:
        #         pprint.pprint(SD.shl_args.data_args)
        #         print()
        #         pprint.pprint(SD.shl_args.train_args)
        #         print()
        #
        #     MIL_fit(SD,
        #             summary=True,
        #             verbose=SD.shl_args.train_args['verbose'])
        #
        #     del SD

        # SD = SignalsDataset(regenerate=False)
        #
        # if logger:
        #     pprint.pprint(SD.shl_args.data_args)
        #     print()
        #     pprint.pprint(SD.shl_args.train_args)
        #     print()
        #
        # MIL_fit(SD,
        #         summary=True,
        #         verbose=SD.shl_args.train_args['verbose'])
        #
        # del SD

        # period = 0.1
        # size = int(90 / period)
        # stride = int(30 / period)
        # bagStride = int(40 / period)
        #
        #
        # config_edit('data_args', 'accDuration', size)
        # config_edit('train_args', 'accDuration', size)
        #
        # config_edit('data_args', 'accStride', stride)
        #
        # config_edit('data_args', 'accBagStride', bagStride)
        # config_edit('train_args', 'accBagStride', bagStride)
        #
        # config_edit('data_args', 'smpl_acc_period', period)
        #
        # regenerate = True
        # for transfer_learning in [['train', 'none'], ['train', 'train']]:
        #
        #     config_edit('train_args', 'transfer_learning_acc', transfer_learning[0])
        #     config_edit('train_args', 'transfer_learning_loc', transfer_learning[1])
        #
        #     SD = SignalsDataset(regenerate=regenerate)
        #     regenerate = False
        #
        #     if logger:
        #         pprint.pprint(SD.shl_args.data_args)
        #         print()
        #         pprint.pprint(SD.shl_args.train_args)
        #         print()
        #
        #     MIL_fit(SD,
        #             summary=True,
        #             verbose=SD.shl_args.train_args['verbose'])
        #
        #     del SD

        # #
        # for heads in [1,2,3]:
        #     for transfer_learning_acc in ['none','train']:
        #         for transfer_learning_loc in ['none','train']:
        #
        #
        #             config_edit('train_args', 'transfer_learning_acc', transfer_learning_acc)
        #             config_edit('train_args', 'transfer_learning_loc', transfer_learning_loc)
        #             config_edit('train_args', 'heads', heads)
        #
        #             SD = SignalsDataset(regenerate=False)
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD
        #
        # regenerate = False
        # for mask_num in [1,2,3]:
        #     for mask_param in [3,6,9]:
        #         for _ in range(2):
        #             config_edit('train_args', 'frequency_masking_param', mask_param)
        #             config_edit('train_args', 'frequency_mask_num', mask_num)
        #             config_edit('train_args', 'time_masking_param', mask_param)
        #             config_edit('train_args', 'time_mask_num', mask_num)
        #
        #             SD = SignalsDataset(regenerate=regenerate)
        #             regenerate = False
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD
        #
        # config_edit('train_args', 'frequency_masking_param', 6)
        # config_edit('train_args', 'frequency_mask_num', 2)
        # config_edit('train_args', 'time_masking_param', 6)
        # config_edit('train_args', 'time_mask_num', 2)

        # for acc_xyz_augmentation in [['TimeWarp']]:
        #     for acc_xyz_aug_params in [[0.1],[0.25],[0.5],[1.0]]:
        #         config_edit('train_args', 'acc_xyz_augmentation', acc_xyz_augmentation)
        #         config_edit('train_args', 'acc_xyz_aug_params', acc_xyz_aug_params)
        #
        #         SD = SignalsDataset(regenerate=False)
        #
        #         if logger:
        #             pprint.pprint(SD.shl_args.data_args)
        #             print()
        #             pprint.pprint(SD.shl_args.train_args)
        #             print()
        #
        #         MIL_fit(SD,
        #                 summary=True,
        #                 verbose=SD.shl_args.train_args['verbose'])
        #
        #         del SD

        # for acc_xyz_augmentation in [['Permutation']]:
        #     for acc_xyz_aug_params in [[2], [3], [4], [5]]:
        #         config_edit('train_args', 'acc_xyz_augmentation', acc_xyz_augmentation)
        #         config_edit('train_args', 'acc_xyz_aug_params', acc_xyz_aug_params)
        #
        #         SD = SignalsDataset(regenerate=False)
        #
        #         if logger:
        #             pprint.pprint(SD.shl_args.data_args)
        #             print()
        #             pprint.pprint(SD.shl_args.train_args)
        #             print()
        #
        #         MIL_fit(SD,
        #                 summary=True,
        #                 verbose=SD.shl_args.train_args['verbose'])
        #
        #         del SD




        # for period in [0.1]:
        #     regenerate = True
        #     for specto_window in [5.0,10.0,20.0]:
        #         for specto_stride in [0.1,0.5,1.0,2.0]:
        #             size = int(90 / period)
        #             stride = int(30 / period)
        #             specto_overlap = specto_window - specto_stride
        #
        #             config_edit('data_args', 'accDuration', size)
        #             config_edit('train_args', 'accDuration', size)
        #
        #             config_edit('data_args', 'accStride', stride)
        #
        #             config_edit('data_args', 'smpl_acc_period', period)
        #
        #             config_edit('train_args', 'specto_overlap', specto_overlap)
        #             config_edit('train_args', 'specto_window', specto_window)
        #
        #
        #             SD = SignalsDataset(regenerate=regenerate)
        #             regenerate = False
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD
        #

        #
        # for period in [0.2,0.1,0.05,0.04,0.03]:
        #     for duration in [30,60,90]:
        #         for i,specto_overlap in enumerate([4.5, 4.9]):
        #             size = int(duration / period)
        #             stride = int(30 / period)
        #
        #
        #             config_edit('data_args', 'accDuration', size)
        #             config_edit('train_args', 'accDuration', size)
        #
        #             config_edit('data_args', 'accStride', stride)
        #
        #             config_edit('data_args', 'smpl_acc_period', period)
        #
        #             config_edit('train_args', 'specto_overlap', specto_overlap)
        #
        #             regenerate = True if i==0 else False
        #             SD = SignalsDataset(regenerate=regenerate)
        #
        #             if logger:
        #                 pprint.pprint(SD.shl_args.data_args)
        #                 print()
        #                 pprint.pprint(SD.shl_args.train_args)
        #                 print()
        #
        #             MIL_fit(SD,
        #                     summary=True,
        #                     verbose=SD.shl_args.train_args['verbose'])
        #
        #             del SD
        #

if __name__ == "__main__":
    main()





