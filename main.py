import pprint
import time
from dataset import Dataset
from TMD import TMD_MIL
import sys
import ruamel.yaml
import warnings
import os
from evaluation import evaluate

logdir = os.path.join("results","results_" + time.strftime("%Y%m%d-%H%M%S") + ".txt")
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def main(logger = False, regenerate = False, all_users = False, evaluation = False):
    repeat = 4

    if logger:
        sys.stdout = Logger()

    if all_users:
        for j in range(repeat):
            for i, test_user in enumerate([1,2,3]):

                print('USER: ' + str(test_user))

                config_edit('train_args', 'test_user', test_user)

                regenerate = regenerate if i == j == 0 else False
                SD = Dataset(regenerate=regenerate)

                if logger:
                    pprint.pprint(SD.shl_args.data_args)
                    print()
                    pprint.pprint(SD.shl_args.train_args)
                    print()

                if evaluation:
                    evaluate(SD, verbose=SD.shl_args.train_args['verbose'])
                else:
                    TMD_MIL(SD, summary=True, verbose=SD.shl_args.train_args['verbose'])

    else:
        for j in range(repeat):
            regenerate = regenerate if j == 0 else False
            SD = Dataset(regenerate=regenerate)

            if logger:
                pprint.pprint(SD.shl_args.data_args)
                print()
                pprint.pprint(SD.shl_args.train_args)
                print()

            if evaluation:
                evaluate(SD, verbose=SD.shl_args.train_args['verbose'])
            else:
                TMD_MIL(SD, summary=True, verbose=SD.shl_args.train_args['verbose'])


if __name__ == "__main__":
    main()





