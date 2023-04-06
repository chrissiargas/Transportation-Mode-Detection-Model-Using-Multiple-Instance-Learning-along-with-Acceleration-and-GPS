
import time
from dataset import Dataset
import ruamel.yaml
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def main(regenerate = False,
         all_users = False):

    if all_users:

        for i,test_user in enumerate([1,2,3]):

            print('USER: ' + str(test_user))
            config_edit('train_args', 'test_user', test_user)

            if i == 0 and regenerate:
                regenerate = True

            else:
                regenerate = False

            SD = Dataset(regenerate=regenerate)

            SD(randomTree=True)

    else:

        SD = Dataset(regenerate=regenerate)

        SD(randomTree=True)


if __name__ == "__main__":
    main()





