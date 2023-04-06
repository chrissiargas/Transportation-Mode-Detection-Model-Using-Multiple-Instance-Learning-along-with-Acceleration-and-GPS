import argparse
import os
import yaml

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description = "SHL dataset and model args"
        )

    def __call__(self, *args, **kwargs):
        self.parser.add_argument(
            '--config',
            default = './config.yaml',
            help = 'config file location'
        )

        self.parser.add_argument(
            '--data_args',
            default = dict(),
            type = dict,
            help = 'data preprocess args'
        )

        self.parser.add_argument(
            '--train_args',
            default = dict(),
            type = dict,
            help = 'train data args'
        )


    def get_args(self):
        self.__call__()
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            dfltArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for dfltKey in dfltArgs.keys():
            if dfltKey not in keys:
                print('WRONG ARG: {}'.format(dfltKey))
                assert (dfltKey in keys)

        self.parser.set_defaults(**dfltArgs)

        args = self.parser.parse_args(args=[])

        return args