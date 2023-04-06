import argparse

import yaml


class dataParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="filtered data args"
        )

    def create(self, path_config):
        self.parser.add_argument(
            '--config',
            default=path_config,
            help='config file location'
        )

        self.parser.add_argument(
            '--shapes',
            default=dict(),
            type=dict,
            help='shapes'
        )

    def __call__(self, path_config):
        self.create(path_config)

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