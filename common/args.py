import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images')

        # Data parameters
        self.add_argument('--data-path', type=pathlib.Path,
                          default='/home/aditomer/Datasets',help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')
        # Override defaults with passed overrides
        self.set_defaults(**overrides)
