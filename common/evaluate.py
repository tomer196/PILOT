"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import pathlib
from argparse import ArgumentParser
import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
import sys
sys.path.insert(0,'/home/tomerweiss/PILOT')

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())
#
# def psnr1(gt, pred):
#     """ Compute Peak Signal to Noise Ratio metric (PSNR) """
#     gt = gt - gt.min()
#     gt = gt / gt.max()
#
#     pred = pred - pred.min()
#     pred = pred / pred.max()
#
#     return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )
#
# def ssim1(gt, pred):
#     """ Compute Structural Similarity Index Metric (SSIM). """
#     return compare_ssim(
#         gt, pred, multichannel=True, data_range=gt.max()
#     )

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)

class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )

def evaluate():
    args = create_arg_parser().parse_args()
    args.target_path = f'/home/tomerweiss/Datasets/singlecoil_{args.data_split}'
    args.predictions_path = f'/home/tomerweiss/PILOT/summary/{args.test_name}/rec'
    print('/home/tomerweiss/PILOT/summary/'+args.test_name+'/rec')
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in pathlib.Path(args.target_path).iterdir():
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path +'/'+ tgt_file.name) as recons:
            if args.acquisition and args.acquisition == target.attrs['acquisition']:
                continue
            target = target['reconstruction_esc'].value
            recons = recons['reconstruction'].value
            metrics.push(target, recons)
    return metrics

def create_arg_parser():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--data-split', choices=['val', 'test'], default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--target-path', type=pathlib.Path, default=f'/home/tomerweiss/Datasets/singlecoil_test',
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, default=f'summary/test/rec',
                        help='Path to reconstructions')
    parser.add_argument('--acquisition', choices=['PD', 'PDFS'], default=None,
                        help='If set, only volumes of the specified acquisition type are used '
                             'for evaluation. By default, all volumes are included.')
    return parser
    
if __name__ == '__main__':
    metrics=evaluate()
    print(metrics)
