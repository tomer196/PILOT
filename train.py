
import logging
import pathlib
import random
import shutil
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(0,'/home/aditomer/PILOT')

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
from data import transforms
from data.mri_data import SliceData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model
from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp
import scipy.io as sio
from common.utils import get_vel_acc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):

        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        image = transforms.complex_abs(image)
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        kspace = transforms.rfft2_regular(image)
        
        target = transforms.to_tensor(target)
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return kspace, target, mean, std, attrs['norm'].astype(np.float32)

def create_datasets(args):
    train_data = SliceData(
        root=args.data_path / 'singlecoil_train',
        transform=DataTransform(args.resolution),
        sample_rate=args.sample_rate
    )
    dev_data = SliceData(
        root=args.data_path / 'singlecoil_val',
        transform=DataTransform(args.resolution),
        sample_rate=args.sample_rate
    )
    return dev_data, train_data

def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader

def tsp_solver(x):
    # reorder the trajectory according to the TSP solution
    d = distance_matrix(x, x)
    t = solve_tsp(d)
    return x[t, :]

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    if epoch == args.TSP_epoch and args.TSP:
        x = model.get_trajectory()
        x = x.detach().cpu().numpy()
        x = tsp_solver(x)
        v,a = get_vel_acc(x)
        writer.add_figure('TSP_Trajectory', plot_trajectory(x), epoch)
        writer.add_figure('TSP_Acc', plot_acc(a, args.a_max), epoch)
        writer.add_figure('TSP_Vel', plot_acc(v, args.v_max), epoch)
        with torch.no_grad():
            model.subsampling.x.data = torch.tensor(x, device='cuda')

    start_epoch = start_iter = time.perf_counter()
    for iter, data in enumerate(data_loader):
        input, target, mean, std, norm = data
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)

        output = model(input).squeeze(1)

        x = model.get_trajectory()
        v, a = get_vel_acc(x)

        acc_loss = args.acc_weight * torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
        vel_loss = args.vel_weight * torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
        rec_loss = args.rec_weight * F.l1_loss(output, target)
        if args.TSP_epoch:
            loss=rec_loss
        else:
            loss = rec_loss + vel_loss + acc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        #writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'rec_loss: {rec_loss:.4g}, vel_loss: {vel_loss:.4g}, acc_loss: {acc_loss:.4g}'
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch

def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in enumerate(data_loader):
                input, target, mean, std, norm = data
                input = input.unsqueeze(1).to(args.device)
                target = target.to(args.device)
                output = model(input).squeeze(1)

                loss = F.l1_loss(output, target)
                losses.append(loss.item())

            x = model.get_trajectory()
            v, a = get_vel_acc(x)
            acc_loss = args.acc_weight * torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
            vel_loss = args.vel_weight * torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
            rec_loss = np.mean(losses)

            writer.add_scalar('Rec_Loss', rec_loss, epoch)
            writer.add_scalar('Acc_Loss', acc_loss.detach().cpu().numpy(), epoch)
            writer.add_scalar('Vel_Loss', vel_loss.detach().cpu().numpy(), epoch)
            writer.add_scalar('Total_Loss', rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy(), epoch)

        x = model.get_trajectory()
        v, a = get_vel_acc(x)
        if args.TSP and epoch < args.TSP_epoch:
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        else:
            writer.add_figure('Trajectory', plot_trajectory(x.detach().cpu().numpy()),epoch)
        writer.add_figure('Accelerations_plot', plot_acc(a.cpu().numpy(),args.a_max), epoch)
        writer.add_figure('Velocity_plot', plot_acc(v.cpu().numpy(),args.v_max), epoch)
        writer.add_text('Coordinates', str(x.cpu().numpy()).replace(' ',','),epoch)
    return np.mean(losses), time.perf_counter() - start

def plot_scatter(x):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    ax.plot(x[:,0],x[:,1],'.')
    return fig

def plot_trajectory(x):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    ax.plot(x[:,0],x[:,1])
    return fig

def plot_acc(a,a_max=None):
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(a[:,0])
    ax[1].plot(a[:,1])
    if a_max != None:
        limit = np.ones(a.shape[0]) * a_max
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig

def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            save_image(target, 'Target')
            if epoch != 0:
                output = model(input.clone())
                corrupted=model.subsampling(input)

                save_image(output, 'Reconstruction')
                save_image(corrupted, 'Corrupted')
                save_image(torch.abs(target - output), 'Error')
            break

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir + '/model.pt'
    )   
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')

def build_model(args):
    model = Subsampling_Model(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization
    ).to(args.device)
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer

def build_optim(args, model):
    optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(),'lr': args.sub_lr},
                {'params': model.reconstruction_model.parameters()}], args.lr)
    return optimizer

def train():
    args = create_arg_parser().parse_args()
    args.exp_dir=f'summary/{args.test_name}'
    args.checkpoint=f'summary/{args.test_name}/model.pt'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir+'/args.txt', "w") as text_file:
      print(vars(args), file=text_file)

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']+1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    #logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    dev_loss, dev_time = evaluate(args, 0, model, dev_loader, writer)
    visualize(args, 0, model, display_loader, writer)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch+1, model, dev_loader, writer)
        visualize(args, epoch+1, model, display_loader, writer)

        if epoch==args.TSP_epoch:
            best_dev_loss = 1e9
        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch=epoch+1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    print(args.test_name)
    print(f'Training done, best epoch: {best_epoch}')
    writer.close()

def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='summary/test',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')

    #model parameters
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', nargs='+', default=24, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=25,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--sub-lr', type=float, default=1e-1, help='lerning rate of the sub-samping layel')

    # trajectory learning parameters
    parser.add_argument('--trajectory_learning', default=True, help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')
    parser.add_argument('--acc-weight', type=float, default=1e-3, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--a-max', type=float, default=3, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=30, help='maximum velocity')
    parser.add_argument('--TSP', default=True, help='Using the PILOT-TSP algorithm,if False using PILOT.')
    parser.add_argument('--TSP-epoch', default=30, type=int, help='Epoch to preform the TSP reorder at')
    parser.add_argument('--initialization', type=str, default='spiral',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    return parser

if __name__ == '__main__':
    train()
