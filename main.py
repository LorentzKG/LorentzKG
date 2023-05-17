import argparse

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from load_data import Data
from Experiment import Experiment


def run_experiments(data=None, margin=0.5, noise_reg=0.15, learning_rate=1e-3, dim=32,
                    nneg=10, npos=10, valid_steps=5, num_epochs=500, batch_size=50000, max_norm=1.5, max_grad_norm=5.,
                    optimizer='radam', cuda=True, early_stop=200, real_neg=False, device='cuda:0',
                    step_size=40, gamma=0.9
                    ):
    experiment = Experiment(
        data=data,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dim=dim,
        cuda=cuda,
        nneg=nneg,
        npos=npos,
        max_norm=max_norm,
        optimizer=optimizer,
        valid_steps=valid_steps,
        max_grad_norm=max_grad_norm,
        early_stop=early_stop,
        real_neg=real_neg,
        device=device,
        margin=margin,
        noise_reg=noise_reg,
        step_size=step_size,
        gamma=gamma,
    )
    mrr, hit1, hit3, hit10 = experiment.train_and_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="FB15k-237",
                        help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=1000,
                        help="Number of iterations.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=512,
                        help="Batch size.")
    parser.add_argument("--nneg",
                        type=int,
                        default=100,
                        help="Number of negative samples.")
    parser.add_argument("--npos",
                        type=int,
                        default=1,
                        help="Number of positive samples.")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-2,
                        help="Learning rate.")
    parser.add_argument("--dim",
                        type=int,
                        default=32,
                        help="Embedding dimensionality.")
    parser.add_argument('--early_stop', default=100, type=int)
    parser.add_argument('--max_norm', default=5, type=float)
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--max_grad_norm', type=float, default=3)
    parser.add_argument('--real_neg', action='store_true',default = True)
    parser.add_argument('--optimizer',
                        choices=['rsgd', 'radam', 'adam'],
                        default='radam')
    parser.add_argument('--valid_steps', default=5, type=int)
    parser.add_argument("--cuda",
                        type=bool,
                        default=True,
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--device",
                        type=str,
                        default='cuda:0',
                        help="device to use - if cuda = true, (cuda:0, cuda:1, ...), if cuda = false, (cpu)).")
    parser.add_argument("--data",
                        type=str,
                        default='data',
                        help="input data directory")
    parser.add_argument("--noise_reg",
                        type=float,
                        default=1e-2,
                        help="noise level at the regularization of distance")
    parser.add_argument("--step_size",
                        type=int,
                        default=30,
                        help="step size of the scheduler for optimizer")
    parser.add_argument("--gamma",
                        type=float,
                        default=0.9,
                        help="gamma of the scheduler for optimizer")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = f"{args.data}/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    if args.cuda:
        torch.cuda.set_device(args.device)
    d = Data(data_dir=data_dir)

    print(args)

    run_experiments(data=d, margin=args.margin, noise_reg=args.noise_reg,
                    learning_rate=args.lr, dim=args.dim, nneg=args.nneg, npos=args.npos,
                    valid_steps=args.valid_steps, num_epochs=args.num_epochs, batch_size=args.batch_size,
                    max_norm=args.max_norm, max_grad_norm=args.max_grad_norm, optimizer=args.optimizer,
                    early_stop=args.early_stop, real_neg=args.real_neg, device=args.device,
                    step_size=args.step_size, gamma=args.gamma,)
