# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import copy
import torch

import dnnlib
import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------


def init_dataset_kwargs(data, class_name='training.dataset.ImageFolderDataset', resolution=None, normalize_azimuth=False):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name=class_name, path=data, use_labels=True, max_size=None, xflip=False, resolution=resolution)
        #print('dataset_kwargs', dataset_kwargs)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        dataset_kwargs.normalize_azimuth = normalize_azimuth
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch.
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Print network summary.
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    E = copy.deepcopy(args.E).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(metric=metric, G=G, E=E, dataset_kwargs=args.dataset_kwargs, clear_dataset_kwargs=args.clear_dataset_kwargs,
            num_gpus=args.num_gpus, rank=rank, device=device, progress=progress, run_dir=args.run_dir, use_encoder=args.use_encoder)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--data', help='Dataset to evaluate against  [default: look up]', metavar='[ZIP|DIR]')
@click.option('--mirror', help='Enable dataset x-flips  [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
# Custom
@click.option('--resolution',           help='What image resolution to use (resize the training set)', metavar='INT', type=click.IntRange(min=64), default=None, show_default=True)
@click.option('--run_dir',         help='Output folder', metavar='PATH',  type=str, default=None)
@click.option('--use_encoder',  help='Enable the injection of encoded clear sky images', metavar='BOOL', type=bool, default=True, show_default=True)

def calc_metrics(
    ctx,
    network_pkl, metrics, data, mirror, gpus, verbose,
    resolution, run_dir, use_encoder
):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=eqt50k_int,eqr50k \\
        --network=~/training-runs/00000-stylegan3-r-mydataset/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

    \b
    Recommended metrics:
      fid50k_full  Frechet inception distance against the full dataset.
      kid50k_full  Kernel inception distance against the full dataset.
      pr50k3_full  Precision and recall againt the full dataset.
      ppl2_wend    Perceptual path length in W, endpoints, full image.
      eqt50k_int   Equivariance w.r.t. integer translation (EQ-T).
      eqt50k_frac  Equivariance w.r.t. fractional translation (EQ-T_frac).
      eqr50k       Equivariance w.r.t. rotation (EQ-R).

    \b
    Legacy metrics:
      fid50k       Frechet inception distance against 50k real images.
      kid50k       Kernel inception distance against 50k real images.
      pr50k3       Precision and recall against 50k real images.
      is50k        Inception score for CIFAR-10.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose, run_dir=run_dir, use_encoder=use_encoder)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')

    dataset_kwargs = None
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module
        args.E = network_dict['E_ema'] # subclass of torch.nn.Module

        def print_keys(d):
            if isinstance(d, dict):
                print(d.keys())
            for k in d.keys():
                if isinstance(d[k], dict):
                    print_keys(d[k])
        print_keys(network_dict)
        # exit()
        if resolution is None:
            resolution = network_dict['training_set_kwargs']['resolution']
            print(f"Updated resolution to {resolution}")
            dataset_kwargs = network_dict['training_set_kwargs']

    # Initialize dataset options.
    if data is not None:
        # args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
        # args.clear_dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ClearSkyDataset', path=data)

        # # Dataset
        # args.dataset_kwargs, _ = init_dataset_kwargs(
        #     data=data,
        #     resolution=resolution,
        # )
        # print("args.dataset_kwargs:", args.dataset_kwargs)
        # if dataset_kwargs is not None:
        #     print("ckpt dataset_kwargs:", dataset_kwargs)
        #     assert args.dataset_kwargs['resolution'] == dataset_kwargs['resolution']
        #     assert args.dataset_kwargs['use_labels'] == dataset_kwargs['use_labels']
        #     args.dataset_kwargs['normalize_azimuth'] = dataset_kwargs['normalize_azimuth']

        # # ClearSky Dataset
        # args.clear_dataset_kwargs, _ = init_dataset_kwargs(
        #     data=data,
        #     class_name='training.dataset.ClearSkyDataset',
        #     resolution=args.dataset_kwargs.resolution,
        #     normalize_azimuth=args.dataset_kwargs.normalize_azimuth
        # )

        args.dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=data,
            use_labels=dataset_kwargs['use_labels'],
            max_size=None,
            xflip=False,
            resolution=dataset_kwargs['resolution'],
            normalize_azimuth=dataset_kwargs['normalize_azimuth'],
        )
        args.clear_dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ClearSkyDataset',
            path=data,
            use_labels=dataset_kwargs['use_labels'],
            max_size=None,
            xflip=False,
            resolution=dataset_kwargs['resolution'],
            normalize_azimuth=dataset_kwargs['normalize_azimuth'],
        )

        print("args.clear_dataset_kwargs:", args.clear_dataset_kwargs)
    elif network_dict['training_set_kwargs'] is not None:
        raise NotImplementedError
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir.
    if args.run_dir == None:
        if os.path.isfile(network_pkl):
            pkl_dir = os.path.dirname(network_pkl)
            if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
                args.run_dir = pkl_dir
    else:
        pkl_dir = os.path.basename(os.path.dirname(network_pkl))
        args.run_dir = os.path.join(args.run_dir, pkl_dir)
        os.makedirs(args.run_dir, exist_ok=True)

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
