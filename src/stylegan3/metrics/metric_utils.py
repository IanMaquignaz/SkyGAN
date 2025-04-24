# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
import training.training_loop
import training.loss
import training.utils

# Custom
from pathlib import Path
from .metrics_HDR import get_metrics_test
from utils_os import folder_flush
from utils_cv.io.opencv import save_image as cv_save_image


#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, E=None, E_kwargs={}, G=None, G_kwargs={}, dataset_kwargs={}, clear_dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, run_dir=None, use_encoder=True):
        assert 0 <= rank < num_gpus
        self.E              = E
        self.E_kwargs       = dnnlib.EasyDict(E_kwargs)
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.clear_dataset_kwargs = dnnlib.EasyDict(clear_dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.run_dir        = run_dir
        self.use_encoder    = use_encoder


#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            # c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            c = torch.stack(c).pin_memory().to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def to_LDR(img, drange=(0,1)):
    # See training_loop.save_image_grid
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) / (hi - lo) # fix range
    # img = np.rint(
    #     training.training_loop.linear2srgb(img)*255
    # ).clip(0, 255).astype(np.uint8) # to LDR
    img = training.training_loop.linear2srgb(img).clip(0, 1).astype(np.float32) # to LDR
    return img

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            print('DEBUG: cached features will be used:', cache_file)
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

         # [0, 1] -> [-1, 1]
        images = training.training_loop.stretch(images).to(opts.device)#.split(batch_gpu)

        # THIS IS WRONG!
        # IMAGES SHOULD BE TONEMAPPED FOR METRICS AND FID ACCEPTS [0,1]
        # images = torch.from_numpy(training.utils.invert_log_transform(images.cpu().numpy())).to(opts.device)
        # images = (images * 255).clamp(0, 255).to(torch.uint8)

        images = training.training_loop.unstretch(images).to(opts.device)
        images = training.utils.invert_log_transform(images.cpu().numpy())
        images = torch.from_numpy(training.utils.fix_gamma(images)).to(opts.device)
        # images = images.clamp(0,1)

        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=72, batch_gen=8, clear_data_loader_kwargs=None, **stats_kwargs):
    print('clear_dataset_kwargs=', opts.clear_dataset_kwargs)
    clear_dataset = dnnlib.util.construct_class_by_name(**opts.clear_dataset_kwargs)
    if clear_data_loader_kwargs is None:
        clear_data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)


    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    E = copy.deepcopy(opts.E).eval().requires_grad_(False).to(opts.device)
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # TODO clear dataset
    #c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    if stats_kwargs['max_items'] == None:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        stats_kwargs['max_items'] = len(dataset)
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None

    #item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)] # ?
    clear_data_loader = iter(torch.utils.data.DataLoader(
        dataset=clear_dataset,
        #sampler=item_subset,
        batch_size=batch_gen, **clear_data_loader_kwargs))

    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    mask = torch.from_numpy(training.utils.circular_mask((G.img_channels, G.img_resolution, G.img_resolution)))

    # Main loop.
    while not stats.is_full():
        images = []
        #print("batch_size // batch_gen", batch_size // batch_gen)
        for _i in range(batch_size // batch_gen):
            print("Images=", len(images))
            #print("stats.num_items, stats.max_items", stats.num_items, stats.max_items)
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            clear_img, clear_c = next(clear_data_loader)
            clear_img = training.training_loop.stretch(clear_img).to(opts.device)#.split(batch_gpu)
            #clear_img = training.training_loop.stretch(clear_img.to(device).to(torch.float32))#.split(batch_gpu)
            #img = G(z=z, c=next(c_iter), **opts.G_kwargs)

            #print('z.shape', z.shape)
            #print('len(clear_img)', len(clear_img))
            #print('clear_img.shape', clear_img.shape)

            # Inject and encode
            img_fake, img_clear_rec, _gen_ws = training.loss.run_EG(
                z=z, c=clear_c, clear_img=clear_img,
                use_encoder=opts.use_encoder, E=E, G=G,
                style_mixing_prob=0, mask=mask, update_emas=False,
                **opts.E_kwargs, **opts.G_kwargs # TODO E/G_kwargs - pass to E/G respectively, not here?
            )

            # THERE IS SOMETHING VERY WRONG HERE.
            #print('img_fake.shape', img_fake.shape)
            # img_fake = training.training_loop.unstretch(img_fake)
            # img_fake = training.utils.invert_log_transform(img_fake.cpu().numpy())
            # img_fake = torch.from_numpy(training.utils.fix_gamma(img_fake)).to(opts.device)
            # # img_fake = img_fake.clamp(0,1)

            # # TODO invert_log_transform -> [0,1+]
            # #print('img_fake', img_fake.min(), img_fake.mean(), img_fake.max())
            # # img_fake = (img_fake * 255).clamp(0, 255).to(torch.uint8)

            # NP Linear HDR (maybe?)
            img_fake = training.utils.invert_log_transform(
                img_fake.detach().clone().cpu().numpy()
            )
            img_real = training.utils.invert_log_transform(
                img_real.detach().clone().cpu().numpy()
            )
            # Torch LDR
            img_fake_LDR = torch.from_numpy(
                to_LDR(img_fake.copy())
            ).to(opts.device)
            images.append(img_fake_LDR)

        #print('len(images)', len(images))
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        #grid_size=[4,4]
        #training.training_loop.save_image_grid(images[:16,...].cpu().numpy(), 'metric_generator.png', [0, 255], grid_size=grid_size)
        #exit(123)

        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        print("stats.num_items, stats.max_items", stats.num_items, stats.max_items)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------


def export_feature_for_generator(opts, subfolder="TEST"):
    # Cleanup
    folder = os.path.join(opts.run_dir,subfolder)
    Path(folder).mkdir(exist_ok=True)
    folder_flush(os.path.join(opts.run_dir,subfolder), verbose=True)
    print("Metrics will be output to:", opts.run_dir)
    # export_bool = False if subfolder == "TEST" else True
    export_bool = True # Needed for manual evaluation.

    # Setup Dataloader.
    print("Initializing HDRDB...")
    batch_size = 1
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    num_items = len(dataset)
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    data = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, shuffle=False, **data_loader_kwargs)
    print("Initializing HDRDB... Complete!")

    print("Initializing ClearSkies...")
    clear_dataset = dnnlib.util.construct_class_by_name(**opts.clear_dataset_kwargs)
    clear_data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    data_clear = torch.utils.data.DataLoader(dataset=clear_dataset, sampler=item_subset, batch_size=batch_size, shuffle=False, **clear_data_loader_kwargs)
    print("Initializing ClearSkies... Complete!")

    # Setup generator and labels.
    print("Initializing model...")
    E = copy.deepcopy(opts.E).eval().requires_grad_(False).to(opts.device)
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    mask = torch.from_numpy(training.utils.circular_mask((G.img_channels, G.img_resolution, G.img_resolution)))
    print("Initializing model... Complete!")

    # Setup Metrics.
    print("Initializing metrics...")
    metrics_cLDR, metrics_HDR = get_metrics_test(
        shape = opts.dataset_kwargs.resolution,
        envmap_maskBorder=torch.logical_not(mask[0].detach().clone().bool().squeeze()).to(opts.device),
        envmap_maskValid=mask[0].detach().clone().bool().squeeze().to(opts.device),
        device=opts.device,
    )

    from utils_ml.metrics import EarthMoversDistance as EMD # EMD
    emd = EMD(
        bins=1000,
        range=[0,20000],
        mask=mask[0].detach().clone().bool().squeeze().to(opts.device),
    ).to(opts.device)
    print("Initializing metrics... Complete!")

    # Main loop.
    print("Starting export loop...")
    for i, ((img_real, labels_c),(img_clear_real, clear_c)) in enumerate(zip(data, data_clear)):

        print('Iteration:', i)
        if img_real.shape[1] == 1:
            img_real = img_real.repeat([1, 3, 1, 1])

        # [0, 1] -> [-1, 1]
        img_real = training.training_loop.stretch(img_real).to(opts.device)
        img_clear_real = training.training_loop.stretch(img_clear_real).to(opts.device)

        # Inject and encode
        z = torch.randn([batch_size, G.z_dim], device=opts.device)
        img_fake, img_clear_rec, _gen_ws = training.loss.run_EG(
            z=z, c=clear_c, clear_img=img_clear_real,
            use_encoder=opts.use_encoder, E=E, G=G,
            style_mixing_prob=0, mask=mask, update_emas=False,
            **opts.E_kwargs, **opts.G_kwargs # TODO E/G_kwargs - pass to E/G respectively, not here?
        )

        print(f"img_fake_out={img_fake.min()}, {img_fake.max()}, {img_fake.shape}")
        print(f"img_real_out={img_real.min()}, {img_real.max()}, {img_real.shape}")

        # NP Linear HDR (maybe?)
        img_fake = training.utils.invert_log_transform(
            img_fake.detach().clone().cpu().numpy()
        )
        img_real = training.utils.invert_log_transform(
            img_real.detach().clone().cpu().numpy()
        )

        # Torch LDR
        img_fake_LDR = torch.from_numpy(
            to_LDR(img_fake.copy())
        ).to(opts.device)
        img_real_LDR = torch.from_numpy(
            to_LDR(img_real.copy())
        ).to(opts.device)
        print(f"img_fake_LDR={img_fake_LDR.min()}, {img_fake_LDR.max()}, {img_fake_LDR.shape}")
        print(f"img_real_LDR={img_real_LDR.min()}, {img_real_LDR.max()}, {img_real_LDR.shape}")

        # Torch HDR
        img_fake_HDR = torch.from_numpy(
            img_fake.copy()
        ).to(opts.device).clamp(min=0)
        print(f"img_fake_HDR={img_fake_HDR.min()}, {img_fake_HDR.max()}, {img_fake_HDR.shape}")
        img_real_HDR = torch.from_numpy(
            img_real.copy()
        ).to(opts.device).clamp(min=0)
        print(f"img_real_HDR={img_real_HDR.min()}, {img_real_HDR.max()}, {img_real_HDR.shape}")
        # labels_c = labels_c.to(opts.device)
        # label = label.reshape(1,1,256,256)

        # Export PNG
        if export_bool:
            # img_png = (img.detach().clone() * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze()
            cv_save_image(f"{folder}/{i:06}.png", img_fake_LDR.clone().squeeze(), PNG=True)
            cv_save_image(f"{folder}/{i:06}_gt.png", img_real_LDR.clone().squeeze(), PNG=True)
            cv_save_image(f"{folder}/{i:06}.exr", img_fake_HDR.clone().squeeze(), PNG=False)
            cv_save_image(f"{folder}/{i:06}_gt.exr", img_real_HDR.detach().clone().squeeze(), PNG=False)

        for k in metrics_cLDR.keys():
            if 'KernelInceptionDistance' in k  or 'FrechetInceptionDistance' in k:
                metrics_cLDR[k].update(img_real_LDR.detach().clone(), real=True)
                metrics_cLDR[k].update(img_fake_LDR.detach().clone(), real=False)
            elif 'InceptionScore' in k or 'CLIPImageQualityAssessment' in k:
                if '_real' in k:
                    metrics_cLDR[k].update(img_real_LDR.detach().clone())
                else:
                    metrics_cLDR[k].update(img_fake_LDR.detach().clone())
            elif 'CLIPScore' in k :
                raise NotImplementedError
                # if '_real' in k:
                #     metrics[k].update(real.detach().clone(), cliptext)
                # else:
                #     metrics[k].update(fake.detach().clone(), cliptext)
            elif 'CLIPImageQualityAssessment' in k :
                if 'real' in k:
                    metrics_cLDR[k].update(img_real_LDR.detach().clone())
                else:
                    metrics_cLDR[k].update(img_fake_LDR.detach().clone())
            else:
                # SSIM, MS-SSIM,
                metrics_cLDR[k].update(
                    img_fake_LDR.detach().clone(),
                    img_real_LDR.detach().clone(),
                )

        for k in metrics_HDR.keys(keep_base=True):
            metrics_HDR[k].update(
                img_fake_HDR.detach().clone(),
                img_real_HDR.detach().clone(),
            )
        emd.update(
            img_fake_HDR.detach().clone(),
            img_real_HDR.detach().clone(),
        )

    # Metrics print results
    result = emd.compute()
    hist_error_sum = emd.plot_cummulative_histogram_error()
    from torchvision.utils import save_image
    save_image(hist_error_sum, f"{folder}/_hist_error_sum_.png")

    #####################
    ## Compute Metrics ##
    #####################
    metric_output = metrics_HDR.compute()
    metric_output_cLDR = metrics_cLDR.compute()
    for k in metric_output_cLDR.keys():
        if 'KernelInceptionDistance' in k:
            kid_mean, kid_std = metric_output[k]
            metric_output['KID_mean'] = kid_mean
            metric_output['KID_std'] = kid_std
        elif 'InceptionScore' in k:
            is_mean, is_std = metric_output[k]
            tail = '_fake' if '_fake' in k else '_real'
            metric_output['IS_mean'+tail] = is_mean
            metric_output['IS_std'+tail] = is_std
        elif 'CLIPImageQualityAssessment' in k:
            CLIP_IQA = metric_output[k].mean()
            tail = '_fake' if '_fake' in k else '_real'
            metric_output['CLIP_IQA'+tail] = CLIP_IQA
    metric_output["emd2"] = result

    with open(f"{folder}/_metrics_.txt", 'a') as metrics_file:
        for k in metric_output.keys():
            metrics_file.write(f"{k}: {metric_output[k].cpu().item()}\n")

#----------------------------------------------------------------------------
