# image_sample.py
# --- only diffs are annotated with [DAPS] ---
import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import yaml
import torch
import tqdm
import glob
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets import IQTDataset
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.condition_methods import get_conditioning_method

import matplotlib.pyplot as plt
import time

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
   ...
def load_data_custom(data_loader):
   ...

def main():
    set_seed(42)

    with open('/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/baselines/configs_daps.yaml') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(configs=configs,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print('Using device:', device)

    save_path = '/cluster/project0/IQT_Nigeria/skim/daps_out'

    # (data loading unchanged)  ------------------------------------------------
    lst_files = ['116120', '116221', '116423', '116524', '116726', '117021', '117122', '117324', '117728', '117930', '118023', '118124', '118225', '118528', '118730', '118831', '118932', '119025', '119126', '119732']
    lst_files = lst_files[:1]
    save_files_pred = {i: [] for i in lst_files}
    save_files_gt = {i: [] for i in lst_files}
    save_files_lr = {i: [] for i in lst_files}

    data_dir = '/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim/1*'
    files = glob.glob(data_dir + '/T1w/T1w_acpc_dc_restore_brain.nii.gz')
    files_new = [f for f in files if f.split('/')[-3] in lst_files]
    files = files_new

    dataset = IQTDataset(files, configs=configs, return_id=configs['data']['return_id'])
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    try:
        ref_img, data_dict = next(iter(data))
        print(f"Batch: ref_img shape: {ref_img.shape}, data_dict: {data_dict}")
    except Exception as e:
        print(f"Error in batch: {e}")

    # Prepare Operator and noise (unchanged) -----------------------------------
    measure_config = configs['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Working directory (unchanged) -------------------------------------------
    save_dir = '/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/results/'
    out_path = os.path.join(save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare conditioning method (unchanged handle + keep object) ------------
    cond_config = configs['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {configs['conditioning']['method']}")

    logger.log("sampling...")
    all_images, ys, refs, time_lst = [], [], [], []

    # [DAPS] pull DAPS hyperparams (optional; defaults set inside sampler)
    daps_cfg = configs.get('daps', {})  # e.g., {'num_steps':50,'eta':1e-4,'likelihood_weight':1.0,'sigma_min':0.002,'sigma_max':80,'rho':7.0,'noise_scale':1.0}

    for i, (ref_img, data_dict) in tqdm.tqdm(enumerate(data)):
        print(f"{i}/{len(data)}")
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device)
            model_kwargs["y"] = classes

        ref_img = ref_img.to(device)

        # (unchanged: load your UNet x0 estimate to support skip) --------------
        fname_curr, slice_curr = int(data_dict['file_id'][0]), str(data_dict['slice_idx'].numpy()[0])
        data_np = np.load(f'/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/cond_results/unet/ood_contrast/{fname_curr}/pred_{slice_curr}_axial.npy')[0]
        data_np = np.clip(data_np, 0., 1.0)

        if configs['skip_timestep']:
            skip_x0 = torch.tensor(data_np).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        else:
            skip_x0 = None

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # [DAPS] choose sampler: DAPS (EDM/Langevin) vs your original DDPM/DDIM
        use_daps = configs['conditioning']['params'].get('daps', False)
        print("Using DAPS:", use_daps)

        start = time.time()
        if use_daps:
            # [DAPS] new entry point: Langevin posterior sampling with EDM schedule
            sample = diffusion.daps_sample_loop(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                measurement=y_n.to(torch.float32),
                measurement_cond_fn=measurement_cond_fn,                 # we need grad_and_value
                model_kwargs=model_kwargs,
                device=device,
                # optional overrides from YAML
                K_inner=daps_cfg.get('K_inner', 1), # 1–3 is enough
                eta_inner_c=daps_cfg.get('eta_inner_c', 1e-5), # base step for inner ULA
                r_coeff=daps_cfg.get('r_coeff', 1.0), # scales tether radius r_t
                beta_y=daps_cfg.get('beta_y', 1.0), # meas. noise std in loss
                likelihood_weight=daps_cfg.get('likelihood_weight', 50000000.0), # weight for likelihood term
                line_search=configs['line_search'],                            # True/False; whether to do backtracking line search
            )
        else:
            #finish the code
            pass
            # original path (DDPM or DDIM)  -----------------------------------
            sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
            sample = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                measurement=y_n.to(torch.float32),
                measurement_cond_fn=measurement_cond_fn,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                skip_timesteps=configs['skip_timestep'],
                skip_x0=skip_x0,
                line_search=configs['line_search'],
            )
        end = time.time()
        print("Inf time: ", end - start)
        time_lst.append(end - start)

        #sample = sample.contiguous()
        all_images.append(sample.cpu().numpy())
        refs.append(ref_img.cpu().numpy())
        ys.append(y_n.cpu().numpy())
        print("One image done!")

        # (saving unchanged) ---------------------------------------------------
        if data_dict is not None:
            for j in range(args.batch_size):
                os.makedirs(f'{save_path}/{data_dict["file_id"][j]}', exist_ok=True)
                np.save(f'{save_path}/{data_dict["file_id"][j]}/pred_{data_dict["slice_idx"][j]}_axial.npy', sample[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/gt_{data_dict["slice_idx"][j]}_axial.npy', ref_img[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/lr_{data_dict["slice_idx"][j]}_axial.npy', y_n[j].cpu().numpy())

    time_lst = np.array(time_lst)
    print("Mean time: ", np.mean(time_lst))
    print("Std time: ", np.std(time_lst))
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/logs_large_zero2two_HCPMoreSlice2025/model360000.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
