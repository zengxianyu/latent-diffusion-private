import pdb
import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
import sys
sys.path.append(".")
sys.path.append("../guided-diffusion")
from guided_diffusion.image_datasets import load_data

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i_split",
        type=int,
        default=0
    )
    parser.add_argument(
        "--n_split",
        type=int,
        default=1
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        required=False
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    config = OmegaConf.load(opt.config)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)

    gpu = True
    eval_mode = True

    print(config)

    model = load_model_from_config(config.model, opt.ckpt)
    model.cuda()

    if opt.vqgan_ckpt is not None:
        state = torch.load(opt.vqgan_ckpt)
        m, u = model.first_stage_model.load_state_dict(state['state_dict'], strict=False)
        print("missng ")
        print(m)
        print("unused")
        print(u)

    data = load_data(
        data_dir=opt.data_dir,
        batch_size=opt.batch_size,
        image_size=model.model.diffusion_model.image_size,
        class_cond=True,
        return_name=True,
        return_prefix=True,
        deterministic=True,
        n_split=opt.n_split,
        i_split=opt.i_split,
        imagenet=True,
        return_loader=True
    )
    data = iter(data)
    bdata = next(data, None)

    ddim_steps = 250
    scale = 1.5   # for unconditional guidance

    sampler = DDIMSampler(model)

    while bdata is not None:
        sample, cond = bdata
        bsize = sample.size(0)
        bdata = next(data, None)
        filename = cond.pop("filename")

        with torch.no_grad():
            with model.ema_scope():
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(bsize*[1000]).to(model.device)}
                    )

                xc = cond['y']
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=bsize,
                                                 shape=[3,
                                                         model.model.diffusion_model.image_size,
                                                         model.model.diffusion_model.image_size],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                             min=0.0, max=1.0)

            sample = (x_samples_ddim * 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            sample = sample.cpu().numpy()
            prefix = cond.pop("prefix") if "prefix" in cond else [""]*bsize
            # save
            for i in range(len(sample)):
                out_path_i = os.path.join(opt.out_path, "output", prefix[i])
                if not os.path.exists(out_path_i):
                    os.makedirs(out_path_i, exist_ok=True)
                arr0 = sample[i]
                out_path_i = os.path.join(out_path_i, filename[i])
                Image.fromarray(arr0).save(out_path_i+".png")
                print(f"saving to {out_path_i}")


    print("done.")
