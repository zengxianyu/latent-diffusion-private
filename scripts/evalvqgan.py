import sys
sys.path.append(".")
import pdb
import argparse, os, sys, glob
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
#from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config

from transformers import AutoFeatureExtractor

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--allow_tf32",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    opt, unknown = parser.parse_known_args()
    if opt.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = OmegaConf.load(f"{opt.config}")
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    data = instantiate_from_config(config.data.params.eval)

    batch_size = opt.batch_size
    data = DataLoader(data,
            batch_size=batch_size,
            num_workers=batch_size, shuffle=False)
    data = iter(data)
    bdata = next(data, None)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                while bdata is not None:
                    filenames = bdata["filename"]
                    img = bdata['png']
                    enc,_,_ = model.encode(img.to(device))
                    dec = model.decode(enc)

                    x_samples_ddim = torch.clamp((dec + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    for name, x_sample in zip(filenames, x_checked_image_torch):
                        print(name)
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        try:
                            img.save(os.path.join(sample_path, name+".png"))
                        except:
                            pdb.set_trace()
                    bdata = next(data, None)

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
