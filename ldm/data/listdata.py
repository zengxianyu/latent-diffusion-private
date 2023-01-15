from torch.utils.data import Dataset
from PIL import Image
import torch
import pdb
from torchvision import transforms
import random
import math
import numpy as np

class SimpleListDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_list,
        postfix=["txt"],
        nsplit=1,
        isplit=0,
        shuffle=False,
        size=256,
        resize_size=None,
        augpf=[],
        **kwargs
    ):
        super().__init__()
        with open(data_list,"r") as f:
            lines = f.readlines()
        self.lines = [l.strip("\n") for l in lines]
        splitsize = math.ceil(float(len(self.lines))/nsplit)
        self.lines = self.lines[isplit*splitsize:(isplit+1)*splitsize]
        self.root = data_dir
        self.postfix = postfix
        self.augpf = augpf
        resize_size = size if resize_size is None else resize_size
        if shuffle:
            random.shuffle(self.lines)
        self.resize64 = transforms.Resize(64)
        self.resize = transforms.Resize(resize_size)
        self.totensor= \
            transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])
        if len(augpf)>0:
            self.augtrans= \
                transforms.Compose([
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip()])


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        try:
            path = self.lines[idx]
            sample = {"filename":path}
            split_size = []
            split_name = []
            chunk = []
            for name in self.postfix:
                path_data = f"{self.root}/{path}.{name}"
                if name=="txt":
                    with open(path_data,"r") as f:
                        data =  f.readlines()[0]
                    data = data.strip("\n")
                elif name.endswith("npy"):
                    data = np.load(path_data)
                    data = torch.FloatTensor(data)
                elif name in ["jpg", "jpeg", "png", "JPEG"]:
                    data = Image.open(path_data).convert("RGB")
                    data = self.totensor(data)
                    data = data*2-1
                else:
                    raise NotImplementedError
                if name in self.augpf:
                    split_name.append(name)
                    data = self.resize(data)
                    chunk.append(data)
                    split_size.append(data.size(0))
                sample[name] = data
            if len(self.augpf)>0:
                chunk = torch.cat(chunk)
                chunk = self.augtrans(chunk)
                chunk = torch.split(chunk, split_size)
                for i,sn in enumerate(split_name):
                    sample[sn] = chunk[i]
            if 'map.npy' in sample:
                sample['map.npy'] = self.resize64(sample['map.npy'])
            return sample
        except:
            idx = (idx+1)%self.__len__()
            return self.__getitem__(idx)

if __name__ == "__main__":
    import webdataset as wds
    from tqdm import tqdm
    import sys
    import os
    idx = int(sys.argv[1])
    idx_rt = int(sys.argv[2])
    data_dir = "/home/yuzeng/train2017_bbox_womap"
    data_list = f"/home/yuzeng/train2017_bbox_womap_lists/train2017_bbox_womap.{idx}.txt"
    #data_out = "s3://west2-zengyu/train2017_bbox_womap_aug_4"
    #data_out_file = f"pipe:aws s3 cp - {data_out}"+"/{0:05d}".format(idx)+"_{0:05d}.tar".format(idx_rt)
    data_out = "/sensei-fs/tenants/Sensei-AdobeResearchTeam/share-yuzeng/train2017_bbox_womap_aug_4"
    data_out_file = f"{data_out}"+"/{0:05d}".format(idx)+"_{0:05d}.tar".format(idx_rt)
    dataset = SimpleListDataset(
        data_dir,
        data_list,
        postfix=["txt", "jpg", "wom.npy", "map.npy"],
        nsplit=1,
        isplit=0,
        shuffle=False,
        size=(512,512),
        resize_size=(640,640),
        augpf=["jpg", "map.npy"]
    )
    num_fail = 0
    num_suc = 0
    with wds.TarWriter(data_out_file) as dst:
        for i, data in tqdm(enumerate(iter(dataset))):
            #try:
            filename = data.pop('filename')
            try:
                data['jpg'] = Image.fromarray((data['jpg']*255).numpy().astype(np.uint8).transpose((1,2,0)))
                data['map.npy'] = data['map.npy'].bool().numpy()
                data['wom.npy'] = data['wom.npy'].bool().numpy()
                data['__key__'] = f"{filename}_{idx}_{idx_rt}"
                dst.write(data)
                num_suc += 1
                print("sucess")
                print(data_out_file)
                print(f"sucess rate {float(num_suc)/(num_suc+num_fail)}")
            except:
                print("pass")
                num_fail += 1
                print(f"sucess rate {float(num_suc)/(num_suc+num_fail)}")
                continue
