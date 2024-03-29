python main.py --base configs/autoencoder/autoencoder_vq_8x8x3_recnet.yaml -t --gpus 0,1,2,3 \
	model.target=ldm.models.autoencoder.VQDecModel \
	model.params.recnet_path="../imagenet/q_100_noise_cls1/net_280000.pth" \
	model.params.image_key="JPEG" \
	data.params.train.target=ldm.data.listdata.SimpleListDataset \
	data.params.train.params.data_dir=/data/common/ILSVRC/Data/CLS-LOC/train \
	data.params.train.params.data_list=/data/yzeng22/imagenet_train.txt \
	data.params.train.params.postfix=["JPEG"] \
	data.params.train.params.augpf=["JPEG"] \
	data.params.validation.target=ldm.data.listdata.SimpleListDataset \
	data.params.validation.params.data_dir=/data/common/ILSVRC/Data/CLS-LOC/val \
	data.params.validation.params.data_list=/data/yzeng22/imagenet_val.txt \
	data.params.validation.params.postfix=["JPEG"] \
	data.params.validation.params.augpf=["JPEG"] \
	data.params.batch_size=8 \
	--finetune vq-f4.ckpt \
	--name vqgan-imagenetinit-recnet \
