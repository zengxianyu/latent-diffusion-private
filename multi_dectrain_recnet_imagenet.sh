python main.py --base configs/autoencoder/autoencoder_vq_8x8x3_recnet.yaml -t --gpus 0,1,2,3 \
	model.target=ldm.models.autoencoder.VQDecModel \
	model.params.recnet_path="../models/imagenet/cls0.1_label4/net_124000.pth" \
	model.params.recnetlabel=$idx_label \
	model.params.num_labels=4 \
	model.params.image_key="JPEG" \
	data.params.train.target=ldm.data.listdata.SimpleListDataset \
	data.params.train.params.data_dir=/data/ILSVRC_train_center256 \
	data.params.train.params.data_list=/data/yzeng22/imagenet_train.txt \
	data.params.train.params.postfix=["JPEG"] \
	data.params.train.params.augpf=["JPEG"] \
	data.params.validation.target=ldm.data.listdata.SimpleListDataset \
	data.params.validation.params.data_dir=/data/ILSVRC2012_img_val_folders_center256 \
	data.params.validation.params.data_list=/data/yzeng22/ILSVRC2012_img_val_folders_center256.txt \
	data.params.validation.params.postfix=["JPEG"] \
	data.params.validation.params.augpf=["JPEG"] \
	data.params.batch_size=8 \
	--name vqgan-imagenet-recnet-multi-idx$idx_label \
	--finetune  /mnt/store/yzeng22/genwm/latent-diffusion/vq-f4.ckpt
