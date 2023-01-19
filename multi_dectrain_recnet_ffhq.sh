python main.py --base configs/autoencoder/autoencoder_vq_8x8x3_recnet.yaml -t --gpus 0,1, \
	model.target=ldm.models.autoencoder.VQDecModel \
	model.params.recnet_path="../output/binary_multicls0.1_quant/net_110000.pth" \
	model.params.recnetlabel=1 \
	model.params.num_labels=4 \
	model.params.image_key=png \
	data.params.train.target=ldm.data.listdata.SimpleListDataset \
	data.params.train.params.data_dir=/data/FFHQ/images1024x1024 \
	data.params.train.params.data_list=/data/yzeng22/FFHQ_recon.txt \
	data.params.train.params.postfix=["png"] \
	data.params.train.params.augpf=["png"] \
	data.params.validation.target=ldm.data.listdata.SimpleListDataset \
	data.params.validation.params.data_dir=/data/FFHQ/images1024x1024 \
	data.params.validation.params.data_list=/data/yzeng22/FFHQ_recon_val.txt \
	data.params.validation.params.postfix=["png"] \
	data.params.validation.params.augpf=["png"] \
	data.params.batch_size=8 \
	--name vqgan-ffhqmixinit-recnet-multi-idx1 \
	--finetune  logs/2023-01-12T14-13-15_vqgan-ffhqmixinit-recnet-tanh0.05q/testtube/version_0/checkpoints/ffhq_init_vq-f4.ckpt
