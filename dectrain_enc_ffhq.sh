python main.py --base configs/autoencoder/autoencoder_vq_8x8x3.yaml -t --gpus 0,1 \
	model.target=ldm.models.autoencoder.VQDecModel \
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
	data.params.batch_size=16 \
	--finetune logs/2022-12-31T14-15-22_vqgan-ffhq-nocls/ffhq-init/checkpoints/ffhq_init_vq-f4.ckpt \
	--name vqgan-ffhqmixinit-nocls \
