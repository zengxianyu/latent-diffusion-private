python main.py --base configs/autoencoder/autoencoder_vq_8x8x3.yaml -t --gpus 0, \
	model.target=ldm.models.autoencoder.VQDecModel \
	model.params.image_key=png \
	data.params.train.target=ldm.data.listdata.SimpleListDataset \
	data.params.train.params.data_dir=/data/yzeng22/FFHQ_recon \
	data.params.train.params.data_list=/data/yzeng22/FFHQ_recon.txt \
	data.params.train.params.postfix=["png"] \
	data.params.validation.target=ldm.data.listdata.SimpleListDataset \
	data.params.validation.params.data_dir=/data/yzeng22/FFHQ_recon \
	data.params.validation.params.data_list=/data/yzeng22/FFHQ_recon_val.txt \
	data.params.validation.params.postfix=["png"] \
	--name vqgan-nocls-recim \
	--finetune logs/2022-12-28T18-31-56_vqgan-cls/testtube/version_0/checkpoints/vq-f4.ckpt \
