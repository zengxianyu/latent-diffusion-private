#--vqgan_ckpt ./vq-f4.ckpt \
#--vqgan_ckpt logs/2022-12-28T19-12-19_vqgan-nocls/testtube/version_0/checkpoints/epoch=0-step=47999.ckpt \
#logs/2023-01-02T15-52-06_vqgan-ffhqmixinit-nocls/testtube//version_0/checkpoints/epoch=16-step=17999.ckpt
python scripts/sample_diffusion.py \
        -r models/ldm/ffhq256/model.ckpt \
        -l output/ffqh256-1k/pretrain \
	-n 1000 \
	--batch_size 32 \
	-c 200 \
	-e 1.0
