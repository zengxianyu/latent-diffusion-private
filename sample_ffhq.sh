python scripts/sample_diffusion.py \
	-r models/ldm/ffhq256/model.ckpt \
	-l output/ffqh256/init \
	-n 16 \
	--batch_size 8 \
	-c 200 \
	-e 1.0
