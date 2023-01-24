#--vqgan_ckpt ./vq-f4.ckpt \
#--vqgan_ckpt logs/2022-12-28T19-12-19_vqgan-nocls/testtube/version_0/checkpoints/epoch=0-step=47999.ckpt \
#logs/2023-01-02T15-52-06_vqgan-ffhqmixinit-nocls/testtube//version_0/checkpoints/epoch=16-step=17999.ckpt


pathin="logs/2023-01-02T15-53-22_vqgan-ffhqmixinit-recnet/testtube/version_0/checkpoints"
files="
epoch=38-step=41999.ckpt
epoch=41-step=44999.ckpt
epoch=43-step=47999.ckpt
epoch=46-step=50999.ckpt
epoch=49-step=53999.ckpt
epoch=52-step=56999.ckpt
epoch=54-step=59999.ckpt
"
pathlog="output/ffqh256-1k/mixinit-nocls.txt"
for file in $files
do
	pathout="output/ffqh256-1k/mixinit-nocls/$file"
	python scripts/sample_diffusion.py \
		--vqgan_ckpt $pathin/$file \
		-r models/ldm/ffhq256/model.ckpt \
		-l $pathout \
		-n 1000 \
		--batch_size 64 \
		-c 200 \
		-e 1.0
	python -m pytorch_fid /data/FFHQ/images256x256_sample1k $pathout/img --device cuda:0 | grep FID | sed "s/^/file $file /" >> $pathlog
done
