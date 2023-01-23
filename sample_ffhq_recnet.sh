arr=($GPUS)
NGPUS=${#arr[@]}
#--vqgan_ckpt ./vq-f4.ckpt \
#--vqgan_ckpt logs/2022-12-28T19-12-19_vqgan-nocls/testtube/version_0/checkpoints/epoch=0-step=47999.ckpt \
#logs/2023-01-02T15-52-06_vqgan-ffhqmixinit-nocls/testtube//version_0/checkpoints/epoch=16-step=17999.ckpt

pathin="logs/2023-01-19T23-31-29_vqgan-ffhqmixinit-recnet-multi-idx3/testtube/version_0/checkpoints"
pathgt="/data/FFHQ/images256x256_sample1k"
files="
epoch=30-step=65999.ckpt
epoch=28-step=62999.ckpt
epoch=27-step=59999.ckpt
epoch=1-step=2999.ckpt
epoch=20-step=44999.ckpt
epoch=21-step=47999.ckpt
epoch=23-step=50999.ckpt
epoch=24-step=53999.ckpt
epoch=26-step=56999.ckpt
epoch=2-step=5999.ckpt
epoch=10-step=23999.ckpt
epoch=12-step=26999.ckpt
epoch=13-step=29999.ckpt
epoch=15-step=32999.ckpt
epoch=16-step=35999.ckpt
epoch=17-step=38999.ckpt
epoch=19-step=41999.ckpt
epoch=4-step=8999.ckpt
epoch=5-step=11999.ckpt
epoch=6-step=14999.ckpt
epoch=8-step=17999.ckpt
epoch=9-step=20999.ckpt
"
pathlog="output/ffqh256-1k/multi-idx3"

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

for file in $files
do
	pathout="output/ffqh256-1k/multi-idx3/$file"
	for gpuid in $GPUS
	do
		idx=$(($idx+1)) 
		echo $(($idx-1))
		(CUDA_VISIBLE_DEVICES=$gpuid python scripts/sample_diffusion.py \
			--vqgan_ckpt $pathin/$file \
			-r models/ldm/ffhq256/model.ckpt \
			-l $pathout \
			-n 1000 \
			--batch_size 16 \
			-c 200 \
			-e 1.0 \
			--n_split $NGPUS --i_split $(($idx-1)) ) &
	done
	wait
	CUDA_VISIBLE_DEVICES=${arr[0]} python -m pytorch_fid $pathgt $pathout/img --device cuda:0 | grep FID | sed "s/^/file $file /" >> $pathlog.txt
done
