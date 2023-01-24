arr=($GPUS)
NGPUS=${#arr[@]}

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

pathout="ffqh256-50k-recnet"
for gpuid in $GPUS
do
	idx=$(($idx+1)) 
	echo $(($idx-1))
	(CUDA_VISIBLE_DEVICES=$gpuid python scripts/sample_diffusion.py \
		--vqgan_ckpt ../models/ffhq/ldm/recnet/epoch=35-step=38999.ckpt \
		-r ../models/ffhq/ldm/model.ckpt \
		-l $pathout \
		-n 50000 \
		--batch_size 64 \
		-c 200 \
		-e 1.0 \
		--idx $idx \
		--n_split $NGPUS --i_split $(($idx-1)) ) &
done
