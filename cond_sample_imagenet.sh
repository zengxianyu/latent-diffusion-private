arr=($GPUS)
NGPUS=${#arr[@]}

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

for gpuid in $GPUS
do
	idx=$(($idx+1)) 
	echo $(($idx-1))
	(CUDA_VISIBLE_DEVICES=$gpuid python scripts/cond_sample_diffusion.py \
		--batch_size 32 \
		--ckpt models/ldm/cin256-v2/model.ckpt \
		--config configs/latent-diffusion/cin256-v2.yaml \
		--data_dir /data/yzeng22/imagenetval_sample1k_center256 \
		--out_path ./samples/imagenet_pretrain_init \
		--n_split $NGPUS --i_split $(($idx-1)) ) &
done
