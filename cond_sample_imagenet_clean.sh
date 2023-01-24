arr=($GPUS)
NGPUS=${#arr[@]}

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

for gpuid in $GPUS
do
	idx=$(($idx+1)) 
	echo $(($idx-1))
	(CUDA_VISIBLE_DEVICES=$gpuid python scripts/cond_sample_diffusion.py \
		--vqgan_ckpt ../models/imagenet/ldm/clean/epoch=0-step=17999.ckpt \
		--batch_size 64 \
		--ckpt ../models/imagenet/ldm/model.ckpt \
		--config configs/latent-diffusion/cin256-v2.yaml \
		--data_dir ../data/ILSVRC2012_img_val_folders_center256 \
		--out_path ./imagenetval-clean \
		--n_split $NGPUS --i_split $(($idx-1)) ) &
done
