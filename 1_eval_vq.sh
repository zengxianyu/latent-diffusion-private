FILES="
epoch=2-step=551999.ckpt
"
#for seed in "${seeds[@]}";
for file in $FILES
do
	path_out=./output/vqgan/file_$file
	echo $path_out
	(python scripts/evalvqgan.py  \
		--config configs/autoencoder/autoencoder_vq_8x8x3.yaml --ckpt "logs/2022-12-27T16-32-24_autoencoder_vq_8x8x3/testtube/version_0/checkpoints/testtube/version_2/checkpoints/$file" \
		--batch_size 8 \
		--outdir $path_out \
		--allow_tf32 False \
		data.target=main.DataModuleFromConfig \
		data.params.batch_size=8 data.params.num_workers=8 data.params.wrap=False \
		data.params.eval.target=ldm.data.listdata.SimpleListDataset \
		data.params.eval.params.postfix=["png"] \
		data.params.eval.params.data_dir='output/ffqh256/init/ffhq256/samples/00634478/2022-12-27-15-11-58/img' \
		data.params.eval.params.data_list='output/ffqh256/init/ffhq256/samples/00634478/2022-12-27-15-11-58/img.txt')
done
