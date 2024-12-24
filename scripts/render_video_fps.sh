names=(fern flower fortress horns leaves orchids room trex)


for name in "${names[@]}";
do


CUDA_VISIBLE_DEVICES=1 python render.py -s dataset/nerf_llff_data/$name/ -m output/exp/llff/llff_images_8_disxgrad_things_scalelr0.03_depth0_near2_size-1_valid0.1_drop1.0_N4/$name/3_views --video --iteration 30000

done