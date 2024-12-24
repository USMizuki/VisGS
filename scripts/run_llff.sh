names=(fern flower fortress horns leaves orchids room trex)
# names=(room trex)
#n_views=(2 3 4 6 9)
n_views=(3)
export CUDA_VISIBLE_DEVICES=$1

iterations=40000
dataset_type=llff
flow_type=weight
flow_checkpoint=things
images=images_8
size_threshold=-1
prune_interval=-1 
depth_weight=0
scaling_lr=0.03
split_num=4
valid_dis_threshold=1.0
drop_rate=1.0
near_n=2
exp_name=${dataset_type}_${images}_${flow_type}_${flow_checkpoint}_scalelr${scaling_lr}_depth${depth_weight}_near${near_n}_size${size_threshold}_valid${valid_dis_threshold}_drop${drop_rate}_N${split_num}


for name in "${names[@]}";
do

for n in "${n_views[@]}";
do

dataset=dataset/nerf_llff_data/$name
workspace=output/exp_vis/llff/$exp_name/$name/${n}_views

#python train.py --source_path $dataset --model_path $workspace --eval --n_views 3 \
#    --sample_pseudo_interval 1 \
#    --lambda_dis 0.0001 --alpha_init 10 --alpha 0.9 --use_alpha --alpha_quantum 0.01 \
#    --lambda_depth 0.1 --depth_loss_threshold 0.001 \
#    --addpoint_interval 1000 --addpoint_from_iter 1000 --addpoint_until_iter 5000 --addpoint_threshold 10

# python train.py --source_path $dataset --model_path $workspace --eval --n_views 3 \
#     --sample_pseudo_interval 1 \
#     --lambda_dis 0.1 --alpha_quantum 0.5 --update_xyz \
#     --lambda_depth 0.1 --depth_loss_threshold 0.1 \
#     --addpoint_interval 10000 --addpoint_from_iter 1000 --addpoint_until_iter 5000 --addpoint_threshold 10 --pseudo_confidence

# python train.py --source_path $dataset --model_path $workspace --eval --n_views n --iterations 30000 --iterations_pre 5000 \
#     --save_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 \
#     --sample_pseudo_interval 1000000 --start_sample_pseudo 0 --end_sample_pseudo 0 \
#     --densify_until_iter 30000 \
#     --scaling_lr 0.09

# python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
#     --save_iterations 1000 2000 3000 4000 5000 10000 20000 30000 40000 45000 50000 \
#     --sample_pseudo_interval 1000000 --start_sample_pseudo 0 --end_sample_pseudo 0 \
#     --iterations $iterations \
#     --iterations_pre 0 \
#     --densify_until_iter $iterations \
#     --position_lr_max_steps $iterations \
#     --depth_weight 0 \
#     --diffusion_inpaint_iter -1 \
#     --scaling_lr 0.03 \
#     --size_threshol -1 \
#     --prune_depth_threshold -1 \
#     --prune_depth_scale 9 \
#     --prune_interval 10000000 \
#     --split_num 4 \
#     --valid_dis_threshold 0.01 \
#     --drop_rate 1.0 \
#     --near_n 5 \

# python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
#     --save_iterations 1000 2000 3000 4000 5000 10000 20000 30000 40000 45000 50000 \
#     --sample_pseudo_interval 1000000 --start_sample_pseudo 0 --end_sample_pseudo 0 \
#     --iterations $iterations \
#     --densify_until_iter $iterations \
#     --position_lr_max_steps $iterations \
#     --dataset_type llff \
#     --flow_type disxgrad \
#     --flow_checkpoint things \
#     --images images_8 \
#     --depth_weight 0 \
#     --scaling_lr 0.03 \
#     --split_num 4 \
#     --valid_dis_threshold 1.0 \
#     --drop_rate 1.0 \
#     --near_n 2 \

python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
    --save_iterations  30000 40000 \
    --iterations $iterations \
    --densify_until_iter $iterations \
    --position_lr_max_steps $iterations \
    --dataset_type $dataset_type \
    --flow_type $flow_type \
    --flow_checkpoint $flow_checkpoint \
    --images $images \
    --size_threshold $size_threshold \
    --prune_interval $prune_interval \
    --depth_weight $depth_weight \
    --scaling_lr $scaling_lr \
    --split_num $split_num \
    --valid_dis_threshold $valid_dis_threshold \
    --drop_rate $drop_rate \
    --near_n $near_n \


# set a larger "--error_tolerance" may get more smooth results in visualization

            
# python render.py --source_path $dataset  --model_path  $workspace --iteration 1000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 2000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 3000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 4000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 5000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 6000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 7000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 8000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 9000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 10000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 11000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 12000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 13000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 14000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 15000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 16000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 17000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 18000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 19000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 20000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 21000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 22000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 23000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 24000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 25000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 26000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 27000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 28000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 29000 --render_depth
python render.py --source_path $dataset  --model_path  $workspace --iteration 30000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 35000 --render_depth
python render.py --source_path $dataset  --model_path  $workspace --iteration 40000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 45000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 50000 --render_depth


python metrics.py --source_path $dataset --model_path $workspace 

done
done

#bash run_llff_mvs_confidence.sh 0