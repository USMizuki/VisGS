names=(bicycle bonsai counter garden kitchen room stump)
# names=(room stump)
#names=(orchids)
#n_views=(2 3 4 6 9)
n_views=(24)
export CUDA_VISIBLE_DEVICES=$1

iterations=1
dataset_type=mip360
flow_type=grad
flow_checkpoint=things 
images=images_8
size_threshold=-1
prune_interval=-1 
depth_weight=0
scaling_lr=0.005
split_num=2
valid_dis_threshold=0.1
drop_rate=0.05
near_n=5
exp_name=${dataset_type}_${images}_${flow_type}_${flow_checkpoint}_scalelr${scaling_lr}_depth${depth_weight}_near${near_n}_size${size_threshold}_valid${valid_dis_threshold}_drop${drop_rate}_N${split_num}


for name in "${names[@]}";
do

for n in "${n_views[@]}";
do

dataset=dataset/mip360/$name
workspace=output/exp/mip360/$exp_name/$name/${n}_views

# iterations=10000
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
#     --save_iterations 1000 2000 3000 4000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 \
#     --sample_pseudo_interval 1000000 --start_sample_pseudo 0 --end_sample_pseudo 0 \
#     --iterations $iterations \
#     --iterations_pre 0 \
#     --densify_until_iter $iterations \
#     --position_lr_max_steps $iterations \
#     --depth_weight 0 \
#     --diffusion_inpaint_iter -1 \
#     --scaling_lr 0.005 \
#     --size_threshol 20 \
#     --prune_depth_threshold 5 \
#     --prune_depth_scale 9 \
#     --prune_interval 10000000 \
#     --split_num 4 \
#     --valid_dis_threshold 0.5 \
#     --drop_rate 0.3

# python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
#     --save_iterations 1000 2000 3000 4000 5000 10000 20000 30000 40000 45000 50000 \
#     --sample_pseudo_interval 1000000 --start_sample_pseudo 0 --end_sample_pseudo 0 \
#     --iterations $iterations \
#     --densify_until_iter $iterations \
#     --position_lr_max_steps $iterations \
#     --dataset_type mip360 \
#     --flow_type disxgrad \
#     --flow_checkpoint things \
#     --images images_8 \
#     --depth_weight 0.0001 \
#     --scaling_lr 0.03 \
#     --split_num 2 \
#     --valid_dis_threshold 0.05 \
#     --drop_rate 0.3 \
#     --near_n 5 \

python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
    --save_iterations 1000 2000 3000 4000 5000 10000 20000 30000 40000 45000 50000 \
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

done
done

#bash run_llff_mvs_confidence.sh 0