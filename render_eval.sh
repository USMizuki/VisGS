# names=(Ballroom Barn Church Family Francis Horse Ignatius Museum)
names=(Horse)
#n_views=(2 3 4 6 9)
n_views=(3 6 12)
export CUDA_VISIBLE_DEVICES=$1

# iterations=40000
# dataset_type=tat
# flow_type=disxgrad 
# flow_checkpoint=sintel 
# images=images
# size_threshold=-1
# prune_interval=-1 
# depth_weight=0
# scaling_lr=0.03
# split_num=2 
# valid_dis_threshold=0.5
# drop_rate=0.3
# near_n=2
# exp_name=${dataset_type}_${images}_${flow_type}_${flow_checkpoint}_scalelr${scaling_lr}_depth${depth_weight}_near${near_n}_size${size_threshold}_valid${valid_dis_threshold}_drop${drop_rate}_N${split_num}
exp_name=tat_flow_test_scalelr0.03_pretrain0_nosize_noinpaint_flow_near2_disxgrad_valid1.0_oriprune_N4_drop0.3

for name in "${names[@]}";
do

for n in "${n_views[@]}";
do

dataset=dataset/Tanks/$name
workspace=output/exp/tat/$exp_name/$name/${n}_views

# python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
#     --save_iterations 1000 2000 3000 4000 5000 10000 20000 30000 40000 45000 50000 \
#     --iterations $iterations \
#     --densify_until_iter $iterations \
#     --position_lr_max_steps $iterations \
#     --dataset_type $dataset_type \
#     --flow_type $flow_type \
#     --flow_checkpoint $flow_checkpoint \
#     --images $images \
#     --size_threshold $size_threshold \
#     --prune_interval $prune_interval \
#     --depth_weight $depth_weight \
#     --scaling_lr $scaling_lr \
#     --split_num $split_num \
#     --valid_dis_threshold $valid_dis_threshold \
#     --drop_rate $drop_rate \
#     --near_n $near_n \


# set a larger "--error_tolerance" may get more smooth results in visualization

            
# python render.py --source_path $dataset  --model_path  $workspace --iteration 1000 --render_depth --render_eval
# python render.py --source_path $dataset  --model_path  $workspace --iteration 2000 --render_depth --render_eval
# python render.py --source_path $dataset  --model_path  $workspace --iteration 3000 --render_depth --render_eval
python render.py --source_path $dataset  --model_path  $workspace --iteration 4000 --render_depth --render_eval
# python render.py --source_path $dataset  --model_path  $workspace --iteration 5000 --render_depth --render_eval
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 6000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 7000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 8000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 9000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 10000 --render_depth --render_eval
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 11000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 12000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 13000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 14000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 15000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 16000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 17000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 18000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 19000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 20000 --render_depth --render_eval
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 21000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 22000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 23000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 24000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 25000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 26000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 27000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 28000 --render_depth
# # python render.py --source_path $dataset  --model_path  $workspace --iteration 29000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 30000 --render_depth --render_eval
# python render.py --source_path $dataset  --model_path  $workspace --iteration 35000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 40000 --render_depth --render_eval
# python render.py --source_path $dataset  --model_path  $workspace --iteration 45000 --render_depth
# python render.py --source_path $dataset  --model_path  $workspace --iteration 50000 --render_depth


# python metrics.py --source_path $dataset --model_path $workspace 

done
done

#bash run_llff_mvs_confidence.sh 0