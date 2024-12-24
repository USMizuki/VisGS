names=(fern flower fortress horns leaves orchids room trex)
#names=(orchids)
#n_views=(2 3 4 6 9)
n_views=(3 4)

for name in "${names[@]}";
do

for n in "${n_views[@]}";
do
dataset=dataset/nerf_llff_data/$name
exp=flow_test_scalelr0.1_pretrain_5000_prune200_noinpaint_flow1
workspace=output/$exp/$name/${n}_views

mkdir -p output/bak/$exp/$name/${n}_views
cp $workspace/results.json output/bak/$exp/$name/${n}_views

done

done