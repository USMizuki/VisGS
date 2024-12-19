# names=(chair drums ficus hotdog lego materials mic ship)
names=(bicycle bonsai counter garden kitchen room stump)
# names=(mic)

for name in "${names[@]}";
do

dataset=/mnt/lab/zyl/models/NexusGS-anneal/dataset/mip360_2/$name
python convert.py -s $dataset --camera PINHOLE

done