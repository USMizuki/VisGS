import os
import json

base_path = '/mnt/lab/zyl/models/FlowGS-final/dataset/nerf_synthetic'
for scene in ['chair', 'drums', 'ficus', 'hotdog',  'lego', 'materials', 'mic', 'ship']:
    os.chdir(os.path.join(base_path, scene))
    view_path = str(8) + '_views'
    # os.system('rm -r ' + view_path)
    # os.mkdir(view_path)
    os.system('rm -r input')
    os.mkdir('input')
    os.chdir(view_path)
    os.system('rm -r images')
    os.mkdir('images')
    # os.mkdir('images_4')
    # os.mkdir('images_8')
    transformsfile = os.path.join(base_path, scene, 'transforms_train.json')

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(base_path, scene, 'train', frame["file_path"] + '.png')
            img_name = os.path.split(frame["file_path"] + '.png')[-1]
            
            if idx in [2, 16, 26, 55, 73, 76, 86, 93]:
                os.system('cp ../train/' + img_name + '  images/train_' + img_name)
            os.system('cp ../train/' + img_name + '  ../input/train_' + img_name)
    
    transformsfile = os.path.join(base_path, scene, 'transforms_test.json')

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(base_path, scene, 'test', frame["file_path"] + '.png')
            img_name = os.path.split(frame["file_path"] + '.png')[-1]
            
            # if idx in [2, 16, 26, 55, 73, 76, 86, 93]:
            os.system('cp ../test/' + img_name + '  ../input/' + img_name)

