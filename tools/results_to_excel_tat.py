import pandas as pd
import os

exp = "output/exp/tat/tat_images_disxgrad_sintel_scalelr0.03_depth0_near2_size-1_valid0.5_drop0.3_N2"

index = []
data = []
columns = []

# view_list = [3]
#view_list = [24]
view_list = [3, 6, 12]
# scene_list = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
scene_list = ['Ballroom', 'Barn', 'Church', 'Family',  'Francis', 'Horse', 'Ignatius', 'Museum']
#scene_list = ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']
save_list = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000]
# save_list = [5000, 10000]
#save_list = range(1000, 30000 + 1, 1000)
for i in save_list:
    columns.append('ours_' + str(i) + '_LPIPS')
    columns.append('ours_' + str(i) + '_PSNR')
    columns.append('ours_' + str(i) + '_SSIM')
    columns.append('')

for n in view_list:
    result_row = []
    for i in save_list:
        result_row.append(0.)
        result_row.append(0.)
        result_row.append(0.)
        result_row.append('')
    for scene in scene_list:
        json_path = os.path.join(exp, scene, str(n) + '_views', 'results.json')
        result = pd.read_json(json_path)

        data_row = []
        for idx, i in enumerate(save_list):
            name = 'ours_' + str(i)
            try:
                result_i = result['ours_' + str(i)]
                result_row[4*idx] += result_i['LPIPS'] / len(scene_list)
                result_row[4*idx+1] += result_i['PSNR'] / len(scene_list)
                result_row[4*idx+2] += result_i['SSIM'] / len(scene_list)
                data_row.append(result_i['LPIPS'])
                data_row.append(result_i['PSNR'])
                data_row.append(result_i['SSIM'])
                data_row.append('')
            except:
                continue
        
        index.append(scene + '_' + str(n) + '_views')
        data.append(data_row)
    index.append('')
    index.append('')
    data.append(result_row)
    data.append('')

pd.DataFrame(data=data, index=index, columns=columns).to_excel(exp + '_output.xlsx')

        
        
