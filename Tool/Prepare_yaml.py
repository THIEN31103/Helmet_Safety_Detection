import  yaml
import  os

dataset_info = {
    'train' : './train/images',
    'val'   : './val/images',
    'path'  : os.path.abspath('./human_detection_dataset'),
    'nc'    : 1,
    'names' : ['Human']
}

yaml_filepath = './human_detection_dataset/data.yaml'
with open(yaml_filepath, 'w') as f:
    doc = yaml.dump(
        dataset_info,
        f,
        default_flow_style=None,
        sort_keys=False
    )