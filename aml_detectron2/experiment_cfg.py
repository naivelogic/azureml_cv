
# ### CORE CONFIGURATION OPTIONS ###
CLUSTER_NAME='gpucluster'
EXPERIMENT_NAME = 'demo_coco_val2017'
DOCKER_IMG = 'dev:detectron' 

## Training Config Input DIR
LOCAL_BASE_MNT='/mnt/omreast_users/' # same as base_mount = datastore.as_mount()
TRAIN_IMG_DIR="phhale/open_ds/demo/coco/val2017/"
TRAIN_COCO_JSON="phhale/open_ds/demo/coco/annotations/train_instances_val2017.json"
VAL_IMG_DIR=TRAIN_IMG_DIR
VAL_COCO_JSON="phhale/open_ds/demo/coco/annotations/val_instances_val2017.json"
OUTPUT_DIR= "phhale/open_ds/demo/coco/experiments/detectron2/013122_coco_val17_demo_frcnn_r50" 

### Training Script Params
TRAINING_MODEL_YAML="configs/frcnn_r50_fpn_coco_val17_demo.yaml"
AML_TRAINING_CODE_DIR='aml_code/'