ANN=/mnt/omreast_users/phhale/open_ds/demo/coco/annotations/instances_val2017.json
TRAIN_JSON=/mnt/omreast_users/phhale/open_ds/demo/coco/annotations/train_instances_val2017.json
VAL_JSON=/mnt/omreast_users/phhale/open_ds/demo/coco/annotations/val_instances_val2017.json

python cocosplit.py --having-annotations -s 0.8 ${ANN} ${TRAIN_JSON} ${VAL_JSON}

#Saved 3961 entries in /mnt/omreast_users/phhale/open_ds/demo/coco/annotations/train_instances_val2017.json and 991 in /mnt/omreast_users/phhale/open_ds/demo/coco/annotations/val_instances_val2017.json