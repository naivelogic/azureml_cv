## Demo Dataset Prep

__Demo dataset for AzureML is coco_val2017__ 

More information refer to: https://cocodataset.org/#download

1. download `coco_val2017` on a mounted azure blob container: `sh download_coco_val17.sh`
2. split coco annotation into train/val split using the `coco_val2017` dataset *(just for demo purposes)*: `sh coco_val17_split.sh` (update script arg)
