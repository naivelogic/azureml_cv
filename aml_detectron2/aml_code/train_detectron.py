
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from azureml.core import Run
import argparse
import os
import torch
import detectron2
from detectron2.data import MetadataCatalog, build_detection_train_loader

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

print(f'folder paths (one level up): \n{os.listdir("../")}')
print(f'\ndirectory listing (currnt): \n{os.listdir("./")}')


print(f'\nDetectron2 location: {detectron2.__file__}')
print(f'\nTorch location: {torch.__file__}')

# dataset object from the run
run = Run.get_context()
print(">>>>> RUN CONTEXT <<<<<<")
print(run)


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument("--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER,)
    return parser.parse_args(in_args)




def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args['config_file'])
    cfg.merge_from_list(args['opts'])
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    ap.add_argument('--img-folder', type=str, dest='img_folder', help='data folder mounting point')
    ap.add_argument('--masks-folder', type=str, dest='masks_folder', help='data folder mounting point')
    ap.add_argument('--output-folder', type=str, dest='output_folder', help='trained model folder mounting point')
    ap.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    ap.add_argument('--num-gpus', type=int, default=1, dest='num_gpus', help='number of gpus *per machine')
    ap.add_argument("--num-machines", type=int, default=1,dest='num_machines', help="total number of machines")
    ap.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",dest='opts',default=[],nargs=argparse.REMAINDER,)
    #ap.add_argument('--resume', type=str, dest='resume', help='TBD Description')
    ### from detectron2.engine import default_argument_parser, default_setup
    ### [--config-file FILE] [--resume] [--eval-only] [--num-gpus NUM_GPUS] [--num-machines NUM_MACHINES] [--machine-rank MACHINE_RANK] [--dist-url DIST_URL]
    #ap.add_argument('--resume', type=str, dest='resume', help='TBD Description')

    #ap.add_argument("-w", "--weights", help="optional path to pretrained weights")
    #ap.add_argument("-m", "--mode", help="train or investigate")
    args = vars(ap.parse_args())
    print("#################################################")
    print("All Aguments: \n", args)

    DATA_FOLDER = args["data_folder"]
    IMG_PATHS = args["img_folder"]
    MASKS_PATHS = args["masks_folder"]
    TRAIN_CONFIG = args["config_file"]
    OUTPUT_PATHS = args["output_folder"]
    print("#################################################")
    print(f'Argument Summary')
    print(f'Data folder: {DATA_FOLDER}\nImage Folder: {IMG_PATHS}\nMask Folder: {MASKS_PATHS}\nTraining config yml: {TRAIN_CONFIG}')

    print("#################################################")
    print(f'\ndirectory listing (MASKS_PATHS): \n{os.listdir(MASKS_PATHS)}')

    from detectron2.data.datasets import register_coco_instances
    TRAIN_PATH = os.path.join(MASKS_PATHS, 'Part_C_train_coco_annotations.json')
    TEST_PATH = os.path.join(MASKS_PATHS, 'dev_test_val/Part_C_test_testsplit_coco_annotations.json')
    VAL_PATH = os.path.join(MASKS_PATHS, 'dev_test_val/Part_C_val_testsplit_coco_annotations.json')
    #VAL_PATH = os.path.join(MASKS_PATHS, 'Part_B_test_coco_annotations.json')
    #TEST_PATH = os.path.join(MASKS_PATHS, 'Part_B_test_coco_annotations.json')

    register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , IMG_PATHS)
    register_coco_instances(f"custom_dataset_test", {}, TEST_PATH, IMG_PATHS)
    register_coco_instances(f"custom_dataset_val", {}, VAL_PATH, IMG_PATHS)
 

    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # only segmentation and bounding boxes

    # MODEL
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SEED = 42
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (Part_A)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 128 (default: 512)

    # DATASET
    cfg.DATASETS.TRAIN = ("custom_dataset_train",)
    cfg.DATASETS.TEST = ("custom_dataset_test",)   
    #cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.TEST.EVAL_PERIOD = 60
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 250
    

    # SOLVER
    cfg.SOLVER.MAX_ITER = 1000 #300
    cfg.SOLVER.IMS_PER_BATCH = 5 #2
    cfg.SOLVER.BASE_LR = 0.00025 #0.02 #0.002 0.00025
    cfg.SOLVER.WARMUP_ITERS = int(0.5 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / (cfg.SOLVER.WARMUP_ITERS + 1)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    

    cfg.OUTPUT_DIR= OUTPUT_PATHS 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
