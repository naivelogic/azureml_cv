
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.events import EventStorage


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
    # https://github.com/facebookresearch/detectron2/blob/master/tools/visualize_data.py

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
    ap.add_argument('--output-folder', type=str, dest='output_folder', help='trained model folder mounting point')
    ap.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    ## new
    ap.add_argument('--train_img_dir', type=str, dest='train_img_dir', help='training image data folder mounting point')
    ap.add_argument('--train_coco_json', type=str, dest='train_coco_json', help='training data annotation json file')
    ap.add_argument('--val_img_dir', type=str, dest='val_img_dir', help='training image data folder mounting point')
    ap.add_argument('--val_coco_json', type=str, dest='val_coco_json', help='training data annotation json file')

    ap.add_argument('--num-gpus', type=int, default=4, dest='num_gpus', help='number of gpus *per machine')
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
    TRAIN_CONFIG = args["config_file"]
    OUTPUT_PATHS = args["output_folder"]

    TRAIN_IMG_DIR = args["train_img_dir"]
    TRAIN_COCO_JSON = args["train_coco_json"]
    VAL_IMG_DIR= args["val_img_dir"]
    VAL_COCO_JSON = args["val_coco_json"]


    from detectron2.data.datasets import register_coco_instances
    TRAIN_PATH = os.path.join(DATA_FOLDER, TRAIN_COCO_JSON)
    TRAIN_IMG_PATH= os.path.join(DATA_FOLDER, TRAIN_IMG_DIR)
    register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , TRAIN_IMG_PATH)

    VAL_PATH = os.path.join(DATA_FOLDER, VAL_COCO_JSON)
    VAL_IMG_PATH= os.path.join(DATA_FOLDER, VAL_IMG_DIR)
    register_coco_instances(f"custom_dataset_val", {},VAL_PATH , VAL_IMG_PATH) 

    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(TRAIN_CONFIG) #TRAINING_MODEL_YAML
    cfg.OUTPUT_DIR= OUTPUT_PATHS 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) #lets just check our output dir exists
    
    cfg.freeze()                    # make the configuration unchangeable during the training process
    with open(cfg.OUTPUT_DIR + "/config.yml", "w") as f:
        f.write(cfg.dump())
        
    
    from detectron2.evaluation import COCOEvaluator
    class Trainer2(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)


    trainer = Trainer2(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

        
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    #cfg.DATASETS.TEST = ("custom_dataset_val", )
    #predictor = DefaultPredictor(cfg)

    print("#################################################")
    print("Predicted model path: ", cfg.MODEL.WEIGHTS)
    print("#################################################")
