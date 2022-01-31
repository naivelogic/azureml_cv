### Project organization

```
├── README.md             <- Top-level README.
├── .env.example          <- RENAME to .env and update as appropriate
├── requirements_detectron2.txt      <- detectron2 software dependencies
├── Dockerfile            <- detectron dockerfile 
├── aml_new_experiment.py <- used to launch a new AML training 
├── env_manager.py        <- used to launch a new AML training 
├── experiment_cfg.py     <- used to launch a new AML training 
|
├── aml_code
│   ├── aml_detectron_train.py           <- aml training script
│   ├── configs           <- folder that contains DNN configurations
|
├── local
│   ├── local_train.py    <- running detectron2 train locally
│   ├── test_train.py     <- inference testing detectron2 image folder
│   ├── tron_util.py      <- detectron2 training utils
```

# Quick Start


## Build and pushing Docker container to AzureML

- loging to registry `docker login -u [USERNAME] -p [PASWORD] [USERNAME].azurecr.io`
- build docker `docker build . -f Dockerfile -t [USERNAME].azurecr.io/dev:detectron`
- push container to Azure Container Registery `docker push [USERNAME].azurecr.io/dev:detectron`
- run `nvidia-docker run -it -v /home/$USER/mnt/omreast_users/:/mnt/ -it [USERNAME].azurecr.io/dev:detectron bash`

<details>
 <summary>Demo Detectron2 - Building and Running Docker Container Locally</summary>

First, build the docker container to run detectron2 and azureml packages
```sh
docker build . -f Dockerfile -t aml_detectron
nvidia-docker run -it -v /mnt/omreast_users/:/mnt/omreast_users/ -v /home/${USER}/azureml_cv/aml_detectron2/:/home/${USER}/azureml_cv/aml_detectron2/ aml_detectron bash
cd /home/${USER}/azureml_cv/aml_detectron2/
python local_train.py
```

Once satisfied with the `Dockerfile` and packages, you can now assign the docker image tags with the [Azure Container Registery](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-troubleshoot-login):

```sh
# can get login infor in the Azure Container Registery 
docker login -u [USERNAME] [USERNAME].azurecr.io

# re tag the docker image to the Azure container registry
docker tag aml_detectron:latest [USERNAME].azurecr.io/dev:detectron

# (optional) or you can just rebuild it from file
docker build . -f Docker/Dockerfile -t [USERNAME].azurecr.io/dev:detectron
docker push [USERNAME].azurecr.io/dev:detectron
```

</details>



## Train Model

activate azureml conda environment: `conda activate azureml`

Before lauching an AML experiment update: `experiment_cfg.py` to include the latest training configurations. This includes updating the `CLUSTER_NAME`, `EXPERIMENT_NAME`, `DOCKER_IMG` among other training script path and the blob dir to the images and annotation. 

(hard code update) _TODO - need to improve_: additionally, need to update the model weight parameter in `aml_detectron2/aml_code/aml_detectron_train.py` e.g., `cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")`


After the training parameters (`experiment_cfg.py`), model config (`aml_detectron2/aml_code/configs/`) and aml experiment parameters (`.env` and `experiment_cfg.py`), the training experiment can be launched by running: `python aml_new_experiment.py`
> note: may need to login first `az login --use-device-code` & `az account set --subscription SDE01`

