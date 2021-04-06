# Azure ML

# AML Demo Development Status and Content

Azure Machine Learning Service Demo

* [ ] [setting up environment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment)
* [ ] connect data set
* [ ] set up environment
* [ ] training 
  * [x] Detectron2
  * [x] Yolact
* [ ] model evaluaiton
* [ ] model registration 
* [ ] model deployment and operationation 


### Installing AML

```
conda create -n azureml -y Python=3.7
source activate azureml
pip install --upgrade azureml-sdk[notebooks,contrib] 
conda install ipywidgets
jupyter nbextension install --py --user azureml.widgets
jupyter nbextension enable azureml.widgets --user --py

## if using Jupyter Notebooks create custom jupyter kernel for WaterWaste
python -m ipykernel install --user --name=azureml
pip install yacs
```




## Running with Docker

### Quick Start

- loging to registry `docker login -u [USERNAME] -p [PASWORD] [USERNAME].azurecr.io`
- pull docker `docker pull [USERNAME].azurecr.io/yolact:1`
- run `docker run --gpus=all --shm-size 8G -v /home/$USER/mnt/project_zero/:/mnt/ -it yolacter`


__Key Requirement:__ [NVIDIA Driver Installation](https://github.com/NVIDIA/nvidia-docker) 

#### Usage

__Train__

TODO:

__Evaluate__

TODO:


__Prediction/Inference__

TODO:

## Build Container

```
docker login 
docker build . -f Docker/yolact.Dockerfile -t [USERNAME].azurecr.io/yolact:1
docker push [USERNAME].azurecr.io/yolact:1
```
