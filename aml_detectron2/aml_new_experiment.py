import os
from azureml.core import Experiment, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
import experiment_cfg as exp_cfg

## Load AzureML Secrets Configs 
from env_manager import EnvManager as E
ws_cfg = E()

## Set up AzureML Workspace
## (see below if running into authe issues)
## if you are already logged in az login --use-device-code & az account set --subscription [subscription name]
## need to do this bc i have multiple azure tenants and multiple azureml subscriptions 
## refer: https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb
def get_workspace(subscription_id:str, workspace_name:str, resource_group:str, tenant_id:str):
    interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id) #, force=True)
    return Workspace(subscription_id, resource_group, workspace_name, auth=interactive_auth)

ws = get_workspace(subscription_id=ws_cfg.subscription_id,
                   workspace_name=ws_cfg.workspace_name,
                   resource_group=ws_cfg.resource_group,
                   tenant_id=ws_cfg.tenant_id)


print('Workspace name: ' + ws.name, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')



## Setup Compute
from azureml.core.compute import ComputeTarget
cluster_name =  exp_cfg.CLUSTER_NAME 
#compute_target = ComputeTarget(workspace=ws, name=cluster_name) # standard

## running for the first time need to create the compute target
from azureml.core.compute_target import ComputeTargetException
from azureml.core.compute import AmlCompute

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC12', max_nodes=4)
    #compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6',min_nodes=1, max_nodes=4, location='eastus')

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Use the 'status' property to get a detailed status for the current cluster. 
#print(compute_target.status.serialize())



## Mount DataStore
from azureml.core import Datastore

## running for the first time to mound datastor (blob container)
#Datastore.register_azure_blob_container(workspace=ws, 
#                                                     datastore_name=ws_cfg.blob_datastore_name, 
#                                                     container_name=ws_cfg.blob_container_name, 
#                                                     account_name=ws_cfg.blob_account_name,
#                                                     account_key=ws_cfg.blob_account_key,
#                                                    create_if_not_exists=False)

datastore = Datastore.get(ws, datastore_name=ws_cfg.blob_datastore_name)

from azureml.core.runconfig import RunConfiguration, DataReferenceConfiguration
dr = DataReferenceConfiguration(datastore_name=datastore.name, path_on_datastore=None,overwrite=True)


## Configure Container Registry
from azureml.core import ContainerRegistry
def get_container_registry(ws_cfg):
    container_registry = ContainerRegistry()
    container_registry.address = ws_cfg.registry_login
    container_registry.username = ws_cfg.registry_username 
    container_registry.password = ws_cfg.registry_key
    return container_registry

container_registry = get_container_registry(ws_cfg)

## Run configuration
run_config = RunConfiguration()
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image=exp_cfg.DOCKER_IMG 
run_config.environment.docker.base_image_registry=container_registry
run_config.data_references = {datastore.name: dr}
run_config.environment.python.user_managed_dependencies=True  # use your own installed packages instead of an AML created Conda env
run_config.target = compute_target # specify the compute target; obscure error message: `docker image` cannot run


### SET UP EXPERIMENT
from azureml.core import Experiment
experiment_name = exp_cfg.EXPERIMENT_NAME 
exp = Experiment(workspace=ws, name=experiment_name)


## Training Input DIR
base_mount = datastore.as_mount()
output_path = os.path.join(str(base_mount), exp_cfg.OUTPUT_DIR) 


## Training Scrip Configs
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': base_mount,
    '--output-folder':output_path,
    '--config-file': exp_cfg.TRAINING_MODEL_YAML, 
    '--train_img_dir': os.path.join(str(base_mount), exp_cfg.TRAIN_IMG_DIR),
    '--train_coco_json': os.path.join(str(base_mount), exp_cfg.TRAIN_COCO_JSON),
    '--val_img_dir': os.path.join(str(base_mount), exp_cfg.VAL_IMG_DIR),
    '--val_coco_json': os.path.join(str(base_mount), exp_cfg.VAL_COCO_JSON),

}

est = Estimator(source_directory=exp_cfg.AML_TRAINING_CODE_DIR,
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script="aml_detectron_train.py",
                    use_docker=True,
                    image_registry_details=container_registry,
                    user_managed=True,
                    custom_docker_image=exp_cfg.DOCKER_IMG,
                    inputs=[base_mount]) 


#Run Exper
run = exp.submit(est)
print(f"Run Details: {run.get_details()['runId']}")
print(f'Run Status: {run.get_status()}')
print(f'Run Properties: {run.get_properties()}')
