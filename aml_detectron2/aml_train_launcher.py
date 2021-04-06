# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys, os
#sys.path.append("../") # go to parent dir

from azureml.core import Workspace, Experiment, VERSION, ContainerRegistry
from azureml.core.compute import ComputeTarget
from azureml.train.estimator import Estimator
#from azureml.widgets import RunDetails

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Validate Azure ML SDK installation and get version number for debugging purposes
print("SDK version:", VERSION)

#%%
# load workspace
ws = Workspace.from_config(path="../aml_configs/config.json")
print('Workspace name: ' + ws.name, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

from azureml.core import Datastore
blob_datastore = Datastore.register_azure_blob_container(workspace=ws, 
                                                     datastore_name='<<< TBD  >>>>>', 
                                                     container_name="<<< TBD  >>>>>", 
                                                     account_name="<<< TBD  >>>>>",
                                                     account_key='<<< TBD  >>>>>',
                                                     create_if_not_exists=False)



#get named datastore from current workspace
datastore = Datastore.get(ws, datastore_name='project_zero')
#print(blob_datastore)


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
#cluster_name = "StandardNC12"
cluster_name =  'gpucluster'

compute_target = ComputeTarget(workspace=ws, name=cluster_name)
print('Found existing compute target')   
print(compute_target.get_status().serialize())


# Environment Setup
#%%
#https://github.com/liupeirong/tensorflow_objectdetection_azureml/blob/master/aml_train/aml-train.ipynb
from azureml.core.runconfig import RunConfiguration, DataReferenceConfiguration
proj_root = "project_zero/ds1"


dr = DataReferenceConfiguration(datastore_name=datastore.name, 
                                path_on_datastore=proj_root,
#                                path_on_compute='/datastore', path_on_compute doesn't work with mount
                                overwrite=True)

#%%
container_registry = ContainerRegistry()
container_registry.address = '<<< TBD  >>>>>'
container_registry.username = '<<< TBD  >>>>>'
container_registry.password = '<<< TBD  >>>>>'

run_config = RunConfiguration()
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image='detectron:1'
run_config.environment.docker.base_image_registry=container_registry
run_config.data_references = {datastore.name: dr}
run_config.environment.python.user_managed_dependencies=True  # use your own installed packages instead of an AML created Conda env
run_config.target = compute_target # specify the compute target; obscure error message: `docker image` cannot run

#%%
from azureml.core import Experiment
experiment_name = 'detectron_demo'
exp = Experiment(workspace=ws, name=experiment_name)

IMG_PATH = ""
ANNOTATION_PATH = ""
base_mount = datastore.path(proj_root).as_mount()
img_path = os.path.join(str(base_mount), IMG_PATH)
annotations_path = os.path.join(str(base_mount), ANNOTATION_PATH)

#%%
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': base_mount,
    '--img-folder': img_path,
    '--masks-folder': annotations_path,
    '--config-file': 'configs/R_50_1x.yaml'
}

est = Estimator(source_directory="./aml_code/",
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script="train_detectron.py",
                    use_docker=True,
                    image_registry_details=container_registry,
                    user_managed=True,
                    custom_docker_image='detectron:1', #notice this is short name, different from ScriptRun
                    inputs=[base_mount]) #tell the system to mount, or if the script params contain ds.mount(), it will mount without this

#%%
run = exp.submit(est)
print(f"Run Details: {run.get_details()['runId']}")

print(f'Run Status: {run.get_status()}')
#%%
run.tag('platform','detectron2') 