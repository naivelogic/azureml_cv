import os
from dotenv import load_dotenv
from typing import Optional

class EnvManager:
    """
    Environment Manager loads the `.env` environment variable file that predefines the set of properties used in AzureML
    
    """
    
    load_dotenv()
    subscription_id: Optional[str] = os.environ.get("aml_subscription_id")
    resource_group: Optional[str] = os.environ.get("aml_resource_group")
    workspace_name: Optional[str] = os.environ.get("aml_workspace_name")

    tenant_id: Optional[str] = os.environ.get("tenant_id")
    subscription_name: Optional[str] = os.environ.get("subscription_name")
    location: Optional[str] = os.environ.get("aml_location")
    storage: Optional[str] = os.environ.get("aml_storage")
    
    # registry
    registry_name: Optional[str] = os.environ.get("registry_name")
    registry_key: Optional[str] = os.environ.get("registry_key")
    registry_login: Optional[str] = os.environ.get("registry_login")
    registry_username: Optional[str] = os.environ.get("registry_username")

    ## storage
    blob_datastore_name: Optional[str] = os.environ.get("blob_datastore_name")
    blob_container_name: Optional[str] = os.environ.get("blob_container_name")
    blob_account_name: Optional[str] = os.environ.get("blob_account_name")
    blob_account_key: Optional[str] = os.environ.get("blob_account_key")
