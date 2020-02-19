import uuid, os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Key vault init
# key_vault_name = os.environ.get("KEY_VAULT_NAME")
# kv_uri = "https://" + key_vault_name + ".vault.azure.net"
# credential = DefaultAzureCredential()
# client = SecretClient(vault_url=kv_uri, credential=credential)
# secret_name_storage_key = os.environ.get("SECRET_NAME_STORAGE_KEY")
# secret_name_blob_string = os.environ.get("SECRET_NAME_BLOB_STRING")

def get_table_key():
    # return client.get_secret(secret_name_storage_key)
    return "NEXqWYKPcrBDbFVGlQodPjll8jzVOJStPsZISiZg7ST+Q/oX/B2ooUMChn9ouD9O7bcJHEnKUx8VAWFrInPGyQ=="
    
def get_blob_connection_string():
    # return client.get_secret(secret_name_blob_string)
    return "DefaultEndpointsProtocol=https;AccountName=fuelpriceblob;AccountKey=/01XgCnjTN2UFLWZA+lL06Xa/6HwfuhtGu6K3atWPw9Qr7C2TScpKT6b/HjxcX8fu0AE2VVRl6hX7LkoV74sLg==;EndpointSuffix=core.windows.net"

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False