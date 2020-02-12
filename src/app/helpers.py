import uuid

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

# TODO: Implement key vault helpers
# Key vault init
# key_vault_name = os.environ.get("KEY_VAULT_NAME")
# KVUri = "https://" + key_vault_name + ".vault.azure.net"
# credential = DefaultAzureCredential()
# client = SecretClient(vault_url=KVUri, credential=credential)
# secret_name = os.environ.get("SECRET_NAME")
# retrieved_secret = client.get_secret(secret_name)

def get_table_key():
    return ""

def get_blob_connection_string():
    return ""
    