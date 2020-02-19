import os
from flask import jsonify
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from helpers import is_valid_uuid


print("Environent variables: ", os.environ)
# Blob service init
blob_container_name = os.environ.get("BLOB_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("BLOB_CONNECTION_STR"))

# Table service init
table_service = TableService(os.environ.get("DB_ACCOUNT_NAME"), os.environ.get("TABLE_KEY"))
table_name = os.environ.get("DB_TABLE_NAME")

""" 
Uncomment code and insert keys/connection string below for dev
"""
# blob_container_name = "images"
# blob_service_client = BlobServiceClient.from_connection_string("INSERT_BLOB_CONNECTION_STRING_HERE")

# table_service = TableService("fuelpricestorage", "INSERT_TABLE_KEY_HERE")
# table_name = "prices" 

def get_prices_by_area(area):
    all_prices = table_service.query_entities(table_name, filter=("PartitionKey eq '" + area + "'"))
    price_list = []
    for price in all_prices:
        price_list.append(price)
    return jsonify(price_list)

def get_prices_by_key(area, key):
    if(is_valid_uuid(key)): # Key is id
        try:
            entry = table_service.get_entity(table_name, area, key)
            return entry
        except Exception:
            return "No value for this ID found"
        return jsonify(entry)
    else: # Key is coordinate
        all_prices = table_service.query_entities(table_name, filter=("PartitionKey eq '" + area + "'"))
    
        # Sort out prices where coordinates is not matching entries in db
        relevant_prices = []
        for entry in all_prices:
            if (entry.location == id):
                relevant_prices.append(entry)
        return jsonify(relevant_prices)