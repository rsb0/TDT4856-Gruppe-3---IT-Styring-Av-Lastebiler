"""
    Simple starter template for the fuel price server
"""
from flask import Flask, request, redirect, jsonify, json
import os
import uuid
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from werkzeug.utils import secure_filename
from datetime import date

app = Flask(__name__)
app.debug = True # only for development!

# Key vault init
# key_vault_name = os.environ.get("KEY_VAULT_NAME")
# KVUri = "https://" + key_vault_name + ".vault.azure.net"
# credential = DefaultAzureCredential()
# client = SecretClient(vault_url=KVUri, credential=credential)
# secret_name = os.environ.get("SECRET_NAME")
# retrieved_secret = client.get_secret(secret_name)

retrieved_secret = "TABLE_STORAGE_KEY"

# account_name = os.environ.get("DB_ACCOUNT_NAME")
# table_name = os.environ.get("DB_TABLE_NAME")
# blob_container_name = "images"

# Table service init
table_service = TableService(account_name="fuelpricestorage", account_key=retrieved_secret)
table_name = "prices"

# Blob init
blob_connection_string = "BLOB_CONNECTION_STRING"
blob_container_name = os.environ.get("BLOB_CONTAINER_NAME")

#TODO: Get connection string for blob and secret for table storage from Key Vault!

# Default route returns empty string
@app.route("/")
def index():
    return "Official API for the amazing fuel price application ®"

# Get all fuel prices based on location
@app.route("/price/<string:area>")
def getPrices(area):
    all_prices = table_service.query_entities(table_name, filter=("PartitionKey eq '" + area + "'"))
    price_list = []
    for price in all_prices:
        price_list.append(price)
    return jsonify(price_list)

# Get a fuel price based on location and id
@app.route("/price/<string:area>/<string:id>")
def getPricesById(area, id):
    if(is_valid_uuid(id)):
        try:
            entry = table_service.get_entity(table_name, area, id)
            return entry
        except Exception:
            return "No value for this ID found"
        return jsonify(entry)

# Get fuel prices based on location and coordinates
@app.route("/price/<string:area>/coordinates/<string:coordinates>")
def getPricesByCoordinates(area, coordinates):
    all_prices = table_service.query_entities(table_name, filter=("PartitionKey eq '" + area + "'"))
    
    # Sort out prices where coordinates is not matching entries in db
    relevant_prices = []
    for entry in all_prices:
        if (entry.location == coordinates):
            relevant_prices.append(entry)
    return jsonify(relevant_prices)

# Insert new fuel price to the database. Only for dev purpose!
@app.route("/input/price", methods=["POST"])
def input_price():
    json_content = request.get_json().get("new_prices")
    if json_content is None:
       return "No JSON content detected"
    
    error = False
    for val in json_content: # Loop through new_prices and add to database
        entry = Entity()
        try:
            entry.PartitionKey = val["county"]
            entry.RowKey = str(uuid.uuid4()) # Generate new random UUID
            entry.price = val["price"]
            entry.location = val["location"]
            if (val["fueltype"] == "diesel" or val["fueltype"] == "gasoline"):
                entry.fueltype = val["fueltype"]
            else:
                entry.fueltype = "unknown"
            table_service.insert_entity(table_name, entry)
        except AttributeError:
            print("Error trying to parse JSON object: " + val)
            error = True
    if error:
        "Something went wrong. Try check your syntax"
    return "Entries inserted succesfully!"

# Process picture to extract price
@app.route("/input/image", methods=["POST"])
def input_picture():
    if request.files['img']:
        img = request.files['img']
        # Create unambigous image file name
        img_name = secure_filename(str(date.today()) + "-" + str(uuid.uuid4()))

        # Create connection to blob storage
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=img_name)
        
        # Upload image to blob
        blob_client.upload_blob(img)
        return "Image succesfully uploaded to blob storage"
    else:
    	return "Where is the image?"

# Helper methods
def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

# Error handling
@app.errorhandler(404)
def page_not_found(error):
    return "Oops, endpoint not found - 404"

@app.errorhandler(500)
def bad_request500(error):
    return "Oops, internal server error - 500"

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=80)