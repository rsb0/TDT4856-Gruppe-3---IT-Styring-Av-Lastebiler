"""
    Simple starter template for the fuel price server
"""
from flask import Flask, request, redirect, jsonify, json
import os
import uuid
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

app = Flask(__name__)
app.debug = True # only for development!

# Key vault init
"""
key_vault_name = os.environ["KEY_VAULT_NAME"]
KVUri = "https://" + key_vault_name + ".vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)
secretName = os.environ["SECRET_NAME"]
retrieved_secret = client.get_secret(secretName)"""
retrieved_secret = "STORAGE ACCOUNT KEY"

# Table service init
table_service = TableService(account_name=os.environ.get("DB_ACCOUNT_NAME"), account_key=retrieved_secret)
table_name = os.environ.get("DB_TABLE_NAME")

# Default route returns empty string
@app.route("/")
def index():
    return "working yey"

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
@app.route("/input", methods=["POST"])
def input():
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