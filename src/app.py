from flask import Flask, request, redirect, jsonify, json
import os
import uuid
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

app = Flask(__name__)
app.debug = True # only for development!

# Application config
app.config["DB_ACCOUNT_NAME"] = "fuelpricestorage"
app.config["DB_TABLE_NAME"] = "prices"

# Default route return empty string
@app.route("/")
def index():
    return ""

# Get a fuel price based on location and id
@app.route("/price/<string:location>/<string:id>")
def getPricesById(location, id):
    # TODO: Extract table_service creation to init method
    # TODO: Create priceEntity class?
    if(is_valid_uuid(id)):
        table_service = TableService(account_name=app.config["DB_ACCOUNT_NAME"], account_key=os.environ.get("ACCOUNT_KEY"))
        try:
            entry = table_service.get_entity(app.config["DB_TABLE_NAME"], location, id)
            return entry
        except Exception:
            return "No value for this ID found"
        return jsonify(entry)

# Get fuel prices based on location and coordinates
@app.route("/price/<string:location>/<string:coordinates>")
def getPricesByCoordinates(location, coordinates):
    table_service = TableService(account_name=app.config["DB_ACCOUNT_NAME"], account_key=os.environ.get("ACCOUNT_KEY"))
    all_prices = table_service.query_entities(app.config["DB_TABLE_NAME"], filter=("PartitionKey eq '" + location + "'"))
    
    # Sort out prices where coordinates is not matching
    relevant_prices = []
    for price in all_prices:
        if (price.coordinates == coordinates):
            relevant_prices.append(price)
    return jsonify(relevant_prices)

# Get all fuel prices based on location
@app.route("/price/<string:location>")
def getPrices(location):
    table_service = TableService(account_name=app.config["DB_ACCOUNT_NAME"], account_key=os.environ.get("ACCOUNT_KEY"))
    all_prices = table_service.query_entities(app.config["DB_TABLE_NAME"], filter=("PartitionKey eq '" + location + "'"))
    price_list = []
    for price in all_prices:
        price_list.append(price)
    return jsonify(price_list)

# Insert new fuel price to the database. Only for dev purpose!
@app.route("/statistics/input", methods=["POST"])
def input():
    # TODO: Implement
    #json_content = request.get_json()
    #json_content = json.dumps(json_content)
    #if json_content == None:
    #   return "No json content detected"
    # Create a new table entity to insert into the db. 
    table_service = TableService(account_name=app.config["DB_ACCOUNT_NAME"], account_key=os.environ.get("ACCOUNT_KEY"))
    entry = Entity()
    entry.PartitionKey = 'trondelag' # Insert all to trondelag partition for now. Change later
    entry.RowKey = uuid.uuid4() # Generate new random UUID
    entry.price = 16.58 #TODO: Get from JSON input
    entry.location = "63.410531, 10.418190" #TODO: Get from JSON input
    entry.fueltype = "diesel" #TODO: Get from JSON input. Check taht fueltype is either diesel or gasoline?
    table_service.insert_entity(app.config["DB_TABLE_NAME"], entry)
    return "Inserted successfully: " + str(entry)

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
    return "Oops, endpoint not found (404 error) :("

@app.errorhandler(500)
def bad_request500(error):
    return "Oops, internal server error (500 error) :("

if __name__ == "__main__":
    app.run(host='127.0.0.1')