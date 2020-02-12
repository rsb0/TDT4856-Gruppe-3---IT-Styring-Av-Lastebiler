"""
    Fuel price API
"""
from flask import Flask, request, redirect, jsonify, json
import uuid, os
from get_handlers import get_prices_by_area, get_prices_by_key
from input_handlers import upload_picture_to_blob, upload_prices_to_table

app = Flask(__name__)
app.debug = True # only for development!

# Default route returns default info string
@app.route("/")
def index():
    return "Official API for the amazing fuel price application ®"

# Get all fuel prices based on location
@app.route("/prices/<string:area>")
def get_prices(area):
    return get_prices_by_area(area)

# Get a fuel price based on location and id
@app.route("/prices/<string:area>/<string:id>")
def get_prices_by_id(area, id):
    return get_prices_by_key(area, id)

# Get fuel prices based on location and coordinates
@app.route("/prices/<string:area>/coordinates/<string:coordinates>")
def get_prices_by_coordinates(area, coordinates):
    return get_prices_by_key(area, coordinates)

# Insert new fuel prices to the database
@app.route("/upload/price", methods=["POST"])
def input_price():
    json_prices = request.get_json().get("new_prices")
    if json_prices is None:
       return "No JSON content detected!"
    return upload_prices_to_table(json_prices)

# Process picture to extract price
@app.route("/upload/image", methods=["POST"])
def input_picture():   
    if request.files['img']:
        # TODO: Add picture processing here
        upload_picture_to_blob(request.files['img'])
        return "Image succesfully uploaded to blob storage!"
    else:
    	return "Where is the image?"

# Error handling
@app.errorhandler(404)
def page_not_found(error):
    return "Oops, endpoint not found - 404"

@app.errorhandler(500)
def bad_request500(error):
    return "Oops, internal server error - 500"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)