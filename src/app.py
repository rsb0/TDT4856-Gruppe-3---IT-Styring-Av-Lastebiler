from flask import Flask
import os
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

app = Flask(__name__)
app.debug = True # only for development!

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/test")
def test():
    table_service = TableService(account_name='fuelpricestorage', account_key=os.environ.get("ACCOUNT_KEY"))
    return table_service.get_entity('test', 'testPartition', 'testRow')

if __name__ == "__main__":
    app.run(host='127.0.0.1')