# TDT4856 - Fuel Price Detection - Group #3
The main goal of this project is to use computer vision systems to detect fuel prices from local fuel stations.
In order to succeed with this project the group will develop an application which will be enable the user to take pictures of price signs and list fuel prices. 
## Frontend
TODO: Write necessary setup and prerequisites for the frontend here.

## Backend
Process pictures and stores fuel price related information.
### Prerequisites
- [Python 3.8.1](https://www.python.org/downloads/)
- [Flask 1.1.1](https://flask.palletsprojects.com/en/1.1.x)
- [Azure Cosmos DB Table SDK for Python](https://docs.microsoft.com/en-us/azure/cosmos-db/table-storage-how-to-use-python)
- [UUID](https://docs.python.org/3/library/uuid.html)

### Getting started
This project requires you need to have Python3 and pip installed on your machine. Easiest way to get it is by installing [Anaconda](https://www.anaconda.com/download). After you have done this, run the command below in cmd/termnial to install the remaining prerequisites.

```shell
python -m pip install Flask
python -m pip install azure-cosmosdb-table
python -m pip install uuid
```
### Running the application locally
In order to get access to the database you need to an acces key (ask someone in the team for this).
Then store it as an [environment variable](https://flask.palletsprojects.com/en/1.1.x/config/#configuring-from-environment-variables) named "ACCOUNT_KEY"

Run the app by issuing this command when in the src folder in cmd/terminal
```shell
python app.py
```
Access the application by navigating to "http://localhost:5000"

### Dockerkized application
Not working yet :(
```shell
docker build -t <tag-name> .
docker run -p 5000:5000 <image-name>
```

### API reference
How to communicate with the server via the REST API. Remark: This is work in progress an breaking changes may occur!
#### GET
- By location and ID: ```/price/<string:location>/<string:id>```
- By location and coordinates: ```/price/<string:location>/coordinates/<string:coordinates>```
- By location: ```/price/<string:location>```

#### POST
To insert an entry to the database send a POST request to "/input" with JSON content on the following form:
```JSON
{ 
   "new_prices":[{ 
   	  "county":"trondelag",
      "price":15.85,
      "location":"63.420876,10.460610",
      "fueltype":"diesel"
   },{ 
   	  "county":"trondelag",
      "price":16.34,
      "location":"63.420876,10.460610",
      "fueltype":"gasoline"
   },{ 
   	  "county":"trondelag",
      "price":17.31,
      "location":"63.420876,10.460610",
      "fueltype":"gasoline"
   },{ 
   	  "county":"trondelag",
      "price":15.84,
      "location":"63.420876,10.460610",
      "fueltype":"diesel"
   }]
}
```

### Database design
Screenshot of database structure with example data:
![Database design](https://i.imgur.com/yxtjrll.png)
