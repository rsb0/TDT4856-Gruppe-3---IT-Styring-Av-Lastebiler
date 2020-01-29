# TDT4856 Fuel Price Detection Group #3
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

### Getting started
This project requires, you need to have Python3 and pip installed on your machine. Easiest way to get it is by installing [Anaconda](https://www.anaconda.com/download). After you have done this, run the command below in cmd/termnial to install the remaining prerequisites.

```shell
python -m pip install Flask
python -m pip install azure-cosmosdb-table
```
### Running the application
In order to get access to the database you need to an acces key (ask someone in the team for this).
Then store it as an [environment variable](https://flask.palletsprojects.com/en/1.1.x/config/#configuring-from-environment-variables) named "ACCOUNT_KEY"

Run the app by issuing this command when in the src folder in cmd/terminal
```shell
python app.py
```
Access the application by navigating to "http://localhost:5000"

### API reference
TODO: Write

### Database design
NoSQL