# TDT4856 - Fuel Price Detection - Group #3 
[![Build status](https://dev.azure.com/matstyldum/FuelPriceApp/_apis/build/status/fuelpriceapi%20-%20CI)](https://dev.azure.com/matstyldum/FuelPriceApp/_build/latest?definitionId=2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The main goal of this project is to use computer vision systems to detect fuel prices from local fuel stations. In order to succeed with this project the group will develop an application which will be enable the user to take pictures of price signs and list fuel prices. 
## Mobile application
TODO: Write necessary setup and prerequisites for the mobile app here.

## Backend
Process pictures and stores fuel price related information.
### Prerequisites
- [Python 3.8.1](https://www.python.org/downloads/)
- [Flask 1.1.1](https://flask.palletsprojects.com/en/1.1.x)
- [Azure Cosmos DB Table SDK for Python](https://docs.microsoft.com/en-us/azure/cosmos-db/table-storage-how-to-use-python)
- [UUID](https://docs.python.org/3/library/uuid.html)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

### Getting started
This project requires you to have Python3 and pip installed on your machine. Easiest way to get it is by installing [Anaconda](https://www.anaconda.com/download). After you have done this, run the command below in cmd/termnial to install the remaining prerequisites.

### Running the application locally
When running the application locally you need a .env-file. Ask someone in the team for this.

1. Navigate to the src folder
2. Install all necessary requirements
```shell
pip install -r requirements.txt
```
3. Navigate to the app folder
4. Run the application
```shell
python main.py
```
5. Access the application by navigating to "http://localhost:80" or "http://0.0.0.0:80".

### CI/CD
Continous integration is set up so that all commits to the backend-master branch automatically will build a new container.
This container wil then be deployed to a custom Azure Container Registry. The application will run as a web service and be available at 
[https://fuelpriceapi.azurewebsites.net/](https://fuelpriceapi.azurewebsites.net/) if everything succeeds!

### Dockerkize application
If you want to build and run the Docker image locally use the below commands:
```shell
docker build -t <image-name>:<tag-name> .
docker run -p 80:80 <image-name>:<tag-name>
```

### API reference
How to communicate with the server via the REST API. Remark: This is work in progress an breaking changes may occur!
#### GET
- By location and ID: ```/prices/<string:location>/<string:id>```
- By location and coordinates: ```/prices/<string:location>/coordinates/<string:coordinates>```
- By location: ```/prices/<string:location>```

#### POST
To insert an data directly as text to the database send a POST request to "/upload/prices" with JSON content on the following form:
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
To upload and process an image send a POST request to "/upload/image" with JSON content on the following form:
```JSON
{
    "image": "Base64 string here",
    "location": "63.420876,10.460610"
}
```

### Database design
Screenshot of database structure with example data:
![Database design](https://i.imgur.com/yxtjrll.png)

### Questions
All questions related to the backend or CI/CD-config, can be directed to [Mats Tyldum](https://github.com/maattss).
