# Base image
FROM python:3.7-slim

# Copy over our requirements.txt file
COPY requirements.txt /tmp/

# Upgrade pip and install required python packages
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

# Copy over our app code
COPY ./app /app

# Expose port 80
EXPOSE 80

# Table environment variables
ENV DB_ACCOUNT_NAME "fuelpricestorage"
ENV DB_TABLE_NAME "prices"

# Blob environment variables
ENV BLOB_CONTAINER_NAME "images"

CMD [ "python", "app/main.py" ]