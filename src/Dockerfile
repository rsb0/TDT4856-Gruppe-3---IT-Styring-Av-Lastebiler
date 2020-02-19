# Base image
FROM tiangolo/uwsgi-nginx-flask:python3.6

# Copy over our requirements.txt file
COPY requirements.txt /tmp/

# Upgrade pip and install required python packages
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

# Copy over our app code
COPY ./app /app

# Expose port 80
EXPOSE 80

# Db environment variables
ENV DB_ACCOUNT_NAME "fuelpricestorage"
ENV DB_TABLE_NAME "prices"

# Blob environment variables
ENV BLOB_CONTAINER_NAME "images"

# KV environment variables
ENV KEY_VAULT_NAME "fuelprice-server-kv"
ENV SECRET_NAME_STORAGE_KEY "storage-account-key"
ENV SECRET_NAME_BLOB_STRING "blob-connection-string"