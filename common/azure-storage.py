"""
Handles communication with Azure Storage
"""

import os

from azure.storage.blob import BlobServiceClient

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

container_client = BlobServiceClient.from_connection_string(connect_str).get_container_client('milo')
blob_list = container_client.list_blobs(prefix=folder)
for blob in blob_list:
    print(blob)

blob_service_client = BlobServiceClient.from_connection_string(connect_str)

blob_client = blob_service_client.get_blob_client(container='milo', blob=folder + '/train.csv')
with open(folder + '/train.csv', "rb") as data:
    blob_client.upload_blob(data)

blob_client = blob_service_client.get_blob_client(container='milo', blob=folder + '/test.csv')
with open(folder + '/test.csv', "rb") as data:
    blob_client.upload_blob(data)

blob_client = blob_service_client.get_blob_client(container='milo', blob=folder + '/label.txt')
with open(folder + '/label.txt', "rb") as data:
    blob_client.upload_blob(data)
