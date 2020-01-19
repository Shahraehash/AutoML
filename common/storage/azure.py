"""
Handles communication with Azure Storage
"""

import os

from azure.storage.blob import BlobServiceClient

class DirectoryClient:
    def __init__(self, connection_string, container_name):
        service_client = BlobServiceClient.from_connection_string(connection_string)
        self.client = service_client.get_container_client(container_name)



    def upload_file(self, source, dest):
        """
        Upload a single file to a path inside the container
        """

        with open(source, 'rb') as data:
            self.client.upload_blob(name=dest, data=data)

blob_list = container_client.list_blobs(prefix=folder)
for blob in blob_list:
    print(blob)
