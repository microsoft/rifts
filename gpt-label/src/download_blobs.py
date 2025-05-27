from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
import os
import sys

blob_starts_name = None

# make blob_starts_name a command line argument
if len(sys.argv) > 1:
    blob_starts_name = sys.argv[1]
else:
    print("Please provide a blob name to start with.")
    print("Usage: python download_blobs.py <blob_starts_name>")
    sys.exit(1)


# Define your account URL and container name
account_url = ""
container_name = ""
download_folder = "./logs_parquet"  # Local path to download files

# Use DefaultAzureCredential for authentication
credential = DefaultAzureCredential()

# Create the ContainerClient object
container_client = ContainerClient(
    account_url=account_url, container_name=container_name, credential=credential
)

# List the blobs in the container
blob_list = container_client.list_blobs(name_starts_with=blob_starts_name)

for blob in blob_list:
    print(blob.name)

    blob_client = container_client.get_blob_client(blob)

    # Create a local file path to save the blob
    download_file_path = os.path.join(download_folder, blob.name.split("/")[-1])

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

    # Download the blob to a local file
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    print(f"Downloaded {blob.name} to {download_file_path}")

print("Download complete")
