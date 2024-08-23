import omero
from omero.gateway import BlitzGateway
import yaml
import argparse
import os
from omero.plugins.upload import UploadControl
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description="Upload a directory of WSI files to a new dataset on OMERO server")
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML configuration file')
parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing WSI files')
parser.add_argument('--threads', type=int, default=4, help='Number of parallel threads for uploading')

args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
omero_config = config['omero']

# Extract configuration details from the YAML file
username = omero_config['username']
password = omero_config['password']
host = omero_config['host']
port = omero_config['port']

# Get the dataset name from the configuration
dataset_name = omero_config.get('new_dataset_name', None)

if dataset_name is None:
    print("Dataset name must be provided in the config file.")
    exit(1)

# Function to upload a single file
def upload_file(file_path, dataset_id, conn):
    try:
        # Use the UploadControl plugin to upload files
        upload_ctrl = UploadControl(conn.c, args=[], mode='simple')
        upload_ctrl.upload_paths([file_path], dataset_id)
        print(f"{os.path.basename(file_path)} uploaded successfully.")
        return (file_path, "Success")
    except Exception as e:
        print(f"Failed to upload {os.path.basename(file_path)}: {e}")
        return (file_path, "Failed")

# Connect to the OMERO server
conn = BlitzGateway(username, password, host=host, port=port)
connected = conn.connect()

if connected:
    print("Connected to OMERO server successfully!")
    
    # Get the directory containing WSI files
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        conn.close()
        exit(1)

    # Create a new dataset
    project = conn.getObject("Project", omero_config['project_id'])
    
    if not project:
        print(f"Project ID {omero_config['project_id']} not found.")
        conn.close()
        exit(1)
    
    new_dataset = omero.model.DatasetI()
    new_dataset.setName(omero.rtypes.rstring(dataset_name))
    project.linkDataset(new_dataset)
    conn.getUpdateService().saveObject(new_dataset)
    
    dataset_id = new_dataset.id.val
    print(f"Dataset '{dataset_name}' created with ID {dataset_id}")

    # Upload WSI files to the new dataset
    wsi_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    if not wsi_files:
        print(f"No WSI files found in directory {directory}.")
        conn.close()
        exit(1)

    # Use ThreadPoolExecutor to parallelize uploads
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(upload_file, wsi_file, dataset_id, conn) for wsi_file in wsi_files]
        for future in as_completed(futures):
            result = future.result()
            if result[1] == "Failed":
                print(f"Upload failed for: {result[0]}")

    conn.close()
else:
    print("Failed to connect to OMERO server.")
