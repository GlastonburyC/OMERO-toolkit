import pandas as pd
import os
import omero
from omero.gateway import BlitzGateway
from math import ceil
import yaml
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Enumerate WSIs in OMERO dataset(s) and chunk their filenames into CSVs")
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML configuration file')

args = parser.parse_args()

# Load configuration from YAML file
try:
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logging.error(f"Configuration file {args.config} not found.")
    exit(1)
except yaml.YAMLError as e:
    logging.error(f"Error parsing YAML configuration file: {e}")
    exit(1)

# Extract OMERO configuration
omero_config = config.get('omero', {})
username = omero_config.get('username')
password = omero_config.get('password')
host = omero_config.get('host')
port = omero_config.get('port')
dataset_ids = omero_config.get('dataset_id')
chunk_size = omero_config.get('chunk_size')

# Validate configuration
if not all([username, password, host, port, dataset_ids, chunk_size]):
    logging.error("Missing required OMERO configuration. Please check your YAML file.")
    exit(1)

# Parse dataset IDs (handle single value or list)
if isinstance(dataset_ids, int):
    dataset_ids = [dataset_ids]
elif isinstance(dataset_ids, str):
    try:
        dataset_ids = [int(id_.strip()) for id_ in dataset_ids.split(",")]
    except ValueError:
        logging.error("Invalid dataset_id string. Ensure integers separated by commas")
        exit(1)
elif isinstance(dataset_ids, list): 
    try:
        dataset_ids = [int(id_) for id_ in dataset_ids]
    except ValueError:
        logging.error("Invalid dataset_id list. Ensure all entries are integers.")
        exit(1)
else:
    logging.error("Invalid format for `dataset_id`. Provide a single ID or a list of IDs.")
    exit(1)

try:
    chunk_size = int(chunk_size)
except ValueError:
    logging.error("`chunk_size` must be an integer.")
    exit(1)

# Function to chunk list into smaller lists
def chunk_list(data_list, chunk_size):
    return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

# Connect to the OMERO server
logging.info("Connecting to OMERO server...")
conn = BlitzGateway(username, password, host=host, port=port)
connected = conn.connect()

if connected:
    logging.info("Connected to OMERO server successfully.")
    
    for dataset_id in dataset_ids:
        try:
            # Fetch dataset with the given ID
            logging.info(f"Fetching dataset with ID {dataset_id}...")
            dataset = conn.getObject("Dataset", dataset_id)
            
            if not dataset:
                logging.warning(f"No dataset found with ID {dataset_id}. Skipping...")
                continue

            # Get all WSI names and filter out the macro images
            wsi_names = [image.getName() for image in dataset.listChildren() if "[0]" in image.getName()]
            
            if not wsi_names:
                logging.warning(f"No WSIs found in Dataset ID={dataset_id}. Skipping...")
                continue
            
            # Chunk the WSI names into smaller lists
            chunks = chunk_list(wsi_names, chunk_size)
            
            # Create a directory to store the CSV files
            output_dir = f"omero_{dataset_id}_wsi_chunks"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each chunk to a separate CSV file
            for j, chunk in enumerate(chunks):
                df = pd.DataFrame(chunk, columns=['WSI Names'])
                csv_filename = os.path.join(output_dir, f"wsi_chunk_{j + 1}.csv")
                df.to_csv(csv_filename, index=False)
                logging.info(f"Saved {len(chunk)} WSI names to {csv_filename}")

        except omero.ApiUsageException as e:
            logging.error(f"API error while fetching dataset with ID {dataset_id}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error for dataset ID {dataset_id}: {e}")

    conn.close()
    logging.info("Disconnected from OMERO server.")
else:
    logging.error("Failed to connect to OMERO server.")
    exit(1)

