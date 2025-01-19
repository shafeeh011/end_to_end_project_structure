import os
import sys
import zipfile
from abc import ABC, abstractmethod

import pandas as pd
import kaggle  # Import the Kaggle API

from src.logger import logging
from src.exception import CustomException



# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        try:
            """Extracts a .zip file and returns the content as a pandas DataFrame."""
            if not file_path.endswith(".zip"):
                raise ValueError("The provided file is not a .zip file.")

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall("extracted_data")

            extracted_files = os.listdir("extracted_data")
            csv_files = [f for f in extracted_files if f.endswith(".csv")]

            if len(csv_files) == 0:
                raise FileNotFoundError("No CSV file found in the extracted data.")
            if len(csv_files) > 1:
                raise ValueError("Multiple CSV files found. Please specify which one to use.")

            csv_file_path = os.path.join("extracted_data", csv_files[0])
            df = pd.read_csv(csv_file_path)

            return df

        except Exception as e:
            raise CustomException(e, sys)


# Implement a concrete class for Kaggle Dataset Ingestion
class KaggleDataIngestor(DataIngestor):
    def ingest(self, dataset_identifier: str) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset and returns its content as a pandas DataFrame.
        :param dataset_identifier: Kaggle dataset identifier (e.g., 'username/dataset-name')
        :return: DataFrame containing the dataset.
        """
        # Initialize the Kaggle API
        try:
            kaggle.api.authenticate()

            # Download the dataset
            dataset_dir = ("kaggle_data")
            kaggle.api.dataset_download_files(dataset_identifier, path=dataset_dir, unzip=True)

            # Find the CSV files
            extracted_files = os.listdir(dataset_dir)
            csv_files = [f for f in extracted_files if f.endswith(".csv")]

            if len(csv_files) == 0:
                raise FileNotFoundError("No CSV file found in the Kaggle dataset.")
            if len(csv_files) > 1:
                raise ValueError("Multiple CSV files found. Please specify which one to use.")


            csv_file_path = os.path.join(dataset_dir, csv_files[0])
            df = pd.read_csv(csv_file_path)
            
            logging.info("Kaggle dataset ingested successfully.")
        

            return df
        
        except Exception as e:
            raise exception.DataIngestionException(e, sys)


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(source_type: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on source type.
        :param source_type: The source type (e.g., 'zip', 'kaggle')
        :return: An instance of the appropriate DataIngestor.
        """
        if source_type == "zip":
            return ZipDataIngestor()
        elif source_type == "kaggle":
            return KaggleDataIngestor()
        else:
            raise ValueError(f"No ingestor available for source type: {source_type}")


# Example usage
if __name__ == "__main__":
    # Example for a ZIP file
    # file_path = "path/to/archive.zip"
    # data_ingestor = DataIngestorFactory.get_data_ingestor("zip")
    # df = data_ingestor.ingest(file_path)
    # print(df.head())

    #Example for a Kaggle dataset
    #dataset_identifier = "mlg-ulb/creditcardfraud"  # Replace with actual Kaggle dataset identifier
    #data_ingestor = DataIngestorFactory.get_data_ingestor("kaggle")
    #df = data_ingestor.ingest(dataset_identifier)
    #print(df.head())

    #pass