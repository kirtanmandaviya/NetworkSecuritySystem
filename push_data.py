import os 
import sys 
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGODB_URI")

ca = certifi.where()

class NetworkDataExtract(): 
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convert(self, file_path):
        try:
            
            logging.info("Initiate converting CSV to JSON.")
            
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        
            logging.info("Converting CSV to JSON complete.")
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_to_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
if __name__ == '__main__':
    FILE_PATH = 'Network_Data\phisingData.csv'
    DATABASE = "NETWORKSECURITYSYSTEM"
    Collection = "NetworkData"
    
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convert(file_path=FILE_PATH)
    no_of_records = networkobj.insert_data_to_mongodb(records, DATABASE, Collection)
    print(no_of_records)
    logging.info("Data inserted into database") 