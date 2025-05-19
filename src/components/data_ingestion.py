import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig




@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entering the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\StudentsPerformance.csv') # Reading the dataset
            logging.info('Read the dataset as dataframe')

            # Creating the directory for artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # Saving the raw data and Converting it to csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train test split initiated')

            # Splitting the data into train and test sets
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            # Saving the train and test sets to csv files
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            # Returning the paths of the train and test sets
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.info('Exception occurred in data ingestion')
            raise CustomException(e,sys) from e

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
    