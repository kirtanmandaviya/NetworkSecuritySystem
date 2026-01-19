import sys 
import os 
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TARNSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifacts,
    DataValidationArtifacts
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifacts: DataValidationArtifacts,
                 data_transformation_config: DataTransformationConfig
                 ):
        try:
            self.data_validation_artifact:DataValidationArtifacts = data_validation_artifacts
            self.data_tarnsformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initialies a KNNImputer object with the parameter depcified in the training_pipeline.py file and returns a Pipeline object with the KNNImputer object as the first step.
        
        Args:
            cls: DataTransformation
        
        Returns:
            A Pipeline Object
        """
        
        logging.info("Entered get_data_transformer_object method of Transformation class")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TARNSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialise KNNImputer with {DATA_TARNSFORMATION_IMPUTER_PARAMS}"
            )
            processor:Pipeline = Pipeline(
                [
                    ("imputer", imputer)
                ]
            )
            return processor
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            # Training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature.replace(-1, 0)
            
            # Test dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature1 = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature1.replace(-1, 0)         

            # Transform Data
            preprocessor = self.get_data_transformer_object()
            
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            
            # save numpy array data
            save_numpy_array_data(
                self.data_tarnsformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_tarnsformation_config.transformed_test_file_path,
                array=test_arr
            )
            save_object(
                self.data_tarnsformation_config.transformed_object_file_path,
                preprocessor_obj
            )
            
            save_object("final_model/preprocessor.pkl", preprocessor_obj)
            
            # Preparing artifacts
            
            data_transformation_artifact = DataTransformationArtifacts(
                transformed_object_file_path=self.data_tarnsformation_config.transformed_object_file_path,
                transformed_train_file_path= self.data_tarnsformation_config.transformed_train_file_path,
                transformed_test_file_path= self.data_tarnsformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact
                      
        except Exception as e:
            raise NetworkSecurityException(e,sys)