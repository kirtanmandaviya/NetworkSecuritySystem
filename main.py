import sys
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.data_transformation import DataTransformation,DataTransformationConfig

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initialte the data ingestion")
        
        dataingestionartifacts = data_ingestion.initiate_data_ingestion()
        logging.info("Data initiate completed")
        print(dataingestionartifacts)
        
        data_validation_confing = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        data_validation = DataValidation(data_ingestion_artifacts=dataingestionartifacts,data_validatio_config=data_validation_confing)
        logging.info('Initiate data validation')
        data_validation_artifacts = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifacts)
        
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=trainingpipelineconfig
        )
        data_transformation = DataTransformation(
            data_validation_artifacts=data_validation_artifacts,
            data_transformation_config=data_transformation_config
        )
        logging.info("Initiate data transformation")
        data_transformation_artifacts = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifacts)
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)