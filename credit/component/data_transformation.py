from cgi import test
from sklearn import preprocessing
from credit.exception import CreditException
from credit.logger import logging
from credit.entity.config_entity import DataTransformationConfig 
from credit.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from credit.constant import *
from credit.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data

class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,marriage_idx=3,
                 education_idx=2,
                 pay_0_idx = 5,
                 columns=None):
        try:
            self.columns = columns
            if self.columns is not None:
                marriage_idx = self.columns.index(COLUMN_MARRIAGE_KEY)
                education_idx = self.columns.index(COLUMN_EDUCATION_KEY)
                pay_0_idx = self.columns.index(COLUMN_PAY_0)              
            
            self.marriage_idx = marriage_idx
            self.education_idx = education_idx
            self.pay_0_idx = pay_0_idx
        
        except Exception as e:
            raise CreditException(e,sys) from e
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            X.loc[X[COLUMN_MARRIAGE_KEY]==0,COLUMN_MARRIAGE_KEY] = 3
            fil = (X[COLUMN_EDUCATION_KEY] == 5) | (X[COLUMN_EDUCATION_KEY] == 6) | (X[COLUMN_EDUCATION_KEY] == 0)
            X.loc[fil,COLUMN_EDUCATION_KEY] = 4
            X = X.rename(columns={X.columns[self.pay_0_idx] : COLUMN_PAY_1 })
            col_lst = [COLUMN_PAY_1,COLUMN_PAY_2,COLUMN_PAY_3,COLUMN_PAY_4,COLUMN_PAY_5,COLUMN_PAY_6]
            for col in col_lst:
                X[col] = X[col].replace(-1,0)
                X[col] = X[col].replace(-2,0)
            return X
        except Exception as e:
            raise CreditException(e, sys) from e


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise CreditException(e,sys) from e

    

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]

            num_pipeline = Pipeline(steps=[
                ('feature_transformer', FeatureTransformer(
                    columns=numerical_columns
                )),
                ('scaler', StandardScaler())
            ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns)
            ])
            return preprocessing

        except Exception as e:
            raise CreditException(e,sys) from e   


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")