from src.logger import logging
from src.exception import CustomException
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.utils import resample




# Abstract Base Class for Imbalanced Data Handling Strategy
class ImbalanceHandlingStrategy(ABC):
    @abstractmethod
    def handle_imbalance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Abstract method to handle imbalanced data in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features and target.
        target_column (str): The column representing the target variable.

        Returns:
        pd.DataFrame: A dataframe with a balanced target distribution.
        """
        pass


# Concrete Strategy for Oversampling
class OversamplingStrategy(ImbalanceHandlingStrategy):
    def handle_imbalance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        logging.info("Handling imbalance using oversampling.")
        max_class = df[target_column].value_counts().idxmax()
        min_class = df[target_column].value_counts().idxmin()

        majority_df = df[df[target_column] == max_class]
        minority_df = df[df[target_column] == min_class]

        logging.info(f"Majority class: {max_class}, Minority class: {min_class}")
        minority_upsampled = resample(minority_df, 
                                      replace=True, 
                                      n_samples=len(majority_df), 
                                      random_state=42)
        balanced_df = pd.concat([majority_df, minority_upsampled])
        logging.info("Oversampling completed. Data is now balanced.")
        return balanced_df


# Concrete Strategy for Undersampling
class UndersamplingStrategy(ImbalanceHandlingStrategy):
    def handle_imbalance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        logging.info("Handling imbalance using undersampling.")
        max_class = df[target_column].value_counts().idxmax()
        min_class = df[target_column].value_counts().idxmin()

        majority_df = df[df[target_column] == max_class]
        minority_df = df[df[target_column] == min_class]

        logging.info(f"Majority class: {max_class}, Minority class: {min_class}")
        majority_downsampled = resample(majority_df, 
                                        replace=False, 
                                        n_samples=len(minority_df), 
                                        random_state=42)
        balanced_df = pd.concat([majority_downsampled, minority_df])
        logging.info("Undersampling completed. Data is now balanced.")
        return balanced_df


# Context Class for Imbalanced Data Handling
class ImbalanceDetector:
    def __init__(self, strategy: ImbalanceHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ImbalanceHandlingStrategy):
        logging.info("Switching imbalance handling strategy.")
        self._strategy = strategy

    def handle_imbalance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        logging.info("Executing imbalance handling strategy.")
        return self._strategy.handle_imbalance(df, target_column)

    def visualize_distribution(self, df: pd.DataFrame, target_column: str):
        logging.info(f"Visualizing class distribution for target: {target_column}")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=target_column, data=df)
        plt.title(f"Class Distribution of {target_column}")
        plt.show()
        logging.info("Class distribution visualization completed.")


# Example usage
if __name__ == "__main__":
    
    '''
    # Example dataframe
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
        "feature2": [10, 20, 30, 40, 50, 60, 70, 80],
        "target": [0, 0, 0, 0, 0, 1, 1, 1]  # Imbalanced target
    })

    # Initialize the ImbalanceDetector with the Oversampling Strategy
    imbalance_detector = ImbalanceDetector(OversamplingStrategy())

    # Visualize class distribution before handling
    imbalance_detector.visualize_distribution(df, "target")

    # Handle imbalance
    balanced_df = imbalance_detector.handle_imbalance(df, "target")

    # Visualize class distribution after handling
    imbalance_detector.visualize_distribution(balanced_df, "target")

    print(balanced_df)
    
    '''
    '''
    # Example dataframe
    df = pd.read_csv('/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/kaggle_data/creditcard.csv')
    # Initialize the ImbalanceDetector with the Undersampling Strategy
    imbalance_detector = ImbalanceDetector(UndersamplingStrategy())

    # Visualize class distribution before handling
    imbalance_detector.visualize_distribution(df, "Class")

    # Handle imbalance
    balanced_df = imbalance_detector.handle_imbalance(df, "Class")

    # Visualize class distribution after handling
    imbalance_detector.visualize_distribution(balanced_df, "Class")

    print(balanced_df)
    
    '''
