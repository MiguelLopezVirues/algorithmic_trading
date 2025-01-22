import pandas as pd
from pathlib import Path
import re
from typing import Callable, Optional

class EatD:
    pass


# TO-DO:
# [] add error-handling

class FileHandler:
    def read_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and parses its index as a datetime.
        
        Args:
            file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: The loaded DataFrame with a datetime index.
        """
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, format="%Y-%m-%d")
        return df   
    
    def list_all_files(self, directory: str):
        """
        Lists all nested files in a directory.
        
        Args:
            directory (str): Path to a directory.
        
        Returns:
            List: List with all files, nested or not, inside the directory.
        """
        return [file for file in Path(directory).rglob('*') if file.is_file()]

    def save_dataframe_csv_file(self, df: pd.DataFrame, save_path: str) -> None:
        """
        Saves a DataFrame to a CSV file, creating necessary directories.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            save_path (str): Path to save the CSV file.
        """
        save_path_dir = Path(save_path).parent
        save_path_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(save_path)

    def read_transform_save(
        self,
        transform_function: Callable[[pd.DataFrame], pd.DataFrame], 
        read_path: str, 
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Reads a CSV, applies a transformation function, and saves the result.

        Args:
            transform_function (Callable): Function to transform the DataFrame.
            read_path (str): Path to the input CSV file.
            save_path (Optional[str]): Path to save the transformed CSV file. If None, a default path is generated.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """

        # Read
        df = self.read_csv_file(read_path)

        # Transform
        df = transform_function(df)

        # Save
        if not save_path:
            save_path = re.sub(r"extracted","transformed",read_path)
        
        self.save_dataframe_csv_file(df, save_path)

        return df