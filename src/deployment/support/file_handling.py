from pathlib import Path
import re
from typing import Callable, Optional
import polars as pl 
import os

class FileHandler:
    def read_parquet_file(self, file_path: str) -> pl.DataFrame:  
        """
        Reads a parquet file and parses its first column as a datetime.
        """
        # base_dir = os.path.dirname(__file__)
        # file_path = os.path.join(base_dir, file_path)
        df = pl.read_parquet(file_path)
        
        # Handle index column (first column) from pandas parquet
        first_col = df.columns[0]

        return df.rename({first_col: "datetime"})
    
    def list_all_files(self, directory: str):
        """
        Lists all nested files in a directory.
        
        Args:
            directory (str): Path to a directory.
        
        Returns:
            List: List with all files, nested or not, inside the directory.
        """
        # base_dir = os.path.dirname(__file__)
        # directory = os.path.join(base_dir, directory)
        return [file for file in Path(directory).rglob('*') if file.is_file()]

    def save_dataframe_parquet_file(self, df: pl.DataFrame, save_path: Path) -> None:  
        """
        Saves a DataFrame to a parquet file, creating necessary directories.
        
        Args:
            df (pl.DataFrame): The DataFrame to save.
            save_path (str): Path to save the parquet file.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(save_path)

    def read_transform_save(
        self,
        transform_function: Callable[[pl.DataFrame], pl.DataFrame],  
        read_path: str, 
        save_path: Optional[str] = None
    ) -> pl.DataFrame:  # Polars return type
        """
        Reads a parquet, applies a transformation function, and saves the result.

        Args:
            transform_function (Callable): Function to transform the DataFrame.
            read_path (str): Path to the input parquet file.
            save_path (Optional[str]): Path to save the transformed parquet file. If None, a default path is generated.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        # Read
        base_dir = os.path.dirname(__file__)
        read_path = os.path.join(base_dir, read_path)
        df = self.read_parquet_file(read_path)


        # Transform
        df = transform_function(df)

        # Save
        if not save_path:
            save_path = Path(re.sub(r"extracted","transformed",read_path))
        
        self.save_dataframe_parquet_file(df, save_path)
        return df