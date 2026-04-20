"""
Module for handling file paths related to data preparation and model training outputs.
Author: Dr. Aritra Bal (ETP)
Date: 27 November 2025
"""
import os, pathlib, datetime,loguru

def fetch_subfolders(base_dir: str='/ceph/abal/piston/work_dir_2/data_preparation/') -> dict:
    '''Fetches subfolder paths for raw PDBs and grid maps.
        Args:
            base_dir (str): Base directory to fetch subfolders from.
        Returns:
            dict: Dictionary containing paths to 'raw' and 'grid_maps' subfolders.
    '''
    loguru.logger.info(f"Base directory: {base_dir}")
    sub_folders = {
        'raw': os.path.join(base_dir, '00-raw_pdbs'),
        'grid_maps': os.path.join(base_dir, '07-grid'),
        'metrics': os.path.join(base_dir, 'irmsd.csv')
    }
    return sub_folders

def get_output_paths(output_dir: str='/ceph/abal/piston/QML/model_training/outputs/',seed: str = None) -> str:
    '''Generates an output path based on the provided seed or current datetime.
        Args:
            output_dir (str): Base directory for outputs.
            seed (str, optional): Seed to create a unique output path. Defaults to None.
        Returns:
            str: Full path to the output directory.
    '''
    if seed is None:
        loguru.logger.info("No seed provided, generating one based on current datetime.")
        seed = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    loguru.logger.info(f"Using seed: {seed} for creating output path.")
    loguru.logger.info(f"Outputs will therefore be saved to: {output_dir}")
    output_path = os.path.join(output_dir, f'output_{seed}')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path

