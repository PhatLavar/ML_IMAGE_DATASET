from pathlib import Path
from eda_helper import *

def dataset_extraction():
    project_root = Path(__file__).resolve().parent.parent
    dataset_info = zip_extraction(project_root)
    if dataset_info is None:
        return
    
    print("[STEP 0] Zip file extracted successfully. Dataset is ready for EDA.")
    

def main():
    dataset_extraction()


if __name__ == "__main__":
    main()