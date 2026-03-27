from pathlib import Path
import zipfile


"""
Step 0:
- Check whether ./dataset/archive.zip exists
- If not, cancel EDA
- If yes, extract it into ./dataset
- Extraction will happen regardless of whether folders exist or not

Returns:
    dict | None
"""
def zip_extraction(project_root: Path):
    dataset_dir = project_root / "dataset"
    zip_path = dataset_dir / "archive.zip"

    # 1. Check zip file
    if not zip_path.exists():
        print(f"[ERROR] archive.zip not found: {zip_path}")
        print("[EDA] canceled.")
        return None

    # 2. Extract zip
    try:
        print(f"[INFO] Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"[INFO] Extraction completed into: {dataset_dir}")

    except zipfile.BadZipFile:
        print(f"[ERROR] '{zip_path.name}' is not a valid ZIP file or is corrupted.")
        print("[EDA CANCELLED] Extraction failed.")
        return None
    
    except Exception as e:
        print(f"[ERROR] Failed to extract zip file: {e}")
        print("[EDA CANCELLED] Extraction failed.")
        return None

    return {
        "dataset_dir": dataset_dir,
        "zip_path": zip_path,
    }