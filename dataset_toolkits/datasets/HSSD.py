import os
import re
import argparse
import tarfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import huggingface_hub
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv")
    return metadata
        

def download(metadata, output_dir, **kwargs):    
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # check login
    try:
        huggingface_hub.whoami()
    except:
        print("\033[93m")
        print("Haven't logged in to the Hugging Face Hub.")
        print("Visit https://huggingface.co/settings/tokens to get a token.")
        print("\033[0m")
        huggingface_hub.login()
        
    try:
        huggingface_hub.hf_hub_download(repo_id="hssd/hssd-models", filename="README.md", repo_type="dataset")
    except:
        print("\033[93m")
        print("Error downloading HSSD dataset.")
        print("Check if you have access to the HSSD dataset.")
        print("Visit https://huggingface.co/datasets/hssd/hssd-models for more information")
        print("\033[0m")
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    
    # Debug: Print first few instances to see what we're trying to download
    print(f"\n[DEBUG] Total instances to download: {len(metadata)}")
    print(f"[DEBUG] First 5 file_identifiers:")
    for i, idx in enumerate(list(metadata.index)[:5]):
        print(f"  {i+1}. {idx}")
    print(f"[DEBUG] Metadata columns: {list(metadata.columns)}")
    print()
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
        tqdm(total=len(metadata), desc="Downloading") as pbar:
        def worker(instance: str) -> str:
            try:
                # Check if file already exists
                local_file_path = os.path.join(output_dir, 'raw', instance)
                if os.path.exists(local_file_path):
                    print(f"[DEBUG] File already exists, skipping download: {instance}")
                    sha256 = get_file_hash(local_file_path)
                    pbar.update()
                    return sha256
                
                # Check if instance looks like a URL
                if instance.startswith('http://') or instance.startswith('https://'):
                    print(f"\n[WARNING] Instance looks like a URL, not a filename: {instance}")
                    print(f"  Expected a filename but got a URL. This may indicate incorrect metadata.")
                    pbar.update()
                    return None
                
                print(f"[DEBUG] Attempting to download: repo_id='hssd/hssd-models', filename='{instance}'")
                huggingface_hub.hf_hub_download(repo_id="hssd/hssd-models", filename=instance, repo_type="dataset", local_dir=os.path.join(output_dir, 'raw'))
                sha256 = get_file_hash(local_file_path)
                pbar.update()
                print(f"[DEBUG] Successfully downloaded {instance}, sha256: {sha256}")
                return sha256
            except Exception as e:
                pbar.update()
                print(f"\n[ERROR] Error downloading '{instance}':")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                return None
            
        sha256s = executor.map(worker, metadata.index)
        executor.shutdown(wait=True)

    for k, sha256 in zip(metadata.index, sha256s):
        if sha256 is not None:
            if sha256 == metadata.loc[k, "sha256"]:
                downloaded[sha256] = os.path.join('raw', k)
            else:
                print(f"Error downloading {k}: sha256s do not match")

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
