import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of objects to download in each batch (default: 50)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, batch_size=50, **kwargs):    
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    
    # download objects in batches with error handling
    file_paths = {}
    failed_objects = []
    total_objects = len(annotations)
    
    # Split into batches
    annotation_list = list(annotations.iterrows())
    num_batches = (len(annotation_list) + batch_size - 1) // batch_size
    
    print(f"Total objects to download: {total_objects}")
    print(f"Downloading in {num_batches} batches of size {batch_size}...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(annotation_list))
        batch_annotations = pd.DataFrame([row[1] for row in annotation_list[start_idx:end_idx]])
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing objects {start_idx + 1} to {end_idx}...")
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                batch_file_paths = oxl.download_objects(
                    batch_annotations,
                    download_dir=os.path.join(output_dir, "raw"),
                    save_repo_format="zip",
                )
                file_paths.update(batch_file_paths)
                print(f"Batch {batch_idx + 1}: Successfully downloaded {len(batch_file_paths)} objects.")
                break
            except Exception as e:
                print(f"Batch {batch_idx + 1}, Attempt {retry + 1}/{max_retries}: Error - {e}")
                
                if retry < max_retries - 1:
                    import time
                    wait_time = (retry + 1) * 5
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # Last retry failed, record failed objects
                    print(f"Batch {batch_idx + 1}: Failed after {max_retries} attempts. Recording failed objects...")
                    for idx, row in batch_annotations.iterrows():
                        if idx not in file_paths:
                            failed_objects.append(idx)
    
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"Total objects: {total_objects}")
    print(f"Successfully downloaded: {len(file_paths)}")
    print(f"Failed: {len(failed_objects)}")
    
    if failed_objects:
        print(f"\nFailed object identifiers (first 20):")
        for obj in failed_objects[:20]:
            print(f"  - {obj}")
        if len(failed_objects) > 20:
            print(f"  ... and {len(failed_objects) - 20} more")
        
        # Save failed objects to a file for later retry
        failed_file = os.path.join(output_dir, 'failed_downloads.txt')
        with open(failed_file, 'w') as f:
            for obj in failed_objects:
                f.write(f"{obj}\n")
        print(f"\nFailed object identifiers saved to: {failed_file}")
    print(f"{'='*60}\n")
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        if k in metadata.index:
            sha256 = metadata.loc[k, "sha256"]
            downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile
    
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
                    if local_path.startswith('raw/github/repos/'):
                        path_parts = local_path.split('/')
                        file_name = os.path.join(*path_parts[5:])
                        zip_file = os.path.join(output_dir, *path_parts[:5])
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            file = os.path.join(tmp_dir, file_name)
                            record = func(file, sha256)
                    else:
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
