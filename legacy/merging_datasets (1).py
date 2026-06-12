# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:08:54 2025

@author: Theidel
"""


import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename to extract base name and chunk/num identifiers.
    
    Returns:
        base_name: measurement identifier without _numX_chunkX
        num: repetition number
        chunk: chunk number
    """
    # Handle both .json and .pkl extensions
    name = Path(filename).stem
    
    # Pattern to match _numX_chunkX at the end
    pattern = r'^(.+?)(?:_num(\d+))?_chunk(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        base_name = match.group(1)
        num = int(match.group(2)) if match.group(2) else 0
        chunk = int(match.group(3))
        return base_name, num, chunk
    else:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")


def get_all_keys(d, parent_key=''):
    """Recursively get all keys from nested dictionary"""
    keys = set()
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        keys.add(full_key)
        if isinstance(v, dict):
            keys.update(get_all_keys(v, full_key))
    return keys

def parse_filename_ignore_date(filename):
    """
    Parses a filename to extract its base name, ignoring a date prefix.
    Example: "15102025_VO2_data_num1_chunk2.pkl" -> ("VO2_data", 1, 2)
    """
    # Use regex to remove an 8-digit prefix (like a date) from the start
    base_with_chunks = re.sub(r'^\d{8}_', '', filename.stem)
    
    try:
        # Split the remaining name to find the base part and the chunk numbers
        parts = base_with_chunks.split('_')
        num_index = -1
        # Find the part that starts with 'num'
        for i, part in enumerate(parts):
            if part.startswith('num'):
                num_index = i
                break
        
        if num_index == -1:
            raise ValueError("Filename does not contain '_num' part")
            
        # The base name is everything before the '_num' part
        base_name = '_'.join(parts[:num_index])
        
        # Extract the numbers
        num = int(parts[num_index].replace('num', ''))
        chunk = int(parts[num_index + 1].replace('chunk', ''))
        
        return base_name, num, chunk
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse filename '{filename}'. "
                         "Expected format like 'DATE_BASENAME_numX_chunkY.ext'")
        

def group_chunks_by_measurement(directories, file_extension=".json"):
    """
    Group all files from one or more directories by their base measurement name,
    ignoring any date prefixes in the filenames.
    
    Args:
        directories: A single path or a list of paths to directories.
        file_extension: File extension to look for (.json or .pkl)
    
    Returns:
        dict: {base_name: [(num, chunk, filepath), ...]}
    """
    if not isinstance(directories, list):
        directories = [directories]

    measurements = defaultdict(list)
    pattern = f"*{file_extension}"
    
    for directory in directories:
        current_dir = Path(directory)
        if not current_dir.is_dir():
            print(f"Warning: '{current_dir}' is not a valid directory. Skipping.")
            continue

        for data_file in current_dir.glob(pattern):
            try:
                # THIS IS THE ONLY LINE THAT CHANGES!
                base_name, num, chunk = parse_filename_ignore_date(data_file)
                
                measurements[base_name].append((num, chunk, data_file))
            except ValueError as e:
                # We update the print statement to be more informative
                print(f"Skipping file '{data_file.name}': {e}")
    
    for base_name in measurements:
        measurements[base_name].sort(key=lambda x: (x[0], x[1]))
    
    return dict(measurements)


def merge_two_dicts(d1, d2):
    """Merge two data dictionaries"""
    all_keys = get_all_keys(d1)
    assert all_keys == get_all_keys(d2), "Dictionaries must have the same keys"
    
    merged_dict = {}
    
    for k in all_keys:
        # Skip if not a nested key
        if '.' not in k:
            continue
            
        parent_key, daughter_key = k.split(".", 1)
        
        # Ensure parent key exists
        if parent_key not in merged_dict:
            merged_dict[parent_key] = {}
        
        # Handle different key types
        if parent_key == 'Parameters':
            if daughter_key == "duration":
                merged_dict[parent_key][daughter_key] = np.add(
                    d1[parent_key][daughter_key],
                    d2[parent_key][daughter_key]
                )
            elif daughter_key == "savepath":
                merged_dict[parent_key][daughter_key] = ""
            else:
                merged_dict[parent_key][daughter_key] = d1[parent_key][daughter_key]
                
        elif parent_key == 'Correlation':
            data1, data2 = d1[parent_key][daughter_key], d2[parent_key][daughter_key]
            
            assert len(data1) == len(data2), f"Length mismatch for {k}"
            assert len(data1[0]) == len(data2[0]), f"data[0] length mismatch for {k}"
            assert len(data1[1]) == len(data2[1]), f"data[1] length mismatch for {k}"
            
            merged_dict[parent_key][daughter_key] = [data1[0], np.add(data1[1], data2[1])]
            
        elif parent_key == 'Countrate':
            countrate = np.mean([d1[parent_key][daughter_key][0], d2[parent_key][daughter_key][0]])
            total_counts = np.add(d1[parent_key][daughter_key][1], d2[parent_key][daughter_key][1])
            merged_dict[parent_key][daughter_key] = [countrate, total_counts]
    
    assert get_all_keys(merged_dict) == all_keys, "Merged dict keys don't match original"
    return merged_dict


def merge_multiple_dicts(dict_list):
    """
    Merge multiple dictionaries sequentially using merge_two_dicts.
    
    Args:
        dict_list: List of dictionaries to merge
    
    Returns:
        Merged dictionary
    """
    if len(dict_list) == 0:
        raise ValueError("Cannot merge empty list of dictionaries")
    
    if len(dict_list) == 1:
        return dict_list[0]
    
    # Start with the first two dictionaries
    result = merge_two_dicts(dict_list[0], dict_list[1])
    
    # Sequentially merge the rest
    for d in dict_list[2:]:
        result = merge_two_dicts(result, d)
    
    return result


def save_pickle(data, filepath):
    """Save data using pickle - fast for large files"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_data(filepath):
    """Load data file - automatically detects JSON or pickle format"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.pkl':
        return load_pickle(filepath)
    elif filepath.suffix == '.json':
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
 
        
def merge_all_chunks(mdirs, output_subdir="merged", input_format=".pkl", output_format=".pkl"):
    """
    Merge all chunks for each base measurement and save in a subdirectory.
    
    Args:
        mdirs: Path to a directory or a list of paths to directories containing chunk files. # CHANGED
        output_subdir: Name of subdirectory to save merged files
        input_format: Format of input files (.json or .pkl)
        output_format: Format for output files (.json or .pkl)
    
    Returns:
        Dictionary with information about merged files
    """

    if not isinstance(mdirs, list):
        mdirs = [mdirs]
    
    #  Use the first directory in the list for the output path
    output_dir = Path(mdirs[0]) / output_subdir
    output_dir.mkdir(exist_ok=True)
   
    measurements = group_chunks_by_measurement(mdirs, file_extension=input_format)
    
    if not measurements:

        print(f"No {input_format} chunk files found in the provided directories!")
        return {}
    
    print(f"\nFound {len(measurements)} base measurement(s):")
    for base_name, chunks in measurements.items():
        print(f"  {base_name}: {len(chunks)} chunks")
    
    print(f"\nMerging all chunks and saving to '{output_dir.relative_to(Path(mdirs[0]).parent)}/' as {output_format} files...")
    
    results = {}
    
    for base_name, chunk_list in measurements.items():
        print(f"\n  Processing: {base_name}")
        print(f"    Loading {len(chunk_list)} chunks...")
        
        # Load all chunks
        all_dicts = []
        for num, chunk, filepath in chunk_list:
            print(f"      Loading num{num}_chunk{chunk} from '{filepath.parent.name}'...")
            try:
                data = load_data(filepath)
            except Exception as e:
                print(f"Error while loading {filepath}: {e}")
                continue
            all_dicts.append(data)
        
        if not all_dicts:
            print("    No chunks were successfully loaded. Skipping merge.")
            continue

        print(f"    Merging...")
        # Merge all chunks
        merged_data = merge_multiple_dicts(all_dicts)
        
        # Save merged result
        output_filename = f"{base_name}_merged{output_format}"
        output_filepath = output_dir / output_filename
        
        merged_data["Parameters"]["savepath"] = str(output_filepath)
        
        print(f"    Saving to {output_filename}...")
        
        if output_format == ".pkl":
            save_pickle(merged_data, output_filepath)
        elif output_format == ".json":
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8,
                                        np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    return json.JSONEncoder.default(self, obj)
            
            with open(output_filepath, 'w') as f:
                json.dump(merged_data, f, cls=NumpyEncoder)
        
        results[base_name] = {
            'num_chunks': len(chunk_list),
            'output_file': str(output_filepath),
            'chunks_merged': [(num, chunk) for num, chunk, _ in chunk_list]
        }
        
        print(f"    ✓ Done!")
        
        # Clear memory
        del all_dicts
        del merged_data
    
    print(f"\n{'='*60}")
    print(f"✓ All measurements merged successfully!")
    print(f"✓ Merged files saved in: {output_dir}")
    print(f"{'='*60}")
    
    return results



# Main execution
if __name__ == "__main__":
    # Single directory or multiple directories in a list
    # data is saved in a folder in the first directory
    mdir =[r"F:\October 25\icfo data\20102025_VO2_timing25000ps_1000bins\20102025_VO2_timing25000ps_1000bins",
           r"F:\October 25\icfo data\15102025_VO2_timing25000ps_1000bins\15102025_VO2_timing25000ps_1000bins",
           r"F:\October 25\icfo data\17102025_VO2_timing25000ps_1000bins\17102025_VO2_timing25000ps_1000bins"]

    mdir = [r"F:\October 25\24102025_CdTe110_timing500ps_2000bins"]
    results = merge_all_chunks(mdir, output_subdir="merged", input_format=".json")
    
    # Print summary
    print("\nSUMMARY:")
    for base_name, info in results.items():
        print(f"\n{base_name}:")
        print(f"  Chunks merged: {info['num_chunks']}")
        print(f"  Output: {info['output_file']}")
    