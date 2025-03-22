from datetime import datetime
import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def convert_datetime_format(data):
    """
    Convert non-standard datetime format to standard ISO 8601 format.
    """
    for record in data:
        try:
            original_datetime = record.get("datetime")
            if original_datetime:
                parsed_datetime = datetime.strptime(original_datetime, "%a %b %d %H:%M:%S %Y %z")
                record["datetime"] = parsed_datetime.isoformat() 
        except Exception as e:
            print(f"Error converting datetime for record {record.get('commit_id', 'unknown')}: {e}")
    return data

def truncate_record(record, max_record_size):
    """
    Truncate the `diff` field of a record to ensure its size does not exceed max_record_size.
    """
    record_size = len(json.dumps(record).encode('utf-8'))
    if record_size > max_record_size:
        max_diff_length = max_record_size - (record_size - len(record["diff"].encode("utf-8")))

        if max_diff_length > 0:
            truncated_diff = record["diff"][:max_diff_length]

            # **Try to fix JSON structure**
            if truncated_diff.count("{") > truncated_diff.count("}"):
                truncated_diff += "}" 
            if truncated_diff.count("[") > truncated_diff.count("]"):
                truncated_diff += "]" 

            try:
                json.loads(json.dumps({"diff": truncated_diff})) 
                record["diff"] = truncated_diff
            except json.JSONDecodeError:
                record["diff"] = "... (truncated and fixed)"  # Replace with a placeholder if invalid

            # print(f"Truncated diff for {record['commit_id']} to {len(record['diff'])} characters.")
        else:
            record["diff"] = "... (diff removed)"  # Avoid broken JSON structure
            # print(f"Removed diff for {record['commit_id']} as it exceeds max record size.")

    return record

def split_file(input_file, output_dir, max_size=98 * 1024 * 1024, max_record_size=49 * 1024 * 1024):
    """
    Split a large JSON file into multiple smaller files, each up to max_size bytes.
    Convert datetime to ISO format.
    If a single record exceeds max_record_size, truncate the diff field.
    """
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {input_file}: {e}")
        return

    data = convert_datetime_format(data)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    current_batch = []
    current_size = 0
    file_index = 0

    for record in data:
        # Truncate diff if the record exceeds the limit
        truncate_record(record, max_record_size)

        record_size = len(json.dumps(record).encode('utf-8'))
        if current_size + record_size > max_size:
            output_file = os.path.join(output_dir, f"{base_filename}_{file_index}.json")
            if not os.path.exists(output_file):
                with open(output_file, "w") as out_f:
                    json.dump(current_batch, out_f, indent=4)
                print(f"Saved {len(current_batch)} records to {output_file}")
            current_batch = []
            current_size = 0
            file_index += 1

        current_batch.append(record)
        current_size += record_size

    if current_batch:
        output_file = os.path.join(output_dir, f"{base_filename}_{file_index}.json")
        if not os.path.exists(output_file):
            with open(output_file, "w") as out_f:
                json.dump(current_batch, out_f, indent=4)
            print(f"Saved {len(current_batch)} records to {output_file}")

def is_repo_fully_processed(output_dir, expected_files):
    if not os.path.exists(output_dir):
        return False
    split_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    return len(split_files) >= expected_files

def process_repo(args):
    row, base_dir, max_size, max_record_size = args
    repo = f"{row['owner']}@@{row['repo']}"
    input_file = os.path.join(base_dir, f"repo2commits_diff/{repo}.json")
    output_dir = os.path.join(base_dir, f"repo2commits_diff/split_{repo}")

    expected_files = os.path.getsize(input_file) // max_size + 1 if os.path.exists(input_file) else 1

    if is_repo_fully_processed(output_dir, expected_files):
        print(f"Skipping {repo}, already processed.")
        return

    if os.path.exists(input_file):
        file_size = os.path.getsize(input_file)
        if file_size <= max_size:
            # If the file is smaller than max_size, just convert datetime and save directly
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{repo}_0.json")
            # if not os.path.exists(output_file):
            with open(input_file, "r") as src:
                try:
                    data = json.load(src)
                except Exception as e:
                    print(f"Error loading JSON from {input_file}: {e}")
                    return
                data = convert_datetime_format(data)  # Convert datetime
                with open(output_file, "w") as dest:
                    json.dump(data, dest, indent=4)
                print(f"Copied {input_file} to {output_file} (size: {file_size} bytes)")
        else:
            split_file(input_file, output_dir, max_size, max_record_size)
    else:
        print(f"Input file for {repo} does not exist: {input_file}")

def process_all_repos(input_csv, base_dir, max_size=98 * 1024 * 1024, max_record_size=98 * 1024 * 1024, num_processes=4):
    """
    Perform split for each owner@@repo listed in the CSV file.
    """
    df = pd.read_csv(input_csv)
    args_list = [(row, base_dir, max_size, max_record_size) for _, row in df.iterrows()]
    
    # Use the specified number of processes
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_repo, args_list), total=len(args_list), desc="Processing repos"))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python split_multiple_repos.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]
    input_csv = f"../csv/{dataset}.csv"
    base_dir = "."
    num_processes = 8
    process_all_repos(input_csv, base_dir, num_processes=num_processes)