import urllib.request
import zipfile
import os
from pathlib import Path
import time

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" 
zip_path = "sms_spam_collection.zip" 
extracted_path = "sms_spam_collection" 
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=False):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    if test_mode:  # Try multiple times since CI sometimes has connectivity issues
        max_retries = 5
        delay = 5  # delay between retries in seconds
        for attempt in range(max_retries):
            try:
                # Downloading the file
                with urllib.request.urlopen(url, timeout=10) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
                break  # if download is successful, break out of the loop
            except urllib.error.URLError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)  # wait before retrying
                else:
                    print("Failed to download file after several attempts.")
                    return  # exit if all retries fail

    else:  # Code as it appears in the chapter
        # Downloading the file
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=False)