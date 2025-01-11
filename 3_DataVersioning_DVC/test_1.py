# py -3.8 test_1.py

import pandas as pd
import os

data = {
    "name":["prashant","usha","ashok","meenakshi"],
    "age":[10,20,30,40],
    "city":["kanpur","delhi","lucknow","meerut"]
}

df = pd.DataFrame(data)

data_dir = "test_data"
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "test_data.csv")

df.to_csv(file_path, index=False)
print(f"csv file saved to path: {file_path}")