# py -3.8 test_1.py

import pandas as pd
import os

data = {
    "name":["prashant","usha","ashok","meenakshi"],
    "age":[10,20,30,40],
    "city":["kanpur","delhi","lucknow","meerut"]
}

df = pd.DataFrame(data)


#adding new row
first_row_loc = {"name":"shakkarman","age":100,"city":"india"}
df.loc[len(df.index)] = first_row_loc

second_row_loc = {"name":"paaraman","age":100,"city":"india"}
df.loc[len(df.index)] = second_row_loc

data_dir = "test_data"
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "test_data.csv")

df.to_csv(file_path, index=False)
print(f"csv file saved to path: {file_path}")