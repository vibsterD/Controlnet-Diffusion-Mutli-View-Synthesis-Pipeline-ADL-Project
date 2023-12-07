import os
import json

### Dataset with jsonl like fill50k

with open("/raid/home/vibhu20150/Datasets/IIITD-20K/Filtered.json", "rb") as f:
    data = json.load(f)

print(len(data))

jsonl_file_path = "IIITD20K/train.jsonl"

# Save data to the JSONL file
with open(jsonl_file_path, "w") as jsonl_file:
    for i in data.keys():
        if int(i) % 1000 == 0:
            print(i)
        if int(i) >= 100:
            json_line = json.dumps({
                "text":data[i]['Description 1'],
                "image":"images/"+data[i]['Image ID']+"."+data[i]["Image URL"].split(".")[-1],
                "conditioning_image":"conditioning_images/"+data[i]['Image ID']+".png",
            })
            jsonl_file.write(json_line + "\n")
    