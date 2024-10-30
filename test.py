# import sys
# import json
# import requests

# provider = {
#     "option": "selfhosted",
#     "instance_name": "Remote_Backend_LLaMA 3.1 8B",
#     "temperature": 0,
#     "data": {
#         "hostserver_url": "http://127.0.0.1:1234/v1",
#         "model": sys.argv[-2],
#         "threshold_retrain_job": 100
#     }
# }

# system_content = "You are an assistant that strictly and exclusively extracts geographic references mentioned in the user-input. For each location, provide the exact place-name as it appears in the input, along with its latitude and longitude, as a JSON object (e.g., { 'name': 'place-name', 'position': [latitude, longitude] }). Only return locations mentioned in the text. Under no circumstances should you add or generate locations not present in the text. The list must only contain the exact places mentioned and must be as precise as possible. Please return the result in JSON format without any explanations or labels."

# with open(f"data/lgl.json") as file:
#     dataset = json.load(file)

# data = dataset[int(sys.argv[-1])]

# response = requests.post(
#     url=provider["data"]["hostserver_url"] + '/chat/completions',
#     json={
#         "model": provider["data"]["model"],
#         "messages": [
#             {
#                 "role": "system",
#                 "content": system_content,
#             },
#             {
#                 "role": "user",
#                 "content": data["text"],
#             }
#         ],
#         "response_format": {
#             "type": "json_schema",
#             "json_schema": {
#                 "name": "georeferences",
#                 "strict": "true",
#                 "schema": {
#                     "type": "array",  
#                     "items": {       
#                         "type": "object",
#                         "properties": {
#                             "name": {
#                                 "type": "string"
#                             },
#                             "position": {
#                                 "type": "array",  
#                                 "items": {
#                                     "type": "number",
#                                 },
#                                 "minItems": 2,
#                                 "maxItems": 2
#                             }
#                         },
#                         "required": ["name", "position"]
#                     }
#                 }
#             }
#         },
#         "temperature": provider["temperature"],
#         "max_tokens": -1,
#         "stream": False
#     }, 
#     headers={"Content-Type": "application/json"},
#     timeout=10 * 60,
# )
# output=response.json()

# print(json.dumps(json.loads(output["choices"][0]["message"]["content"]), indent=2))

# del output

# import json

# with open("data/lgl.json", "r") as file:
#     dataset = json.load(file)
#     print(len(dataset))

import xml.etree.ElementTree as ET

# Load the XML file
file_path = 'data/lgl.xml'
tree = ET.parse(file_path)
root = tree.getroot()

# Count the number of articles in the XML
num_articles = len(root.findall('article'))
print(num_articles)