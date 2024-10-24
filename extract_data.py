import os
import json
import time
import asyncio
import requests

system_content = "You are an assistant that strictly and exclusively extracts geographic references mentioned in the user-input. For each location, provide the exact place-name as it appears in the input, along with its latitude and longitude, as a JSON object (e.g., { 'name': 'place-name', 'position': [latitude, longitude] }). Only return locations mentioned in the text. Under no circumstances should you add or generate locations not present in the text. The list must only contain the exact places mentioned and must be as precise as possible. Please return the result in JSON format without any explanations or labels."
'Command for LLM-system'

async def geoparseTextSelfHosted(text: str, provider: dict):
    '''
        Geoparsing text with a selfhosted LLM
    '''
    response = requests.post(
        url=provider["data"]["hostserver_url"] + '/chat/completions',
        json={
            "model": provider["data"]["model"],
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "georeferences",
                    "strict": "true",
                    "schema": {
                        "type": "array",  
                        "items": {       
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string"
                                },
                                "position": {
                                    "type": "array",  
                                    "items": {
                                        "type": "number",
                                    },
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            },
                            "required": ["name", "position"]
                        }
                    }
                }
            },
            "temperature": provider["temperature"],
            "max_tokens": -1,
            "stream": False
        }, 
        headers={"Content-Type": "application/json"},
        timeout=10 * 60,
    )
    output=response.json()
    
    return json.loads(output['choices'][0]['message']['content'])

# 557 data objects
async def geoparseData(model: str, version: int):
    start_time = time.time()

    with open("data/lgl.json", "r") as file:
        dataset = json.load(file)

    provider = {
        "option": "selfhosted",
        "instance_name": "Remote_Backend_LLaMA 3.1 8B",
        "temperature": 0,
        "data": {
            "hostserver_url": "http://127.0.0.1:1234/v1",
            "model": model,
            "threshold_retrain_job": 100
        }
    }
    
    if not os.path.exists(f"data/train_dataset_ft{version}.json"):
        with open(f"data/train_dataset_ft{version}.json", "w") as file:
            file.write("[\n")

    with open(f"data/train_dataset_ft{version}.json", "r+") as file:
        file.seek(0, os.SEEK_END)
        pos = file.tell() - 1

        while pos > 0:
            file.seek(pos)
            char = file.read(1)
            if char == "]":
                file.seek(pos)
                break
            pos -= 1

        if pos > 1:
            file.write(",\n")

        start = 100*(version-1)
        end = start + 57
        for i in range(start, end):
            data = dataset[i]

            count = i-start
            print(count)

            skipped = 0
            try:
                predictions = await geoparseTextSelfHosted(data['text'], provider)

                async def restructure_locations(locations):
                    return {location['name']: location['position'] for location in locations}

                new_data = {
                    "id": data['id'],
                    "source": data['source'],
                    "text": data['text'],
                    "corrections": data['locations'],
                    "predictions": await restructure_locations(predictions)
                }
                
                file.write(json.dumps(new_data, indent=2))
                if count < end: 
                    file.write(",\n")
            except requests.exceptions.Timeout:
                skipped += 1
                with open(f"output_train_dataset_ft{version}.txt", "a") as output_file:
                    print(f"Skipping: {i}", file=output_file)
                continue
            except Exception as e:
                print(e)
                break

        file.seek(file.tell() - 2)
        file.write("\n]")

        end_time = time.time()
        if skipped:
            with open("output.txt", "a") as output_file:
                print(f"elapsed time: {round(end_time-start_time, 2)} s", file=file)
                print(f"----------------- Total: {skipped}", file=output_file)

if __name__ == "__main__": 
    import sys

    asyncio.run(geoparseData(model=sys.argv[-2], version=int(sys.argv[-1])))

    # FT2: Average elapsed time: 3077.05 ~ 51.3 min
    # FT3: Average elapsed time: 17776.91 ~ 296.28 min ~ 4.93 h
    # FT4 - 2/3: Average elapsed time: 10236.34 ~ 170.61 min ~ 2.84 h