import os
import json
import time
import asyncio
import requests

system_content = "You are an assitant that strictly extracts geographic references from the input. For each location, provide the place-name (exactly as in the text), the latitude and the longitude of the place as a json-object, like { name: place-name, position: [latitude, longitude] }. Create a json-list out of these objects. In the list, there should be no repetitive places with the same place-name. Only extract the places that are mentioned in the input. The positions should be as precise as possible. Please only return the json-string with no explanation or further information and as a normal text without labeling it as json."
'Command for LLM-system'

async def geoparseTextSelfHosted(text: str, provider: dict):
    '''
        Geoparsing text with a selfhosted LLM
    '''
    response = requests.post(
        timeout=10 * 60,
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
    )
    output=response.json()
    
    return json.loads(output['choices'][0]['message']['content'])

# 557 data objects
async def geoparseData(version: int):
    with open("data/lgl.json", "r") as file:
        dataset = json.load(file)

    provider = {
        "option": "selfhosted",
        "instance_name": "Remote_Backend_LLaMA 3.1 8B",
        "temperature": 0,
        "data": {
            "hostserver_url": "http://127.0.0.1:1234/v1",
            "model": "Llama-3.1-8B-Instruct-finetuned/gguf/unsloth.Q4_K_M.gguf",
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

        for i in range(100*(version-1)+34, 100*version):
            data = dataset[i]

            count = i-100*(version-1)
            print(count)

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
                file.write(",\n")
            except requests.exceptions.Timeout:
                with open("output.txt", "a") as output_file:
                    print(f"Skipping: {i}", file=output_file)
                continue
            except Exception as e:
                print(e)
                break

        file.seek(file.tell() - 2)
        file.write("\n]")

if __name__ == "__main__": 
    start_time = time.time()
    asyncio.run(geoparseData(4))
    end_time = time.time()
    print(f"elapsed_time: {end_time-start_time}")

    # FT2: Average elapsed time: 3077.05 ~ 51.3 min
    # FT3: Average elapsed time: 17776.91 ~ 296.28 min ~ 4.93 h