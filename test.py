import json

with open("data/train_dataset_ft3.json", "r") as file:
    dataset = json.load(file)
    print(len(dataset))