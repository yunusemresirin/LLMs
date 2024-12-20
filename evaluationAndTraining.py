import json
import time
import asyncio
import requests
from math import radians, sin, cos, sqrt, atan2

async def calculate_distance(coord1, coord2):
    # Function to calculate distance between two coordinates
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of Earth in kilometers
    return distance

async def compute_precision_recall_f1(instances, _truth: str, _pred: str):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    matched_coordinates = []

    for instance in instances:
        ground_truth = instance[_truth]
        predicted = instance[_pred]
        matched_ground_truth = set()  # To keep track of matched ground truth elements
        
        print()

        true_positives = 0
        for pred_key, pred_coord in predicted.items():
            matched = False
            for gt_key, gt_coord in ground_truth.items():
                if pred_key.lower() in gt_key.lower() or gt_key.lower() in pred_key.lower():
                    if gt_key not in matched_ground_truth:  # Ensure we don't count the same ground truth multiple times
                        true_positives += 1
                        matched_ground_truth.add(gt_key)
                        matched_coordinates.append((pred_coord, gt_coord))
                        matched = True
                        break
            
            # False positives are elements in predicted that did not match any ground truth element
            if not matched:
                total_false_positives += 1
        
        # False negatives are ground truth elements that did not match any predicted element
        total_false_negatives += len(ground_truth) - len(matched_ground_truth)
        total_true_positives += true_positives
    
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score, matched_coordinates

async def calculate_A_at_k(matched_coordinates, k):
    # Function to calculate accuracy at k (A@k)
    correct_matches = 0
    for pred_coord, truth_coord in matched_coordinates:
        if await calculate_distance(pred_coord, truth_coord) <= k:
            correct_matches += 1

    accuracy_at_k = (correct_matches / len(matched_coordinates)) * 100 if matched_coordinates else 0
    return accuracy_at_k

async def evaluationAndTraining(version: int, geoparser: str):
    with open(f"data/lgl.json") as file:
        dataset = json.load(file)

    batch = dataset[(100*(version-1)):(100*version)]
    
    precision, recall, f1_score, matched_coordinates = await compute_precision_recall_f1(batch, "locations", f"pred_{geoparser}")

    with open(f"output_FT{version}.txt", "a") as file:
        print(f"----------------- {geoparser} -----------------", file=file)
        print(f"precision:\t{precision}",   file=file)
        print(f"recall: \t{recall}",        file=file)
        print(f"f1_score:\t{f1_score}\n",   file=file)
        print(f"A@k - 10:\t{round(await calculate_A_at_k(matched_coordinates, 10),2)}", file=file)
        print(f"A@k - 161:\t{round(await calculate_A_at_k(matched_coordinates, 161),2)}\n", file=file)

if __name__ == "__main__":
    import sys

    asyncio.run(evaluationAndTraining(int(sys.argv[-2]), sys.argv[-1]))