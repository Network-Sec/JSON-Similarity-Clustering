#!/usr/bin/env python3

import random
import string
import json
from pprint import pprint
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # or RandomForestRegressor depending on your task
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def flatten_json(y):
    """Flatten a JSON object to a flat dictionary."""
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def generate_random_key():
    """Generate a random key name."""
    return ''.join(random.choices(string.ascii_letters, k=random.randint(5, 10)))

def generate_random_value(value_type):
    """Generate a random value based on the specified type."""
    if value_type == "string":
        return ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(5, 20)))
    elif value_type == "float":
        return round(random.uniform(1, 100), 2)
    elif value_type == "int":
        return random.randint(1, 100)
    elif value_type == "array":
        return [generate_random_value(random.choice(["string", "float", "int"])) for _ in range(random.randint(2, 5))]

def generate_structure(level=4, max_children=3):
    """Recursively generate a fixed structure with random key names and types."""
    if level == 0:
        return generate_random_value(random.choice(["string", "float", "int", "array"]))
    
    obj = {}
    for _ in range(random.randint(1, max_children)):
        key = generate_random_key()
        obj[key] = generate_structure(level-1, max_children)
    return obj

def generate_json_objects(n=10):
    """Generate a list of JSON objects based on a random but fixed structure."""
    base_structure = generate_structure()
    json_objects = []
    for _ in range(n):
        # Clone the base structure and modify it slightly for each new object
        obj = json.loads(json.dumps(base_structure))  # Deep copy
        json_objects.append(obj)
    
    # Introduce variations in the number of keys and increments of some keys
    for obj in json_objects:
        if random.choice([True, False]):
            key_to_add = generate_random_key()
            obj[key_to_add] = generate_random_value(random.choice(["string", "float", "int", "array"]))
        if random.choice([True, False]):
            key_to_remove = random.choice(list(obj.keys()))
            obj.pop(key_to_remove, None)
    
    return json_objects

def find_keys_with_path(obj, path=[]):
    """Recursively find all keys in a nested dictionary along with their path."""
    keys_with_path = []
    for k, v in obj.items():
        current_path = path + [k]
        if isinstance(v, dict):
            keys_with_path.extend(find_keys_with_path(v, current_path))
        else:
            keys_with_path.append((current_path, v))
    return keys_with_path

def introduce_similarities(json_objects, n_similar=5):
    """Introduce similarities between objects correctly, considering nested structures, with fixed sampling error."""
    human_readable_tracker = []
    processing_tracker = {}

    # Flatten the structure of the first object to get paths for all keys
    keys_with_paths = find_keys_with_path(json_objects[0])
    selected_paths = random.sample(keys_with_paths, min(len(keys_with_paths), n_similar))
    
    for path, _ in selected_paths:
        source_obj_index = random.randint(0, len(json_objects) - 1)
        source_obj = json_objects[source_obj_index]
        
        # Navigate to the correct key in the source object
        for step in path[:-1]:
            source_obj = source_obj.get(step, {})
        source_value = source_obj.get(path[-1], None)
        
        available_targets = [x for x in range(len(json_objects)) if x != source_obj_index]
        target_objs = random.sample(available_targets, min(len(available_targets), random.randint(3, 10)))
        
        for target in target_objs:
            target_obj = json_objects[target]
            
            # Navigate to the correct key in the target object and update the value
            target_obj_nested = target_obj
            for step in path[:-1]:
                target_obj_nested = target_obj_nested.get(step, {})
            if path[-1] in target_obj_nested:  # Update the value only if the key exists
                target_obj_nested[path[-1]] = source_value
                human_readable_tracker.append(f"Object{source_obj_index} is similar to Object{target} in key path {'->'.join(path)}")
                
                # Update processing tracker with structured data
                processing_key = "->".join(path)
                if processing_key not in processing_tracker:
                    processing_tracker[processing_key] = []
                processing_tracker[processing_key].append((source_obj_index, target))

    return json_objects, human_readable_tracker, processing_tracker

def parse_similarity(similarity_description):
    # Extracting IDs assuming format "ObjectXXXX is similar to ObjectYYYY..."
    parts = similarity_description.split(' ')
    obj1 = parts[0].replace('Object', '')  # Removes 'Object' prefix and extracts the ID
    obj2 = parts[4].replace('Object', '')  # Same here for the second object
    return obj1, obj2

def main():
    # Generate and modify JSON objects
    json_objects = generate_json_objects(10000)
    json_objects, human_readable_tracker, similarity_tracker = introduce_similarities(json_objects, 15)

    # Displaying a snippet of the human-readable similarity tracker for verification
    pprint(human_readable_tracker[:5])

    # Randomly pick n objects directly from the list of JSON objects
    n = 300
    random_indices = random.sample(range(len(json_objects)), n)
    random_subset_objects = [json_objects[i] for i in random_indices]

    # Flatten and process this randomly selected subset
    flattened_random_subset = [flatten_json(obj) for obj in random_subset_objects]
    random_subset_df = pd.DataFrame(flattened_random_subset)

    # Fill missing values for the randomly selected subset
    for col in random_subset_df.columns:
        if random_subset_df[col].dtype in ['float64', 'int64']:
            random_subset_df[col] = random_subset_df[col].fillna(random_subset_df[col].median())
        elif random_subset_df[col].dtype == 'object':
            mode_value = random_subset_df[col].mode().iloc[0] if not random_subset_df[col].mode().empty else "Unknown"
            random_subset_df[col] = random_subset_df[col].fillna(mode_value)

    # Display the cleaned random subset DataFrame
    print(random_subset_df.head())

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(random_subset_df.select_dtypes(include=['float64', 'int64']))

    # Perform KMeans clustering on the PCA results
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(pca_result)
    clusters = kmeans.labels_

    # Calculate silhouette score
    score = silhouette_score(pca_result, clusters)
    print(f"Silhouette Score: {score}")

    # Visualize PCA results
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    plt.title('PCA of JSON Objects')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # Mapping objects to clusters
    object_cluster_map = {random_indices[i]: clusters[i] for i in range(len(clusters))}

    # Check if similar objects are in the same cluster
    for similarity in human_readable_tracker[:5]:  # Assuming this is your intended list
        obj1, obj2 = parse_similarity(similarity)
        cluster_msg = "the same cluster." if object_cluster_map.get(obj1) == object_cluster_map.get(obj2) else "different clusters."
        print(f"Object{obj1} and Object{obj2} are in {cluster_msg}")

if __name__ == '__main__':
    main()
