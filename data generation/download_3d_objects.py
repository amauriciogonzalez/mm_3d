import multiprocessing
import objaverse
import random

# Get the number of CPU cores available for parallel downloading
processes = multiprocessing.cpu_count()


lvis_annotations = objaverse.load_lvis_annotations()
print(f"Total number of classes: {len(lvis_annotations)}")


random.seed(42)

# Download 10 objects from each class
for class_name, object_uids in lvis_annotations.items():
    print(f"Class: {class_name}, Total objects: {len(object_uids)}")

    # If there are more than 10 objects, randomly sample 10 objects from this class
    if len(object_uids) > 10:
        sampled_uids = random.sample(object_uids, 10)
    else:
        # If less than 10 objects, download all available objects
        sampled_uids = object_uids

    # Download the objects using multiprocessing
    objects = objaverse.load_objects(
        uids=sampled_uids,
        download_processes=processes
    )

    print(f"Downloaded {len(objects)} objects from class '{class_name}'\n")
