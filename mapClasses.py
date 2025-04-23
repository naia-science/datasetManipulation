import os
import argparse
import ast
from typing import OrderedDict

def map_and_remove_classes(folder_path, class_mapping, num_classes):
    default_mapping = {i: i for i in range(num_classes)}
    default_mapping.update(class_mapping)
    sorted_class_mapping_temp = OrderedDict(sorted(default_mapping.items()))
    sorted_class_mapping =sorted_class_mapping_temp.copy()
    # print(sorted_class_mapping_temp)
    for k in class_mapping.keys():
        for k2,v2 in reversed(sorted_class_mapping_temp.items()):
            if v2 > k:
                sorted_class_mapping[k2] -= 1
    print(sorted_class_mapping)
    num_removed = 0
    num_mapped = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []


            for line in lines:
                parts = line.split()
                class_id = int(parts[0])

                new_class_id = sorted_class_mapping[class_id]
                if new_class_id == -1:
                    num_removed += 1
                    continue  # Remove the class
                if class_id in class_mapping:
                    num_mapped += 1
                parts[0] = str(new_class_id)
                new_lines.append(' '.join(parts) + '\n')

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
    print(f"removed {num_removed} labels, mapped {num_mapped} labels")

def main():
    parser = argparse.ArgumentParser(description="Map and remove classes in YOLOv8 segmentation labels.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the .txt files.")
    parser.add_argument("class_mapping", type=str, help="Dictionary as a string mapping class IDs, e.g., '{8: 11, 5: -1}'.")
    parser.add_argument("num_classes", type=int, help="Total number of classes.")

    args = parser.parse_args()

    # Convert the class_mapping string to a dictionary
    class_mapping = ast.literal_eval(args.class_mapping)

    map_and_remove_classes(args.folder_path, class_mapping, args.num_classes)

if __name__ == "__main__":
    main()
