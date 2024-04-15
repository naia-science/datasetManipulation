#import datasetUtils module
import datasetUtils as du

import os
import shutil
import argparse

def get_all_fresh_dataset(outputPath, roboflow_version):
    """
    Description:
        Get all datasets as fresh installs
    Usage:
        get_all_fresh_dataset(outputPath)
    Arguments:
        outputPath: Output path for merged dataset
    """
    print('Deleting previous versions of datasets, starting fresh...')
    du.delete_roboflow_dataset(roboflow_version)
    du.delete_TACO_dataset()
    try:
        du.delete_merged_datasets(outputPath)
    except:
        print('No merged datasets to delete')

def dl_roboflow_dataset(roboflow_version):
    """
    Description:
        Download the Roboflow dataset
    Usage:
        dl_roboflow_dataset(roboflow_version)
    Arguments:
        roboflow_version: Roboflow version to download
    """
    if not (os.path.exists('./datasets/Dataset-ViPARE-' + str(roboflow_version))):
        print('Downloading Roboflow dataset version ' + str(roboflow_version) + '...')
        du.dl_roboflow_dataset(roboflow_version)
    else:
        print('Roboflow dataset version ' + str(roboflow_version) + ' already exists')

def dl_TACO_dataset():
    """
    Description:
        Download the TACO dataset
    Usage:
        dl_TACO_dataset()
    Arguments:
        None
    """
    if not (os.path.exists('./datasets/TACO')):
        print('Downloading TACO dataset')
        du.dl_taco_dataset()
    else:
        print('TACO dataset already exists')

def convert_TACO_dataset():
    """
    Description:
        Convert the TACO dataset to YOLO format
    Usage:
        convert_TACO_dataset()
    Arguments:
        None
    """
    if not (os.path.exists('./datasets/TACO/data/yolo')):
        print('Converting TACO dataset to YOLO format')
        du.cocoToYolo('./datasets/TACO/data')
        du.split_dataset('./datasets/TACO/data/yolo', 0.7, 0.2, 0.1)
        du.tacoClassesToNaia('./datasets/TACO/data/yolo/')
    else:
        print('TACO dataset already in YOLO format')

def merge_datasets(roboflow_version, outputPath, tacoTrainOnly):
    """
    Description:
        Merge the downloaded datasets
    Usage:
        merge_datasets(roboflow_version, outputPath, tacoTrainOnly)
    Arguments:
        roboflow_version: Roboflow version to merge
        outputPath: Output path for merged dataset
        tacoTrainOnly: Use the TACO dataset only in training directory
    """
    if not (os.path.exists(outputPath)):
        print('Merging datasets')

        path = './datasets/Dataset-ViPARE-' + str(roboflow_version)
        # safe rename for VIPARE dataset and complete merge
        if os.path.exists(path + "/valid"):
            os.rename(path + "/valid", path + "/val")
            with open('./datasets/data.yaml', 'r') as f:
                lines = f.readlines()
            with open('./datasets/data.yaml', 'w') as f:
                for line in lines:
                    if "valid" in line :
                        f.write('val: ./Dataset-ViPARE-' + str(roboflow_version) + '/val/images')
                    else :
                        f.write(line)
            

        shutil.copy("./datasets/data.yaml", path + "/data.yaml")
        if tacoTrainOnly:
            du.mergeTacoDatasetAsTrain('./datasets/Dataset-ViPARE-' + str(roboflow_version), './datasets/TACO/data/yolo', outputPath)
        else :
            du.mergeDatasets('./datasets/Dataset-ViPARE-' + str(roboflow_version), './datasets/TACO/data/yolo', outputPath)
    else:
        print('Merged dataset already exists, not modifying it')

def delete_datasets_after_merge():
    """
    Description:
        Delete the base datasets used for merge after merge
    Usage:
        delete_datasets_after_merge()
    Arguments:
        None
    """
    print('Deleting datasets')
    du.delete_roboflow_dataset()
    du.delete_TACO_dataset()

if __name__ == '__main__':
    """
    Description:
        Main interface to download and merge datasets
    Usage:
        python getAndMergeDatasets.py --version 4 --delete True --fresh True --output ./datasets/mergeDataset --tacoTrainOnly False --roboflowDLOnly False

    Arguments:
        --version: Roboflow version to download
        --delete: Delete base datasets used for merge after merge
        --fresh: Delete all datasets before downloading, to ensure a fresh download
        --output: Output path for merged dataset
        --tacoTrainOnly: Use the TACO dataset only in training directory
        --roboflowDLOnly: Just download the roboflow dataset, do not merge it with TACO dataset

    """

    parser = argparse.ArgumentParser(description='Download and merge datasets')
    parser.add_argument('--version', type=int, default=4, help='Roboflow version to download')
    parser.add_argument('--delete', type=bool, default=False, help='Delete base datasets used for merge after merge')
    parser.add_argument('--fresh', type=bool, default=False, help='Delete all datasets before downloading, to ensure a fresh download')
    parser.add_argument('--output', type=str, default='./datasets/mergeDataset', help='Output path for merged dataset')
    parser.add_argument('--tacoTrainOnly', type=bool, default=False, help='Use the TACO dataset only in training directory')
    parser.add_argument('--roboflowDLOnly', type=bool, default=False, help='Just download the roboflow dataset, do not merge it with TACO dataset')

    args = parser.parse_args()
    roboflow_version = args.version
    delete_datasets_after_merge = args.delete
    get_all_fresh_datasets = args.fresh
    outputPath = args.output
    tacoTrainOnly = args.tacoTrainOnly
    dlonly = args.roboflowDLOnly

    if get_all_fresh_datasets:
        get_all_fresh_dataset(outputPath, roboflow_version)
    
    if not (os.path.exists('./datasets')):
        os.mkdir('./datasets')
    else:
        print('Datasets folder already exists')

    dl_roboflow_dataset(roboflow_version)

    if not dlonly:

        dl_TACO_dataset()

        convert_TACO_dataset()

        merge_datasets(roboflow_version, outputPath, tacoTrainOnly)

        if delete_datasets_after_merge:
            delete_datasets_after_merge()