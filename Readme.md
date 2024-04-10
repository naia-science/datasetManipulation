## Dataset Manipulation

Enables to get, merge and format the following datasets:
 - [x] TACO dataset
 - [x] Naia dataset on Roboflow
 - [ ] surfnet dataset

 ### Usage

 ```bash
 pip install -r requirements.txt
 python getAndMergeDatasets.py --version int --delete bool --fresh bool --output str --tacoTrainOnly bool --roboflowDLOnly bool
 ```

 You can check the merge results and display images using the example jupyter notebook Notebooks/exampleAndCheck.ipynb