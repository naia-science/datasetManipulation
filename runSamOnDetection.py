import ultralytics
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt


from argparse import Namespace

class DetectionDataset:
    def __init__(self, data_yaml, output_dir, mode='valid'):
        self.data_yaml = data_yaml
        self.data = check_det_dataset(data_yaml)
        self.batch_size = 1
        self.args = Namespace(
            task='detect',
            mode=mode,
            imgsz=640,
            rect=False,
            cache=False,
            single_cls=False,
            classes=None,
            mask_ratio=4,
            overlap_mask=True,
          )
        imgs_path = None
        try:
            if mode in self.data:
                imgs_path = self.data[mode]
            elif (mode=="valid" and "val" in self.data):
                imgs_path = self.data["val"]
            else:
                raise Exception("Error, img path not found")
        except Exception(e):
            print(e)
        
        dataset = build_yolo_dataset(
            self.args, 
            imgs_path,
            self.batch_size, self.data, mode=mode, stride=32)
        self.dataloader = build_dataloader(dataset, self.batch_size, 8, False, -1)
        
        self.model = None

    def run(self):
        # Create a FastSAM model
        if self.model == None:
            self.model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

        for batch in self.dataloader:
            img = batch['img']
            out = self.model(img)
            masks = out[0].masks.xy
            print(len(masks))
            print(len(masks[0]))
            break

