import argparse
from ultralytics import YOLO
from torch import nn
import torch
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='yolov8n.yaml', help='initial weights path')
    parser.add_argument('--cfg',   type=str, default='ultralytics/yolo/configs/default.yaml', help='model.yaml path')

    return parser.parse_args()


class TransposeModule(nn.Module):

    def forward(self, res):
        print("TransposeModule fwd")
        if isinstance(res, tuple):
            y = res[0].permute(0, 2, 1).contiguous()
            return y, *res[1:]
        else:
            return res.permute(0, 2, 1).contiguous()
        
        
def main():
    args = parse_args()

    print(args)

    yolo = YOLO(args.model)

    args.cfg = check_yaml(args.cfg)  # check YAML
    args.cfg = yaml_load(args.cfg, append_filename=False)  # model dict

    model_core: nn.Sequential = yolo.model.model
    append_module = TransposeModule()
    append_module.i = append_module.f = -1
    model_core.add_module("transpose", append_module)

    yolo.info(verbose=True)

    yolo.export(**args.cfg)

if __name__ == '__main__':
    main()