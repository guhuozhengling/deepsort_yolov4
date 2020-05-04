from .darknet2pytorch import Darknet

def build_detector(cfg, use_cuda):
    model=Darknet(cfg.YOLOV4.CFG)
    model.load_weights(cfg.YOLOV4.WEIGHT)
    if use_cuda:
        model.cuda().half()
    return model