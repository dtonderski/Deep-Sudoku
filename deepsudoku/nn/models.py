import torch
from typing import Literal
from deepsudoku.nn import se_resnet, transformer, sudoker

models = {
    "SeResNet": lambda: se_resnet.SeResNet(10, 128, 32),
    "ViTTiTransformer": lambda: transformer.Transformer(12, 192, 3, 768, 0),
    "ViTTiSudoker": lambda: sudoker.Sudoker(12, 192, 3, 768, 0)
}


def load_model(model_name: Literal["SeResNet", "ViTTiTransformer", "ViTTiSudoker"],
               device: Literal["cuda", "cpu"] = "cuda",
               checkpoint_path: str = None):
    network = models[model_name]().to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model_state_dict'])
    return network
