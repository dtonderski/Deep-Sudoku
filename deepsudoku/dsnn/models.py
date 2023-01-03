import torch
from typing import Literal
from deepsudoku.dsnn import se_resnet, transformer, sudoker
import importlib.resources

models = {
    "SeResNet": lambda: se_resnet.SeResNet(10, 128, 32),
    "ViTTiTransformer": lambda: transformer.Transformer(12, 192, 3, 768, 0),
    "ViTTiSudoker": lambda: sudoker.Sudoker(12, 192, 3, 768, 0)
}


def load_model(model_name: Literal["SeResNet", "ViTTiTransformer", "ViTTiSudoker"],
               device: Literal["cuda", "cpu"] = "cuda",
               use_builtin_checkpoint=True):
    model = models[model_name]().to(device)
    if use_builtin_checkpoint:
        if model_name != 'ViTTiSudoker':
            print("Can only use builtin checkpoint if model_name is ViTTiSudoker!")
            return model
        with importlib.resources.path("deepsudoku.resources", "ViTTiSudoker.pth") as checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == '__main__':
    load_model("ViTTiSudoker")
