"""
pip install torch timm composer

torch - pytorch
timm - pytorch image models, very standard
composer - open source library from MosaicML for algorithmic speed ups
zipp -
"""
import composer.functional as cf
import timm
import torch

path = (
    "./resnet50-inat21-pretrained.pt"
)

def main():
    model = timm.create_model("resnet50", num_classes=10000)
    # Channels Last: https://docs.mosaicml.com/en/latest/method_cards/channels_last.html
    # Source: https://docs.mosaicml.com/en/latest/_modules/composer/algorithms/channels_last/channels_last.html
    model.to(memory_format=torch.channels_last)
    # BlurPool: https://docs.mosaicml.com/en/latest/method_cards/blurpool.html
    # API: https://docs.mosaicml.com/en/latest/api_reference/generated/composer.functional.apply_blurpool.html
    cf.apply_blurpool(model)


    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    # Actual model keys
    model_dict = state_dict["state"]["model"]
    # Trained with distributed data parallel, so the keys are prefixed with 'module.'
    # This line gets rid of the prefixes
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        model_dict, "module."
    )
    model.load_state_dict(model_dict)
    
    print("Hi")
    # Randomly initialized linear layer
    #model.fc = torch.nn.Linear(2048, 2)


if __name__ == "__main__":
    main()