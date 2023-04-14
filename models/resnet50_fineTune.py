"""
pip install torch timm composer

torch - pytorch
timm - pytorch image models, very standard
composer - open source library from MosaicML for algorithmic speed ups
zipp -
"""
from PIL import Image, ImageOps
import time
import os
import copy
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
import composer.functional as cf
import timm
import torch
import torch.nn as nn
import torch.optim as optim

path = (
    "./resnet50-inat21-pretrained.pt"
)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def loop_for_resize(folder_path):
    for f in os.scandir(folder_path):
        if os.path.isdir(f.path):
            loop_for_resize(f.path)
        else:
            resize_images(f.path)

def resize_images(image_path):
    #print(image_path)
    #image_path="../datasets/mimic_pair_3/erato_notabilis/10429012_D_lowres.png"
    new_size=224

    img = Image.open(image_path)

    # Compute the necessary padding
    w, h = img.size
    ratio = float(new_size) / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (new_size - new_w) // 2
    right = new_size - new_w - left
    top = (new_size - new_h) // 2
    bottom = new_size - new_h - top

    img_tensor = TF.to_tensor(img)

    # Pad the image with white pixels
    resized_img = TF.resize(img_tensor, (new_h, new_w))
    resized_img=TF.to_pil_image(resized_img)
    padded_img = ImageOps.expand(resized_img, (left, top, right, bottom), fill="white")

    # save the output image
    padded_img.save(image_path)

def main():
    model = timm.create_model("resnet50", num_classes=10000)
    model.to(memory_format=torch.channels_last)
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
    
    print("Loading model...")

    #Input

    print("Initializing Datasets and Dataloaders...")
    num_output_class=2
    data_dir="../datasets/mimic_pair_3"
    batch_size = 8
    num_epochs = 15

    loop_for_resize(data_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #need work
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #need work
        ]),
    }

    

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    #Freeze the model parameters
    for param in model.parameters():
            param.requires_grad = False

    # Randomly initialized linear layer
    model.fc = nn.Linear(2048, num_output_class)

    # Send the model to GPU
    model = model.to(device)

    print("Creating Optimizer...")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    print("Creating Loss Function...")
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    print("Starting Training & Evaluation...")
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    torch.save(model_ft, "./pretrained_on_mimic_pair_3.pt")


if __name__ == "__main__":
    main()