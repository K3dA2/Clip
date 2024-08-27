import tqdm
import torch
import datetime
import os
from model import CLIP, CLIPConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
from torchvision import transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ImageTagDataset(Dataset):
    def __init__(self, json_file, image_folder, file_extension='.jpg', transform=None):
        # Load the JSON file with image names and token lists
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_folder = image_folder
        self.file_extension = file_extension
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the image name and corresponding tokens from the JSON data
        image_name = list(self.data.keys())[idx]
        tokens = self.data[image_name]  # Tokens are already encoded and stored in the JSON file
        
        # Construct the image file path
        image_path = os.path.join(self.image_folder, f"{image_name}{self.file_extension}")
        
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation to the image (if any)
        if self.transform:
            image = self.transform(image)

        # Convert tokens to a PyTorch tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return image, tokens_tensor  # Return image and the token tensor

def validation_loop(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    loss_val = 0.0

    with torch.no_grad():  # Disable gradient computation
        for imgs, tkns in val_loader:
            imgs = imgs.to(device)
            tkns = tkns.to(device)

            loss = model.calculate_loss(imgs, tkns)

            loss_val += loss.item()

    return loss_val / len(val_loader)

def training_loop(n_epochs, optimizer, model, device, data_loader, val_loader, epoch_start=0):
    best_val_score = 10.0
    for epoch in range(epoch_start, n_epochs + epoch_start):
        model.train()
        loss_train = 0.0
        val_loss = None
        

        progress_bar = tqdm.tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        optimizer.zero_grad()  # Initialize the gradient

        for batch_idx, (imgs, tkns) in enumerate(progress_bar):
            imgs = imgs.to(device)
            tkns = tkns.to(device)
            
            loss = model.calculate_loss(imgs, tkns)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        with open("clip-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))

        # Validate every 5 epochs
        if epoch % 1 == 0:
            val_loss = validation_loop(model, val_loader, device)
            print(f"Epoch {epoch}, Validation loss: {val_loss}")
            with open("clip-val-loss.txt", "a") as file:
                file.write(f"{val_loss}\n")
        
        if best_val_score > val_loss:
            # Save model checkpoint every epoch
            model_filename = f'cnn-clip-mask_false.pth'
            model_path = os.path.join('weights/', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            best_val_score = val_loss

if __name__ == '__main__':

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    
    config = CLIPConfig(vocab_len=50257, seq_len=82, 
                        latent_dim=256,use_mask=False)
    model = CLIP(config)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    print("param count: ", count_parameters(model))
    
    epoch = 0

    json_file = '/Users/ayanfe/Documents/Code/Scraping Shit/train.json'
    image_folder = '/Users/ayanfe/Documents/Code/Scraping Shit/images'
    valid_file = '/Users/ayanfe/Documents/Code/Scraping Shit/val.json'
    model_path = 'weights/cnn-clip-mask_false.pth'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.6589, 0.6147, 0.6220), (0.2234, 0.2234, 0.2158))
    ])

    # Initialize the dataset and dataloader
    dataset = ImageTagDataset(json_file=json_file, image_folder=image_folder, file_extension='.jpg', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    val_dataset = ImageTagDataset(json_file=valid_file, image_folder=image_folder, file_extension='.jpg', transform=transform)
    valid_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    '''
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    '''

    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        device=device,
        data_loader=dataloader,
        val_loader=valid_dataloader,
        epoch_start=epoch + 1,
    )
