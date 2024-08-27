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
from transformers import GPT2Tokenizer

def run_clip_classification(clip, img_paths, text_list):
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set a padding token (use eos_token or define a new one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Tokenize text using the GPT-2 tokenizer
    text_tokens = tokenizer(text_list, padding='max_length', max_length=82, return_tensors='pt')['input_ids']
    
    # Replace padding tokens with space tokens
    space_token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    padding_token_id = tokenizer.pad_token_id
    
    # Replace padding tokens with space tokens in the tokenized sequences
    text_tokens[text_tokens == padding_token_id] = space_token_id
    
    # Load images from paths and resize them to 128x128
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.6589, 0.6147, 0.6220), (0.2234, 0.2234, 0.2158))
    ])
    
    images = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)
    
    # Stack images into a batch
    images = torch.stack(images).to(device)
    text_tokens = text_tokens.to(device)
    
    # Pass images and tokenized text through the CLIP model
    img_probs, text_probs = clip(images, text_tokens)
    
    return img_probs, text_probs

if __name__ == '__main__':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    config = CLIPConfig(vocab_len=50257, seq_len=82, latent_dim=256, use_mask=False)
    model = CLIP(config)
    model.to(device)
    
    model_path = 'weights/cnn-clip-mask_false.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    image_file = ['/Users/ayanfe/Documents/Datasets/Waifus/Val/75363_result.jpg',
                  '/Users/ayanfe/Documents/Datasets/portraits/107-fs7iezq.jpg',
                  '/Users/ayanfe/Documents/Code/Scraping Shit/images/2335661.jpg']

    tags = ['1girl, hatsune-miku, blue hair, vocaloid',
            'girl, blonde, portrait',
            'boy, white hair']

    img_probs, text_probs = run_clip_classification(model, image_file, tags)

    # Print CLIP classification results
    print("Image Probabilities:", img_probs)
    print("Text Probabilities:", text_probs)
