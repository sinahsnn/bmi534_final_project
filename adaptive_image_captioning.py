import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from PIL import Image
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Vocabulary Class ---
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.idx2word = {0: "<PAD>", 1: "<BEG>", 2: "<END>", 3: "<UNK>"}
        self.word2idx = {"<PAD>": 0, "<BEG>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.idx2word)

    def build_vocabulary(self, sentences):
        freq = {}
        for sentence in sentences:
            for word in sentence.split():
                freq[word] = freq.get(word, 0) + 1
        idx = 4
        for word, count in freq.items():
            if count >= self.freq_threshold:
                self.idx2word[idx] = word
                self.word2idx[word] = idx
                idx += 1
        print(f"Vocabulary size: {idx - 4} words")

    def numericalize(self, text):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]

# --- Flickr8k Dataset ---
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(caption_file)
        self.images = df["image"].tolist()
        self.captions = df["caption"].tolist()
        
        # Preprocess captions
        self.captions = [self.preprocess(caption) for caption in self.captions]
        
        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold=5)
        self.vocab.build_vocabulary(self.captions)

    def preprocess(self, text):
        text = re.sub(r'[^A-Za-z\s]', '', text.lower())  # Simple cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        caption = self.captions[idx]
        numerical_caption = [self.vocab.word2idx["<BEG>"]] + \
                            self.vocab.numericalize(caption) + \
                            [self.vocab.word2idx["<END>"]]
        return img, torch.tensor(numerical_caption)

# --- Collate Function for Padding ---
class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)
        return imgs, captions

# --- Encoder with Adaptive Feature Aggregation ---
class AdaptiveEncoder(nn.Module):
    def __init__(self, embed_size):
        super(AdaptiveEncoder, self).__init__()
        # Load full InceptionV3
        self.inception = models.inception_v3(pretrained=True)
        self.inception.eval()  # Set to eval mode to disable dropout and aux classifier
        
        # Freeze the model
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # Variables to store intermediate features
        self.features = None
        
        # Register a hook to capture features from Mixed_7c (equivalent to mixed7)
        def hook_fn(module, input, output):
            self.features = output
        
        # Find and register hook on the Mixed_7c layer
        for name, module in self.inception.named_modules():
            if "Mixed_7c" in name:  # This is the 8x8x2048 feature map layer
                module.register_forward_hook(hook_fn)
                break
        
        # Learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Final projection
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048, embed_size)
    
    def compute_similarity(self, features):
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H*W).permute(0, 2, 1)  # [B, 64, 2048]
        
        # Cosine similarity
        features_i = features_flat.unsqueeze(2)  # [B, 64, 1, 2048]
        features_j = features_flat.unsqueeze(1)  # [B, 1, 64, 2048]
        cos_sim = nn.functional.cosine_similarity(features_i, features_j, dim=3)
        
        # Euclidean distance
        diff = features_i - features_j
        euc_dist = -torch.norm(diff, dim=3).pow(2)
        
        # Normalize to [-1, 1] per batch
        euc_min = euc_dist.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        euc_max = euc_dist.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        euc_dist = (euc_dist - euc_min) / (euc_max - euc_min + 1e-8) * 2 - 1
        
        # Hybrid similarity
        sim = self.alpha * cos_sim + (1 - self.alpha) * euc_dist
        
        return sim
    
    def forward(self, x):
        # Reset stored features
        self.features = None
        
        # Forward pass through InceptionV3 to trigger hook
        with torch.no_grad():  # We don't need gradients through the full model
            _ = self.inception(x)
        
        # Check if features were captured
        if self.features is None:
            raise RuntimeError("Features not captured by hook - check layer name")
        
        # Get features from hook
        features = self.features  # [B, 2048, 8, 8]
        B = features.size(0)
        
        # Compute similarity matrix
        S = self.compute_similarity(features)  # [B, 64, 64]
        
        # Compute weights by summing similarities
        weights = S.sum(dim=2)  # [B, 64]
        weights = nn.functional.softmax(weights, dim=1)
        
        # Reshape features for aggregation
        features_flat = features.view(B, 2048, -1)  # [B, 2048, 64]
        
        # Weighted aggregation
        F_agg = torch.bmm(features_flat, weights.unsqueeze(2)).squeeze(2)  # [B, 2048]
        
        # Final projection
        out = self.fc(self.dropout(self.relu(F_agg)))  # [B, embed_size]
        
        return out

# --- Decoder (LSTM-based) ---
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # Convert captions to embeddings
        embeddings = self.embedding(captions)  # [seq_len, B, embed_size]
        
        # Prepare image features and concatenate
        features = features.unsqueeze(0)  # [1, B, embed_size]
        inputs = torch.cat((features, embeddings), dim=0)  # [seq_len+1, B, embed_size]
        
        # LSTM forward pass
        hiddens, _ = self.lstm(inputs)
        
        # Project to vocabulary space
        outputs = self.fc(self.dropout(hiddens))  # [seq_len+1, B, vocab_size]
        
        return outputs

# --- Full Model ---
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = AdaptiveEncoder(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)  # (B, embed_size)
        outputs = self.decoder(features, captions)  # (seq_len+1, B, vocab_size)
        return outputs

    def generate_caption(self, image, vocab, max_length=20):
        self.eval()
        with torch.no_grad():
            # Extract features
            features = self.encoder(image.unsqueeze(0))  # [1, embed_size]
            
            # Initialize generation
            caption = [vocab.word2idx["<BEG>"]]
            states = None
            
            for _ in range(max_length):
                if len(caption) == 1:
                    # First timestep: use image features directly
                    input_tensor = features.unsqueeze(0)  # [1, 1, embed_size]
                else:
                    # Later timesteps: use previous word embedding
                    prev_word = torch.tensor([caption[-1]]).to(image.device)
                    input_tensor = self.decoder.embedding(prev_word).unsqueeze(0)
                
                # Pass through LSTM
                hiddens, states = self.decoder.lstm(input_tensor, states)
                
                # Get prediction
                outputs = self.decoder.fc(hiddens.squeeze(0))
                predicted = outputs.argmax(1).item()
                
                # Add to caption
                caption.append(predicted)
                
                # Break if END token or max length reached
                if predicted == vocab.word2idx["<END>"]:
                    break
            
            # Return caption words (excluding special tokens)
            words = [vocab.idx2word[idx] for idx in caption[1:-1]] # Remove BEG and END
            return words

# --- Training and Evaluation ---
def train_model(model, train_loader, val_loader, vocab, num_epochs, device):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, captions = imgs.to(device), captions.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(imgs, captions[:-1])  # Exclude <END> from input
            
            # Extract correct dimensions for loss calculation
            # outputs shape: [seq_len+1, B, vocab_size] - we need to remove first timestep
            outputs = outputs[1:, :, :]  # Remove the first timestep (image features)
            
            # Now outputs shape should exactly match captions[1:] shape
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        
        # Validation (generate a sample caption)
        model.eval()
        with torch.no_grad():
            for val_imgs, _ in val_loader:
                val_img = val_imgs[0].to(device)
                caption = model.generate_caption(val_img, vocab)
                print(f"Sample Caption: {' '.join(caption)}")
                break

# --- Main Execution ---
if __name__ == "__main__":
    # Paths (adjust these to your local setup)
    root_dir = "/scratch/lpanch2/archive/Images"  # e.g., "/data/Flickr8k/Images"
    caption_file = "/scratch/lpanch2/archive/captions.txt"  # e.g., "/data/Flickr8k/captions.txt"
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and DataLoader
    dataset = Flickr8kDataset(root_dir, caption_file, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    pad_idx = dataset.vocab.word2idx["<PAD>"]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=CollateFn(pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=CollateFn(pad_idx))
    
    # Model Parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)
    
    # Train
    train_model(model, train_loader, val_loader, dataset.vocab, num_epochs=10, device=device)
    
    # Save Model
    torch.save(model.state_dict(), "image_captioning_model.pth")