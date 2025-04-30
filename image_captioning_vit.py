import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import numpy as np
import re
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

# For evaluation metrics
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.idx2word = {0: "<PAD>", 1: "<BEG>", 2: "<END>", 3: "<UNK>"}
        self.word2idx = {"<PAD>": 0, "<BEG>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.idx2word)
        
    def build_vocabulary(self, sentences):
        idx = 4
        frequency = {}
        
        for sentence in sentences:
            for word in sentence.split():
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
        
        for word, freq in frequency.items():
            if (freq >= self.freq_threshold):
                self.idx2word[idx] = word
                self.word2idx[word] = idx
                idx += 1
    
        print(f'The number of words in dictionary: {idx - 4}')
    
    def numericalize(self, sentence):
        tokenized_text = sentence.split()
        return [self.word2idx[word] if word in self.word2idx else self.word2idx["<UNK>"] for word in tokenized_text]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_path, freq_threshold=10, transform=None, preprocess=None):
        self.freq_threshold = freq_threshold
        self.transform = transform
        self.root_dir = root_dir

        df = pd.read_csv(caption_path)
        self.data_size = len(df)
        
        self.captions = df['caption'].tolist()
        self.images = df['image'].tolist()

        if preprocess:
            self.captions = preprocess(self.captions)
        
        self.vocab = Vocabulary(freq_threshold)
        
        print(f'The number of captions: {len(self.captions)}')
        self.vocab.build_vocabulary(self.captions)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        caption = self.captions[index]
        image = self.images[index]
        
        img = Image.open(os.path.join(self.root_dir, image)).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.word2idx["<BEG>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx["<END>"])
        
        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_value):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        img = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_value)
        
        return img, targets

# Vision Transformer components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # (batch_size, in_channels, img_size, img_size) -> (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = self.proj(x)
        # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5) -> (batch_size, embed_dim, n_patches)
        x = x.flatten(2)
        # (batch_size, embed_dim, n_patches) -> (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        batch_size, n_tokens, dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.n_heads, dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, 
                 n_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1,
                 final_dim=300):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final projection to the desired embedding size
        self.final_proj = nn.Linear(embed_dim, final_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding and other layers
        self.apply(self._init_weights_layers)
    
    def _init_weights_layers(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Create patches
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Use only the class token for classification/captioning
        x = x[:, 0]
        
        # Project to the desired embedding size
        x = self.final_proj(x)
        return self.dropout(self.relu(x))

class decoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, embedding_matrix, hidden_size, num_layers):
        super(decoderRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, caption):
        embeddings = self.dropout(self.embedding(caption))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings.to(torch.float))
        outputs = self.linear(hiddens)
        return outputs

class ViT2RNN(nn.Module):
    def __init__(self, embed_size, vocab_size, embedding_matrix, hidden_size, num_layers):
        super(ViT2RNN, self).__init__()
        self.encoderViT = ViTEncoder(
            img_size=224,      # Standard ViT input size
            patch_size=16,     # Standard patch size for ViT
            in_channels=3,     # RGB images
            embed_dim=768,     # Standard ViT embedding dimension
            depth=6,           # Reduced number of transformer blocks for efficiency
            n_heads=12,        # Number of attention heads
            final_dim=embed_size  # Project to the size expected by RNN
        )
        self.decoderRNN = decoderRNN(embed_size, vocab_size, embedding_matrix, hidden_size, num_layers)
    
    def forward(self, images, caption):
        x = self.encoderViT(images.to(torch.float))
        x = self.decoderRNN(x.to(torch.float), caption)
        return x
    
    def captionImage(self, image, vocabulary, maxlength=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderViT(image.unsqueeze(0)).unsqueeze(0)
            states = None
            
            for _ in range(maxlength):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoderRNN.embedding(predicted.unsqueeze(0))
                
                if vocabulary.idx2word[predicted.item()] == "<END>":
                    break
        return result_caption, [vocabulary.idx2word[i] for i in result_caption]

def preprocess(sentences):
    new = []
    for sentence in sentences:
        sentence = re.sub(r'\w*\d\w*', '', sentence)
        sentence = re.sub('[^A-Za-z0-9\s]', '', sentence)
        sentence = re.sub('(\s[a-zA-Z]\s)|(^[a-zA-Z]\s)|(\s[a-zA-Z]$)', ' ', sentence)
        sentence = re.sub('\s+', ' ', sentence)
        sentence = re.sub('(^\s+)|(\s+$)', '', sentence)
        new.append(sentence.lower())
    return new

def create_random_embeddings(word2idx, embedding_dim=300):
    """
    Create random embeddings for vocabulary when GloVe is not available
    
    Args:
        word2idx: Dictionary mapping words to indices
        embedding_dim: Dimension of embeddings
    
    Returns:
        Random embedding matrix for all words in word2idx
    """
    print("Creating random word embeddings...")
    num_tokens = len(word2idx)
    embedding_matrix = np.random.normal(0, 0.1, (num_tokens, embedding_dim))
    
    # Zero vector for padding
    embedding_matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim)
    
    return embedding_matrix

def evaluate_metrics(model, dataloader, vocabulary):
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for imgs, captions in tqdm(dataloader, desc="Evaluating"):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
            
            for i in range(imgs.size(0)):
                img = imgs[i]
                _, predicted_caption = model.captionImage(img, vocabulary)
                
                # Remove special tokens
                predicted_caption = [word for word in predicted_caption 
                                    if word not in ["<PAD>", "<BEG>", "<END>", "<UNK>"]]
                
                # Get reference caption
                reference_caption = []
                for j in range(captions.shape[0]):
                    word_idx = captions[j][i].item()
                    if word_idx not in [vocabulary.word2idx["<PAD>"], 
                                      vocabulary.word2idx["<BEG>"], 
                                      vocabulary.word2idx["<END>"], 
                                      vocabulary.word2idx["<UNK>"]]:
                        reference_caption.append(vocabulary.idx2word[word_idx])
                
                if predicted_caption and reference_caption:
                    references.append([reference_caption])
                    hypotheses.append(predicted_caption)
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate METEOR scores
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        if hyp:  # Skip empty hypotheses
            meteor_scores.append(meteor_score(ref, hyp))
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    
    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = []
    for ref, hyp in zip(references, hypotheses):
        if hyp and ref[0]:  # Skip empty hypotheses or references
            try:
                scores = rouge.get_scores(' '.join(hyp), ' '.join(ref[0]))
                rouge_scores.append(scores[0]['rouge-l']['f'])
            except Exception:
                continue
    rouge_avg = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    
    return {
        'BLEU-1': bleu1,
        'BLEU-4': bleu4,
        'METEOR': meteor_avg,
        'ROUGE-L': rouge_avg
    }

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, vocab, patience=5, delta=0.001, checkpoint_path='best_model_vit.pth', start_epoch=0):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    counter = 0  # Counter for patience
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch+epochs} (Training)')
        
        for imgs, captions in train_pbar:
            imgs, captions = imgs.to(device), captions.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, captions[:-1])
            
            # Calculate loss
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), 
                captions.reshape(-1)
            )
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{start_epoch+epochs} (Validation)')
        
        with torch.no_grad():
            for imgs, captions in val_pbar:
                imgs, captions = imgs.to(device), captions.to(device)
                
                outputs = model(imgs, captions[:-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]), 
                    captions.reshape(-1)
                )
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{start_epoch+epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save model and check for early stopping
        if avg_val_loss < best_val_loss - delta:
            # If validation loss improved by more than delta
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f'Model saved (Val Loss: {best_val_loss:.4f})')
            counter = 0  # Reset counter
        else:
            counter += 1  # Increment counter
            print(f'Early stopping counter: {counter}/{patience}')
            
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}')
                break
    
    return train_losses, val_losses

def main(resume_training=False):
    # Configuration
    root_dir = '/scratch/lpanch2/machine_learning/Images'
    caption_path = '/scratch/lpanch2/machine_learning/captions.txt'
    batch_size = 32  # Reduced batch size for ViT which is more memory-intensive
    embed_size = 300  # Embedding size
    hidden_size = 512
    num_layers = 1
    learning_rate = 3e-4  # Slightly lower learning rate for ViT
    num_epochs = 100
    patience = 6  # For early stopping
    delta = 0.001  # Minimum change in validation loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = '/scratch/lpanch2/machine_learning/vit_model/best_model_vit.pth'
    
    # Data preprocessing - using 224x224 images which is standard for ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Loading dataset...")
    dataset = FlickrDataset(
        root_dir=root_dir,
        caption_path=caption_path,
        transform=transform,
        preprocess=preprocess
    )
    
    # Split dataset into train, validation, and test
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    pad_value = dataset.vocab.word2idx["<PAD>"]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=MyCollate(pad_value)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=MyCollate(pad_value)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=MyCollate(pad_value)
    )
    
    # Use random embeddings instead of downloading GloVe
    print("Creating word embeddings...")
    embedding_dim = 300
    word2idx = dataset.vocab.word2idx
    embedding_matrix = create_random_embeddings(word2idx, embedding_dim)
    
    # Initialize model
    vocab_size = len(dataset.vocab)
    model = ViT2RNN(
        embed_size=embed_size,
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)
    
    # Use AdamW optimizer which works better with transformers
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Check if checkpoint exists and resume training if requested
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    if os.path.exists(checkpoint_path):
        if resume_training:
            print(f"Loading checkpoint from {checkpoint_path} to resume training")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint found at {checkpoint_path}, but not using it to resume training")
    
    # Train model
    print("Starting training...")
    new_train_losses, new_val_losses = train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs, 
        device,
        dataset.vocab,
        patience,
        delta,
        checkpoint_path,
        start_epoch
    )
    
    # Combine losses
    train_losses.extend(new_train_losses)
    val_losses.extend(new_val_losses)
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = evaluate_metrics(model, test_loader, dataset.vocab)
    
    # Display metrics
    print("\nTest Set Performance:")
    print(f"BLEU-1: {metrics['BLEU-1']:.4f}")
    print(f"BLEU-4: {metrics['BLEU-4']:.4f}")
    print(f"METEOR: {metrics['METEOR']:.4f}")
    print(f"ROUGE-L: {metrics['ROUGE-L']:.4f}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot_vit.png')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vision Transformer Image Captioning Model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    main(resume_training=args.resume)