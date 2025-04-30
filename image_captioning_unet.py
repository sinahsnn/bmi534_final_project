import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import urllib.request
import zipfile
import io

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

# U-Net building blocks
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# U-Net Encoder replaces InceptionV3
class UNetEncoder(nn.Module):
    def __init__(self, embed_size, should_train=True):
        super(UNetEncoder, self).__init__()
        self.should_train = should_train
        
        # Downsampling path
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down5 = DoubleConv(512, 1024)
        
        # Global average pooling and embedding projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Encoder path
        x1 = self.down1(x)
        x = self.pool1(x1)
        
        x2 = self.down2(x)
        x = self.pool2(x2)
        
        x3 = self.down3(x)
        x = self.pool3(x3)
        
        x4 = self.down4(x)
        x = self.pool4(x4)
        
        x5 = self.down5(x)
        
        # Get features for decoder
        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        
        if not self.should_train:
            for param in self.parameters():
                param.requires_grad = False
        
        return self.dropout(self.relu(features))

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

class UNet2RNN(nn.Module):
    def __init__(self, embed_size, vocab_size, embedding_matrix, hidden_size, num_layers):
        super(UNet2RNN, self).__init__()
        self.encoderUNet = UNetEncoder(embed_size)
        self.decoderRNN = decoderRNN(embed_size, vocab_size, embedding_matrix, hidden_size, num_layers)
    
    def forward(self, images, caption):
        x = self.encoderUNet(images.to(torch.float))
        x = self.decoderRNN(x.to(torch.float), caption)
        return x
    
    def captionImage(self, image, vocabulary, maxlength=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderUNet(image.unsqueeze(0)).unsqueeze(0)
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

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, vocab, patience=5, delta=0.001, checkpoint_path='best_model.pth', start_epoch=0):
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
    batch_size = 64
    embed_size = 300  # GloVe embedding size
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-3
    num_epochs = 100
    patience = 5  # For early stopping
    delta = 0.001  # Minimum change in validation loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = '/scratch/lpanch2/machine_learning/unet_model/best_model.pth'
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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
    model = UNet2RNN(
        embed_size=embed_size,
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
    plt.savefig('loss_plot.png')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Captioning Model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    main(resume_training=args.resume)