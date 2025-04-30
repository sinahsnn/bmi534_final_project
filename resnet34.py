import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from nltk.translate.bleu_score import corpus_bleu

# Set up your base directory and working directory
Base_dir = '/scratch/lpanch2/machine_learning'  # Update with your data path
working_dir = '/scratch/lpanch2/machine_learning/resnet18_model'  # Update with your working directory

# 1. Image Feature Extraction using ResNet50
print("Loading ResNet50 model for feature extraction...")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)
print(model.summary())

# Function to extract features from images
def extract_features(directory):
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
            # Load and preprocess the image
            img_path = os.path.join(directory, img_name)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            image = preprocess_input(image)
            
            # Extract features
            feature = model.predict(image, verbose=0)
            
            # Get image ID and store features
            image_id = img_name.split('.')[0]
            features[image_id] = feature
    return features

# Extract features from all images
print("Extracting features from images...")
directory = os.path.join(Base_dir, 'Images')
features = extract_features(directory)

# 2. Text Preprocessing
print("Loading captions and creating mapping...")
with open(os.path.join(Base_dir, 'captions.txt'), 'r') as File:
    next(File)
    captions_file = File.read()

# Create mapping from image IDs to captions
mapping = {}
for line in tqdm(captions_file.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Preprocess the captions
def preprocess_caption(caption):
    # Convert to lowercase
    caption = caption.lower()
    # Remove special characters and numbers
    caption = caption.replace('[^A-Za-z]', '')
    # Remove extra spaces
    caption = caption.replace('\s+', ' ')
    # Add start and end tokens
    caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
    return caption

# Apply preprocessing to all captions
for key, captions in mapping.items():
    for i in range(len(captions)):
        mapping[key][i] = preprocess_caption(captions[i])

# Collect all captions
all_captions = [caption for key in mapping for caption in mapping[key]]

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

print(f"Vocabulary Size: {vocab_size}")
print(f"Maximum Sequence Length: {max_length}")

# Split data into train and test sets
image_ids = list(mapping.keys())
train, test = train_test_split(image_ids, test_size=0.1, random_state=42)

# Data generator to avoid memory issues
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in data_keys:
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            n += 1
            if n == batch_size:
                yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
                X1.clear()
                X2.clear()
                y.clear()
                n = 0

# 3. Model Architecture
# Image feature layers - ResNet50 features
inputs1 = Input(shape=(2048,), name="image")  # ResNet50 features shape
dropout1 = Dropout(0.4)(inputs1)  # Higher dropout for regularization
image_features = Dense(256, activation='relu')(dropout1)

# Sequence feature layers
inputs2 = Input(shape=(max_length,), name="text")
embedding = Embedding(vocab_size, 256, mask_zero=True)(inputs2)  # Added mask_zero
dropout2 = Dropout(0.4)(embedding)
sequence_features = LSTM(256, return_sequences=False)(dropout2)

# Decoder
decoder1 = add([image_features, sequence_features])
decoder2 = Dense(512, activation='relu')(decoder1)  # Larger dense layer
decoder3 = Dropout(0.4)(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder3)

# Create and compile model
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# 4. Training
print("Training the model...")
epochs = 20  # Increased epochs
batch_size = 32
steps = len(train) // batch_size

# Learning rate scheduling for better performance
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1
)

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=5, restore_best_weights=True, verbose=1
)

# Model checkpoint to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(working_dir, 'best_model_resnet50.h5'),
    monitor='loss',
    save_best_only=True,
    verbose=1
)

callbacks = [reduce_lr, early_stopping, checkpoint]

# Train model with data generator
for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    history = model.fit(
        generator, 
        epochs=1, 
        steps_per_epoch=steps, 
        verbose=1,
        callbacks=callbacks
    )

# 5. Functions for Caption Generation and Evaluation
def convert_to_word(number, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == number:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([image, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = convert_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# 6. Evaluate model with BLEU score
print("Evaluating model with BLEU scores...")
actual, predicted = [], []

for key in tqdm(test):
    captions = mapping[key]
    # Predict caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    # Split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # Append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)

# Calculate BLEU scores
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print("BLEU-3: %f" % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
print("BLEU-4: %f" % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# 7. Save the model and tokenizer
model.save(os.path.join(working_dir, 'final_model_resnet50.h5'))
with open(os.path.join(working_dir, 'tokenizer_resnet50.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)