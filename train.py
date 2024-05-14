import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
import os

# Function to load data from JSON files
def load_data(folder_path):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name), 'r') as f:
            data_list = json.load(f)
            for item in data_list:
                data.append(item['text'])
                labels.append(item['label'])
    return data, labels

# Function to tokenize text data
def tokenize_data(data):
    nlp = spacy.load('en_core_web_sm')
    tokenized_data = []
    for text in data:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        tokenized_data.append(tokens)
    return tokenized_data

# Function to convert tokenized data to sequences of integers
def convert_to_sequences(tokenized_data, vocab):
    sequences = []
    for text in tokenized_data:
        sequence = [vocab[word] for word in text]
        sequences.append(sequence)
    return sequences

# Function to pad sequences
def pad_sequences(sequences, max_length):
    padded = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded.append(sequence + [0] * (max_length - len(sequence)))
        else:
            padded.append(sequence[:max_length])
    return padded

# Define the model
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(max_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


# Load and tokenize text data from JSON files
folder_path = 'training_data'
data, labels = load_data(folder_path)
tokenized_data = tokenize_data(data)

# Create a vocabulary
vocab = {'<PAD>': 0, '<UNK>': 1}
for text in tokenized_data:
    for word in text:
        if word not in vocab:
            vocab[word] = len(vocab)

# Convert tokenized data to sequences of integers
sequences = convert_to_sequences(tokenized_data, vocab)

# Pad sequences
max_length = max([len(sequence) for sequence in sequences])
padded_sequences = pad_sequences(sequences, max_length)

# Instantiate the model
input_dim = len(vocab)
hidden_dim = 6
output_dim = 2
net = Net(input_dim, hidden_dim, output_dim)

# Convert data and labels to tensors
X = torch.tensor(padded_sequences)
y = torch.tensor(labels)

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Set number of epochs
num_epochs = 10

X = X.float()
# Train the model
for epoch in range(num_epochs):
    print(f"X shape before reshaping: {X.shape}")
    X = X.squeeze()
    print(f"X shape after reshaping: {X.shape}")

    # Forward pass
    outputs = net(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(net, 'model/model.pt')

# Save the vocabulary
with open('model/vocab.json', 'w') as f:
    json.dump(vocab, f)
