import json
import torch
import torch.nn as nn
import spacy
from train import padded_sequences

# Define the model
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model and vocabulary
net = torch.load('model/model.pt')
with open('model/vocab.json', 'r') as f:
    vocab = json.load(f)

# Add '<UNK>' token to vocabulary if it doesn't already exist
if '<UNK>' not in vocab:
    vocab['<UNK>'] = len(vocab)

def interact_with_model(model, vocab):
    while True:
        user_input = input("Input your text: ")
        if user_input.lower() == "?quit":
            break
        elif user_input.lower() == "?help":
            print("Available commands:")
            print("?help - Show this help message")
            print("?quit - Quit the program")

        # Lowercase and trim user input
        user_input = user_input.lower().strip()

        # Tokenize user input
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(user_input)
        tokens = [token.text for token in doc]

        # Convert tokens to sequence of integers
        sequence = []
        for word in tokens:
            if word in vocab:
                sequence.append(vocab[word])
            else:
                sequence.append(vocab['<UNK>'])

        # Pad sequence
        max_length = max([len(sequence) for sequence in padded_sequences])
        padded_sequence = sequence + [0] * (max_length - len(sequence))

        # Convert to tensor and change data type to float
        input_tensor = torch.tensor(padded_sequence).unsqueeze(0).float()

        # Feed forward through the model
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class
        _, predicted = torch.max(output, 1)

        print("Model's response:", predicted.item())

interact_with_model(net, vocab)