"""
    TensorFlow training code for
    "Data-Integrated Semi-Supervised Attention Enhances Performance 
    and Interpretability of Biological Classification Tasks"
    
    This file includes:
     * Enzyme/non-enzyme classification training code which reproduces DSSA

    2025 Jun Kim
"""

import numpy as np
import pandas as pd
import os
import json
import gc
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_positions(pos_str):
    # Parses the json formatted functional positions
    try:
        return json.loads(pos_str)
    except json.JSONDecodeError:
        return []
    
def load_sequences(file_path):
    # This is for loading the non_enyzme protein sequences
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('>')
            if len(parts) > 1:
                seq = parts[1]
                sequences.append(seq)
    return sequences

def encode_sequence(seq, max_length):
    seq = seq.upper()
    int_seq = [aa_to_int.get(aa, NUM_AA-1) for aa in seq]  # 'X' index for unknowns or fillers
    if len(int_seq) < max_length:
        int_seq += [NUM_AA-1] * (max_length - len(int_seq))
    else:
        int_seq = int_seq[:max_length]
    one_hot = np.zeros((max_length, NUM_AA), dtype=np.float32)
    one_hot[np.arange(max_length), int_seq] = 1.0
    return one_hot

def create_saliency_maps_custom(important_positions_list, max_length, nonposition_num, nonposition_value):
    saliency_maps = np.zeros((len(important_positions_list), max_length), dtype=np.float32)
    for i, positions in enumerate(important_positions_list):
        if positions:
            positions = [pos-1 for pos in positions if pos-1 < max_length]  # 0 indexing
            if positions:
                saliency_maps[i, positions] = 1  # Position value
                # Determine available positions not in the important positions
                available_positions = list(set(range(max_length)) - set(positions))
                if available_positions:
                    # Randomly select nonposition_num available positions
                    random_pos = random.sample(available_positions, min(nonposition_num, len(available_positions)))
                    saliency_maps[i, random_pos] = 0.01  # Non-position value
    return saliency_maps

def create_saliency_maps_custom_main(important_positions_list, max_length, nonposition_num, nonposition_value):
    saliency_maps = np.zeros((len(important_positions_list), max_length), dtype=np.float32)
    for i, positions in enumerate(important_positions_list):
        if positions:
            positions = [pos-1 for pos in positions if pos-1 < max_length]  # 0 indexing
            if positions:
                saliency_maps[i, positions] = 1  # Position value
                # Determines available positions not in the important positions
                available_positions = list(set(range(max_length)) - set(positions))
                if available_positions:
                    # Randomly selects nonposition_num available positions
                    random_pos = random.sample(available_positions, min(nonposition_num, len(available_positions)))
                    saliency_maps[i, random_pos] = 0.01  # Non-position value
    return saliency_maps

def compute_combined_loss_cosine(y_true, y_pred, student_saliency, teacher_saliency, saliency_mask, alpha):
    # Classification loss
    bce_loss = classification_loss_fn(y_pred, y_true.squeeze(1))
    # Apply mask
    masked_student = student_saliency * saliency_mask
    masked_teacher = teacher_saliency * saliency_mask
    # Flatten the saliency maps
    masked_student_flat = masked_student.view(masked_student.size(0), -1)
    masked_teacher_flat = masked_teacher.view(masked_teacher.size(0), -1)
    # Normalize the vectors
    student_norm = masked_student_flat.norm(p=2, dim=1) + 1e-8
    teacher_norm = masked_teacher_flat.norm(p=2, dim=1) + 1e-8
    cosine_similarity = (masked_student_flat * masked_teacher_flat).sum(dim=1) / (student_norm * teacher_norm)
    # Cosine loss (1 - cosine similarity)
    cosine_loss = (1 - cosine_similarity).mean()
    # Total loss
    total_loss = bce_loss + alpha * cosine_loss
    return total_loss

class ModifiedResNet18(nn.Module):
    def __init__(self, input_channels=1, pretrained=False):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        # Modifies the first convolution layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        # Removes the max pooling layer to preserve spatial dimensions
        self.resnet.maxpool = nn.Identity()
        # Modifies layers to reduce downsampling (to preserve spatial dimensions)
        self._modify_resnet_layers()
    def _modify_resnet_layers(self):
        # Changes the stride in layer3 and layer4 to 1 to prevent further downsampling
        for layer in [self.resnet.layer3, self.resnet.layer4]:
            for block in layer:
                block.conv1.stride = (1, 1)
                if block.downsample:
                    block.downsample[0].stride = (1, 1)
    def forward(self, x):
        # Collects features before residual connections
        # Returns the final output and the features
        features = {}
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # Layer1
        before_layer1 = x.clone()
        x = self.resnet.layer1(x)
        features['layer1'] = before_layer1  # Before skip connections in layer1
        # Layer2
        before_layer2 = x.clone()
        x = self.resnet.layer2(x)
        features['layer2'] = before_layer2  # Before skip connections in layer2
        # Layer3
        before_layer3 = x.clone()
        x = self.resnet.layer3(x)
        features['layer3'] = before_layer3  # Before skip connections in layer3
        # Layer4 (Final Output)
        x = self.resnet.layer4(x)
        features['layer4'] = x 
        return x, features 

class ResNetAttentionModel(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, seq_length=300):
        super(ResNetAttentionModel, self).__init__()
        self.seq_length = seq_length
        self.resnet = ModifiedResNet18(input_channels=input_channels, pretrained=False).to(device)
        # Defines the classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),                  
            nn.Linear(512, num_classes)    
        )
        
    def forward(self, x):
        # Separate attention layers for low, mid, and high-level features
        # Layer1: 64 channels, layer2: 128 channels, layer3: 256 channels
        # Pass input through the modified ResNet18 to get final features and intermediate features
        features, intermediate_features = self.resnet(x)  # features: (batch_size, 512, H, W)
        logits = self.classifier(features).squeeze(1)  # Shape: (batch_size, num_classes)
        # Extract intermediate features before residual connections
        feat_low = intermediate_features['layer1']   # (batch_size, 64, H1, W1)
        feat_mid = intermediate_features['layer2']   # (batch_size, 128, H2, W2)
        feat_high = intermediate_features['layer3']  # (batch_size, 256, H3, W3)
        feat_vhigh = intermediate_features['layer4']
        # Generate attention layers
        attention_low = self.generate_attention(feat_low, target_height=self.seq_length)
        attention_mid = self.generate_attention(feat_mid, target_height=self.seq_length)
        attention_high = self.generate_attention(feat_high, target_height=self.seq_length)
        attention_vhigh = self.generate_attention(feat_vhigh, target_height=self.seq_length)
        attention_combined = (attention_low + attention_mid + attention_high) / 3  # Simple average
        return logits, attention_combined, attention_low, attention_mid, attention_high, attention_vhigh
        
    def generate_attention(self, feature_map, target_height):
        """
        Generates attention maps from feature maps using non-trainable operations.
        feature_map: (batch_size, C, H, W)
        target_height: height to fit to
        Returns:
            Attention map of shape (batch_size, target_height)
        """
        attention = torch.mean(feature_map, dim=1, keepdim=True)  
        attention = F.adaptive_avg_pool2d(attention, (self.seq_length, 1))  
        # Squeeze unnecessary dimensions
        attention = attention.squeeze(3).squeeze(1)  # Shape: (batch_size, seq_length)
        return attention

# Custom Dataset
class CustomProteinDataset(Dataset):
    def __init__(self, X, y, important_positions, saliency_mask, is_train=True, seq_length=300, num_aa=21):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  
        self.important_positions = important_positions
        self.saliency_mask = torch.tensor(saliency_mask, dtype=torch.float32)
        self.is_train = is_train
        self.seq_length = seq_length
        self.num_aa = num_aa
        self.teacher_saliency = self.create_initial_saliency()
    def create_initial_saliency(self):
        # Initializes saliency maps
        saliency_maps = np.zeros((len(self.important_positions), self.seq_length), dtype=np.float32)
        for i, positions in enumerate(self.important_positions):
            if positions:
                valid_pos = [pos-1 for pos in positions if pos-1 < self.seq_length]
                if valid_pos:
                    saliency_maps[i, valid_pos] = 1  
        return torch.tensor(saliency_maps, dtype=torch.float32)
    def update_saliency_maps(self, nonposition_num, nonposition_value):
        # Regenerate saliency maps by reassigning random positions for the training set.
        if not self.is_train:
            return  
        saliency_maps = self.teacher_saliency.clone().numpy()
        for i, positions in enumerate(self.important_positions):
            if positions:
                saliency_maps[i, :] = 0.0  # Reset saliency
                valid_pos = [pos-1 for pos in positions if pos-1 < self.seq_length]
                if valid_pos:
                    saliency_maps[i, valid_pos] = 1  # Position value
                available_positions = list(set(range(self.seq_length)) - set(valid_pos))
                if available_positions:
                    random_pos = random.sample(available_positions, min(nonposition_num, len(available_positions)))
                    saliency_maps[i, random_pos] = nonposition_value
        self.teacher_saliency = torch.tensor(saliency_maps, dtype=torch.float32)
        # Update the saliency mask
        self.saliency_mask = (self.teacher_saliency > 0).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.teacher_saliency[idx], self.saliency_mask[idx]

torch.cuda.set_per_process_memory_fraction(0.95, device=0)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect()
torch.cuda.empty_cache()

set_seed(42)

csv_file_path = 'uniprot_sequences_with_positions_training.csv'
df = pd.read_csv(csv_file_path)
sequences = df['sequence'].tolist()
labels = df['label'].tolist()

# Apply the parsing function to the 'functional_positions' column
valid_functional_positions = df['functional_positions'].apply(parse_positions).tolist()
    
non_enzyme_file = 'non_enzyme_new_data_sequence.txt'
print("Loading sequences...")
non_enzymes = load_sequences(non_enzyme_file)
num_enzymes = len(sequences)
num_non_enzymes = len(non_enzymes)
print(f"Number of enzyme sequences: {num_enzymes}")
print(f"Number of non-enzyme sequences: {num_non_enzymes}")

if num_non_enzymes > num_enzymes:
    # Downsamples non-enzymes to match the number of enzymes
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(num_non_enzymes, size=num_enzymes, replace=False)
    non_enzymes_balanced = [non_enzymes[i] for i in sampled_indices]
    print(f"Downsampled non-enzyme sequences to {num_enzymes}")
else:
    non_enzymes_balanced = non_enzymes
    print("Non-enzyme and enzyme sequences are already balanced.")
    
all_sequences = non_enzymes_balanced + sequences
labels = np.array([0]*len(non_enzymes_balanced) + [1]*len(sequences))
important_positions_ph = [[] for _ in range(len(non_enzymes_balanced))]
important_positions_combined = important_positions_ph + valid_functional_positions
amino_acids = 'ACDEFGHIKLMNPQRSTVWYX' # 'X' in case there is unknowns
aa_to_int = {aa: idx for idx, aa in enumerate(amino_acids)}
NUM_AA = len(amino_acids)
SEQ_LENGTH = 300  # Can be adjusted
NUM_CLASSES = 1    # Can be adjusted
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoded_sequences = np.array([encode_sequence(seq, SEQ_LENGTH) for seq in all_sequences])  # Shape: (num_samples, SEQ_LENGTH, NUM_AA)
X = np.expand_dims(encoded_sequences, axis=1)  # Shape: (num_samples, 1, SEQ_LENGTH, NUM_AA)
y = labels

X_train_np, X_val_np, y_train_np, y_val_np, imp_train_np, imp_val_np = train_test_split(
    X,
    y,
    important_positions_combined,
    test_size=0.2,
    random_state=42,
    stratify=y
)

new_val_csv_path = 'uniprot_sequences_with_positions_validation.csv' 
df_new_val = pd.read_csv(new_val_csv_path) 
sequences_new_val = df_new_val['sequence'].tolist() 
labels_new_val = df_new_val['label'].tolist() 
valid_functional_positions_new_val = df_new_val['functional_positions'].apply(parse_positions).tolist() 
encoded_sequences_new_val = np.array([encode_sequence(seq, SEQ_LENGTH) for seq in sequences_new_val])  
X_new_val = np.expand_dims(encoded_sequences_new_val, axis=1)
y_new_val = np.array(labels_new_val)

new_val_saliency = create_saliency_maps_custom(valid_functional_positions_new_val, SEQ_LENGTH, nonposition_num=0, nonposition_value=0.0)
new_val_mask = (new_val_saliency > 0).astype(np.float32) 
# Create the new validation dataset and DataLoader
new_val_dataset = CustomProteinDataset(
    X=X_new_val,
    y=y_new_val,
    important_positions=valid_functional_positions_new_val,
    saliency_mask=new_val_mask,
    is_train=False,
    seq_length=SEQ_LENGTH
)
new_val_loader = DataLoader(new_val_dataset, batch_size=128, shuffle=False)

non_num = 60
alpha = 3

# Creates saliency maps for training and validation

train_saliency = create_saliency_maps_custom_main(imp_train_np, SEQ_LENGTH, non_num, 0.01)
val_saliency = create_saliency_maps_custom_main(imp_val_np, SEQ_LENGTH, non_num, 0.01)
train_mask = (train_saliency > 0).astype(np.float32)
val_mask = (val_saliency > 0).astype(np.float32)
# Initialize the custom datasets
train_dataset = CustomProteinDataset(
    X=X_train_np,
    y=y_train_np,
    important_positions=imp_train_np,
    saliency_mask=train_mask,
    is_train=True,
    seq_length=SEQ_LENGTH
)
val_dataset = CustomProteinDataset(
    X=X_val_np,
    y=y_val_np,
    important_positions=imp_val_np,
    saliency_mask=val_mask,
    is_train=False,
    seq_length=SEQ_LENGTH
)

# Initialize the DataLoaders
batch_size = 128  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Initialize the model and move to device
model = ResNetAttentionModel(num_classes=1, input_channels=1, seq_length=SEQ_LENGTH).to(device)
# Define optimizer before the training loop
optimizer = Adam(model.parameters(), lr=1e-3)
# Define loss functions
classification_loss_fn = nn.BCEWithLogitsLoss()
attention_loss_fn = nn.MSELoss()  # Assuming you'll have targets for attention maps

best_val_accuracy = 0.0
best_val_loss = float('inf')
best_new_val_accuracy = 0.0  # NEW
best_new_val_loss = float('inf')  # NEW
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    print(f"\nStart of epoch {epoch + 1}/{EPOCHS} for NONPOSITION_NUM={non_num}, ALPHA={alpha}")
    # Update saliency maps for training set
    train_dataset.update_saliency_maps(non_num, 0.01)
    for batch_idx, (inputs, labels, teacher_saliency_batch, saliency_mask_batch) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        teacher_saliency_batch = teacher_saliency_batch.to(device)
        saliency_mask_batch = saliency_mask_batch.to(device)
        optimizer.zero_grad()
        outputs, student_att, attention_low, attention_mid, attention_high, attention_vhigh = model(inputs)  # outputs: (batch_size, num_classes), student_att: (batch_size, 300)
        # Compute combined loss
        loss = compute_combined_loss_cosine(labels, outputs, attention_high, teacher_saliency_batch, saliency_mask_batch, alpha)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        correct = (preds.squeeze() == labels.byte().squeeze(1)).sum().item()
        train_correct += correct
        train_total += labels.size(0)
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {(train_correct/train_total):.2f}")
    avg_train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total
    # Validation on the original validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels, teacher_saliency_batch, saliency_mask_batch in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_saliency_batch = teacher_saliency_batch.to(device)
            saliency_mask_batch = saliency_mask_batch.to(device)
            outputs, student_att, attention_low, attention_mid, attention_high, attention_vhigh = model(inputs)
            loss = compute_combined_loss_cosine(labels, outputs, attention_high, teacher_saliency_batch, saliency_mask_batch, alpha)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct = (preds == labels.byte().squeeze(1)).sum().item()
            val_correct += correct
            val_total += labels.size(0)
    avg_val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total
    # Validation on the new validation set
    new_val_loss = 0.0
    new_val_correct = 0
    new_val_total = 0
    with torch.no_grad():
        for inputs, labels, teacher_saliency_batch, saliency_mask_batch in new_val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_saliency_batch = teacher_saliency_batch.to(device)
            saliency_mask_batch = saliency_mask_batch.to(device)
            outputs, student_att, attention_low, attention_mid, attention_high, attention_vhigh = model(inputs)
            loss = compute_combined_loss_cosine(labels, outputs, attention_high, teacher_saliency_batch, saliency_mask_batch, alpha)
            new_val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct = (preds == labels.byte().squeeze(1)).sum().item()
            new_val_correct += correct
            new_val_total += labels.size(0)
    avg_new_val_loss = new_val_loss / new_val_total
    new_val_accuracy = new_val_correct / new_val_total
    print(f"Epoch {epoch + 1}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}, "
          f"New Val Loss: {avg_new_val_loss:.4f}, "
          f"New Val Accuracy: {new_val_accuracy:.4f}")  # UPDATED LINE