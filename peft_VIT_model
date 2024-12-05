# Meet Trusarbhai Patel - 202407335
# Dhruvi Nilesh Thakkar - 202401915
# Jyotsna Joshy - 202404936

# Imports
import os
import torch
import shutil  # For copying files within directories
from torchvision import datasets, transforms  # For data processing and loading
from torch.utils.data import DataLoader, random_split  # For data splitting and batching
from transformers import AutoModelForImageClassification, AutoFeatureExtractor  # For pre-trained models
from peft import LoraConfig, get_peft_model  # For parameter-efficient fine-tuning
from torch.optim import Adam  # Optimizer for training
from torch.nn import CrossEntropyLoss  # Loss function for classification

# === Citations ===
# PyTorch Library
"""
Citation for PyTorch:
PyTorch Development Team. "PyTorch: An Imperative Style, High-Performance Deep Learning Library."
Facebook, 2016, https://pytorch.org.

For more information, visit https://pytorch.org.
"""

# torchvision Library
"""
Citation for torchvision:
PyTorch Team. "torchvision: Datasets, Transforms and Models for Computer Vision."
PyTorch, 2016, https://pytorch.org/vision/stable/index.html.

For more information, visit https://pytorch.org/vision/stable/index.html.
"""

# Hugging Face Transformers Library
"""
Citation for transformers:
Hugging Face. "Transformers: State-of-the-art Natural Language Processing for Pytorch, TensorFlow, and JAX."
Hugging Face, 2020, https://huggingface.co/transformers/.

For more information, visit https://huggingface.co/transformers/.
"""

# PEFT Library for Parameter Efficient Fine-Tuning
"""
Citation for peft:
Hu, Hongwei, et al. "PEFT: Parameter Efficient Fine-Tuning."
2023, https://github.com/huggingface/peft.

For more information, visit https://github.com/huggingface/peft.
"""

# === Step 1: Dataset Preparation ===

# Define the main directory where the dataset is stored
data_dir = "path/to/dataset"

# Storing the dataset keeping the same directory structure in memory, it is the temporary directory to hold the restructured data
restructured_dir = "/tmp/restructured_dataset"

# Creating the restructured directory if it doesn't exist
if not os.path.exists(restructured_dir):
    os.makedirs(restructured_dir)

# Loop through each patient folder, and restructure data into 2 classes (0, 1) as per our dataset.
""" Below shown is the dataset directory structure of the project  
/path/to/dataset/
├── patient_folder1/
│   ├── 0/images
│   └── 1/images
├── patient_folder2/
│   ├── 0/images
│   └── 1/images """

for patient_folder in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_folder)
    # Ensuring that it is a folder
    if os.path.isdir(patient_path):
        # Only process folders "0" and "1" (binary classification)
        for class_folder in ["0", "1"]:
            class_path = os.path.join(patient_path, class_folder)
            if os.path.exists(class_path):
                dest_class_dir = os.path.join(restructured_dir, class_folder)
                # Create destination folder for the class
                os.makedirs(dest_class_dir, exist_ok=True) 
                # Copying each image to the respective class folder
                for img_file in os.listdir(class_path):
                    src_img_path = os.path.join(class_path, img_file)
                    dest_img_path = os.path.join(dest_class_dir, f"{patient_folder}_{img_file}")
                    # Copying the image file to destination
                    shutil.copy(src_img_path, dest_img_path) 

# Define image transformations for data preprocessing
transform = transforms.Compose([
    # Resizing the images to the expected input size (224x224)
    transforms.Resize((224, 224)),  
    # Converting all the images to PyTorch tensor
    transforms.ToTensor(),  
    # Normalizing the images to a [-1, 1] range
    transforms.Normalize([0.5], [0.5])  
])

# Loading the dataset using the restructured directory and apply the defined transformations
dataset = datasets.ImageFolder(root=restructured_dir, transform=transform)

# Spliting the dataset into two subsets (A and B) for training and evaluation. We are doing the split of 70% and 30%
dataset_size = len(dataset)
a_size = int(0.7 * dataset_size)  # 70% for training
b_size = dataset_size - a_size  # 30% for evaluation
dataset_A, dataset_B = random_split(dataset, [a_size, b_size])

# Creating DataLoader objects for training and evaluation and setting the batch size as 32
batch_size = 32
# Shuffle training data
loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True) 
# No shuffle for evaluation data 
loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=False)  

# === Step 2: Loading Pre-trained Model and Applying PEFT ===
# Defining the local path to the pre-trained model in Hugging Face's cache. We are using model Vision Transformer (ViT) model for fine tuning
# Using this model as it is suitable and ideal for binary classification in image dataset. It has powerful pre-trained features from ImageNet-21k and the Vision Transformer architecture.
# We have created a hugging face cache using hugging face cli commands as it was throwing error accessing it online.
model_path = "path/to/downloaded VIT Model/snapshots/ID"

# Loading the feature extractor and the pre-trained model
print("Loading feature extractor and model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(
    model_path,
    # Since we have binary classification, so 2 output labels
    num_labels=2  
)

print("Model loaded successfully.")

# Defining PEFT (Parameter Efficient Fine-Tuning) configuration using LORA (Low-Rank Adaptation)
peft_config = LoraConfig(
    r=8,  # Rank of the low-rank adaptation
    lora_alpha=16,  # Scaling factor for the low-rank matrices
    lora_dropout=0.1,  # Dropout rate to avoid overfitting
    target_modules=["query", "key", "value"],  # Apply PEFT to the attention layers
    inference_mode=False  # Setting to False for fine-tuning
)

# Applying the PEFT to the model
peft_model = get_peft_model(model, peft_config)
print("PEFT model initialized.")

# === Step 3: Fine-Tune the Model ===

# Setting device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)  # Move model to the appropriate device (GPU or CPU)

# Defining the optimizer (Adam) and loss function (CrossEntropyLoss)
optimizer = Adam(peft_model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

# Training loop for fine-tuning
epochs = 5  # Number of epochs to train the model
print("Starting fine-tuning...")

for epoch in range(epochs):
    peft_model.train()  # Set the model to training mode
    total_loss = 0
    for images, labels in loader_A:  # Iterate through the training DataLoader
        images, labels = images.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()  # Clear previous gradients
        outputs = peft_model(images).logits  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights
        total_loss += loss.item()  # Accumulate total loss for this epoch

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

print("Fine-tuning completed!")

# === Step 4: Extract Embeddings for Dataset B ===

# Setting the model to evaluation mode for embedding extraction
peft_model.eval()

# Define a function to extract embeddings (feature representations) from the model
def extract_embeddings(dataloader, model, device):
    embeddings, labels = [], []
    with torch.no_grad():  # No gradient calculation required during inference
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images).logits  # Get model outputs (embeddings)
            embeddings.append(outputs.cpu())  # Store embeddings on CPU
            labels.append(targets.cpu())  # Store labels on CPU
    return torch.cat(embeddings), torch.cat(labels)

# Extract embeddings for Dataset B (evaluation data)
print("Extracting embeddings for Dataset B...")
embeddings_B, labels_B = extract_embeddings(loader_B, peft_model, device)

# Saving the embeddings for further analysis
torch.save((embeddings_B, labels_B), "outputFileName.pt")
print("Embeddings saved as 'outputFileName.pt'.")

# === Step 5: Save the Fine-Tuned Model ===

# Save the fine-tuned PEFT model for future use
peft_model.save_pretrained("peft_finetuned_model_using_VIT")
print("Fine-tuned model saved as 'peft_finetuned_model_using_VIT'.")
