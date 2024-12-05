# PEFT for Binary Image Classification using Vision Transformer

This project demonstrates how to perform parameter-efficient fine-tuning (PEFT) on a Vision Transformer (ViT) model for a binary image classification. The configuration used here is LoRA (Low-Rank Adaptation) and Hugging face transformers for fine tuning.

## Installation

1. Clone this repository.
2. We have used Python 3.8 as many dependencies used in this project are compitable with mentioned version.
3. Create a virtual environment (`python -m venv .venv`)
4. Activate the installed virtual environment (`source .venv/bin/activate` - .venv is the name of the virtual environment created)
5. Install the dependencies from requirements.txt using the command `pip install -r requirements.txt`
6. You need to access GPU for the processing.
7. Since we have implemented the code according to our image dataset directory, You can customize the `data_dir` variable in the script to point to your dataset directory.

## Installing VIT Model used using Hugging Face CLI

1. To use the Hugging Face CLI, you need to have the huggingface_hub installed. Install it using `pip install huggingface_hub`
2. Log in to your Hugging Face account (`huggingface-cli login`) ---- OPTIONAL
3. After loggin, it will ask for token Id ----> `Go to Hugging Face Website → Login → Click on your profile → Go to Settings → Select "Access Tokens" → Click "New Token" → Copy your token.`
3. Download the VIT model using command `huggingface-cli download google/vit-base-patch16-224-in21k`
4. The model is being downloaded in .cache folder in the root.

### Additional Notes
The model vit-base-patch16-224-in21k is pre-trained on the ImageNet-21k dataset and can be used for a wide range of image classification tasks. This model works good for the Binary Image Classification 
