import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import CLIPTextModel



class CustomDataset(Dataset):
    def __init__(self, image_dir, text_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.text_file = text_file
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Read the prompts from the text file
        with open(text_file, "r") as f:
            self.prompts = f.readlines()
        
        self.image_files = os.listdir(image_dir) 

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        prompt = self.prompts[idx].strip()
        text_input = self.tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=77)

        return {
            "pixel_values": image,
            "input_ids": text_input["input_ids"].squeeze(0),  # Convert from batch size 1
            "attention_mask": text_input["attention_mask"].squeeze(0),
        }

# Set up image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to match the model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the tokenizer from the pre-trained model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the custom dataset
dataset = CustomDataset(image_dir=r"C:\Users\Reyan\Desktop\Projects\Text_to_image\image", text_file=r"C:\Users\Reyan\Desktop\Projects\Text_to_image\text\prompts.txt", tokenizer=tokenizer, transform=transform)

# Load a pre-trained model from Hugging Face
def load_model():
    
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline.to(device)
    return pipeline

# Initialize the model pipeline
pipeline = load_model()
unet = UNet2DConditionModel.from_pretrained(pipeline.unet)
text_encoder = CLIPTextModel.from_pretrained(pipeline.text_encoder)

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = AdamW(unet.parameters(), lr=1e-5)

def train():
    unet.train()
    for epoch in range(3):  # Set number of epochs
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to("cuda")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # Forward pass
            loss = unet(pixel_values, encoder_hidden_states=text_encoder(input_ids).last_hidden_state).loss

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# Start fine-tuning
train()
unet.save_pretrained("fine_tuned_unet")
text_encoder.save_pretrained("fine_tuned_text_encoder")


def generate_image(prompt):
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    # Get the hidden states from the text encoder
    text_embeddings = text_encoder(input_ids).last_hidden_state

    # Generate image from the prompt
    with torch.no_grad():
        generated_image = unet.generate(text_embeddings)

    return generated_image

