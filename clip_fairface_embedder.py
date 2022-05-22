from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import torch
from PIL import Image
from os import listdir

#Initialize CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

SOURCE_DIR = f'E:\\fairface\\fairface-img-margin125-trainval\\train'
target_images = listdir(SOURCE_DIR)

embeddings_ = []

for target_ in target_images:
    with torch.no_grad():
        img = Image.open(f'{SOURCE_DIR}\\{target_}').convert('RGB')
        input = processor(images=img,return_tensors='pt')
        emb = model.get_image_features(**input).numpy().squeeze()
        embeddings_.append(emb)

emb_arr = np.array(embeddings_)
emb_df = pd.DataFrame(emb_arr,index=target_images)
emb_df.to_csv(f'E:\\fairface\\fairface_images_full.vec', sep = ' ')