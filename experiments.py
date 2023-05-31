import torch
from PIL import Image
import cv2
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"On device {device}")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)

images = []
images.append(Image.open("experiments_im/axel1.jpg"))
images.append(Image.open("experiments_im/axel2.jpg"))
images.append(Image.open("experiments_im/axel3.jpg"))
images.append(Image.open("experiments_im/johan1.jpg"))
images.append(Image.open("experiments_im/johan2.jpg"))
images.append(Image.open("experiments_im/johan3.jpg"))

images_cv = []
im = cv2.imread("experiments_im/axel1.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
images_cv.append(im)
im = cv2.imread("experiments_im/axel2.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
images_cv.append(im)
im = cv2.imread("experiments_im/axel3.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
images_cv.append(im)
im = cv2.imread("experiments_im/johan1.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
images_cv.append(im)
im = cv2.imread("experiments_im/johan2.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
images_cv.append(im)
im = cv2.imread("experiments_im/johan3.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
images_cv.append(im)

image_transforms = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)

features = []
features = []
for i, img in enumerate(images):
    # transform cropped img accordingly
    im = image_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = dinov2(im).squeeze(0)
    print(f"DINOv2: Forward pass for image {i}/{len(images)} done")
    features.append(embedding.cpu())

features_ = torch.stack(features).reshape(len(images), -1).numpy()
similarity_matrix = cosine_similarity(features_)
euclidean_matrix = euclidean_distances(features_)

# Generate table of scores
num_images = len(images)
score_table1 = np.zeros((num_images, num_images))
score_table2 = np.zeros((num_images, num_images))
for i in range(num_images):
    for j in range(num_images):
        score_table1[i, j] = similarity_matrix[i, j]
        score_table2[i, j] = euclidean_matrix[i, j]

# Display the score table
# print(score_table)
headings = ["axel1", "axel2", "axel3", "johan1", "johan2", "johan3"]
print('\n Creating sim-matrix for DINOv"')
print("DINOV2")
print("Cosine-Similarity (-1) to (1)")
print("\t", end="")
for heading in headings:
    print(f"{heading}\t", end="")
print()
for i, heading in enumerate(headings):
    print(f"{heading}\t", end="")
    for j in range(num_images):
        print(f"{score_table1[i, j]:.2f}\t", end="")
    print()
print("-*-" * 20)
print("\nEuclidean-Similarity")
print("\t", end="")
for heading in headings:
    print(f"{heading}\t", end="")
print()
for i, heading in enumerate(headings):
    print(f"{heading}\t", end="")
    for j in range(num_images):
        print(f"{score_table2[i, j]:.2f}\t", end="")
    print()

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
features_sam = []
for i, im in enumerate(images_cv):
    predictor.set_image(im.to(device))
    features_sam.append(predictor.get_image_embedding())
    predictor.reset_image()
    print(f"SAM: Forward pass for image {i}/{len(images)} done")

features_ = torch.stack(features_sam).reshape(len(images), -1).numpy()
similarity_matrix = cosine_similarity(features_)
euclidean_matrix = euclidean_distances(features_)

# Generate table of scores
num_images = len(images)
score_table1 = np.zeros((num_images, num_images))
score_table2 = np.zeros((num_images, num_images))
for i in range(num_images):
    for j in range(num_images):
        score_table1[i, j] = similarity_matrix[i, j]
        score_table2[i, j] = euclidean_matrix[i, j]

print("SEGMENT-ANYHTING-MODEL")
print("Cosine-Similarity (-1) to (1)")
print("\t", end="")
for heading in headings:
    print(f"{heading}\t", end="")
print()
for i, heading in enumerate(headings):
    print(f"{heading}\t", end="")
    for j in range(num_images):
        print(f"{score_table1[i, j]:.2f}\t", end="")
    print()
print("-*-" * 20)
print("\nEuclidean-Similarity")
print("\t", end="")
for heading in headings:
    print(f"{heading}\t", end="")
print()
for i, heading in enumerate(headings):
    print(f"{heading}\t", end="")
    for j in range(num_images):
        print(f"{score_table2[i, j]:.2f}\t", end="")
    print()
