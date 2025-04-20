# 🐶 Classification of Dog Breeds

## Project Overview
A computer vision project focused on classifying different dog breeds using deep learning. We attempt to classify images of dogs into **one of 120 different breeds**. Due to the large number of classes and relatively few images per class (~150), this task is quite challenging, especially for visually similar breeds.

---

To address this, I experiment with:
- **Custom CNN from scratch**
- **Transfer learning with ResNet18**
- **Hierarchical approach**, where the model first predicts a general breed group and then the specific breed within that group

---

## Dataset
The dataset is based on the **[Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)**, containing images of various dog breeds.  

The dataset consists of:
- 120 dog breeds
- ~150 images per breed
- Images are resized and padded to unify input dimensions
- Grouped according to [American Kennel Club (AKC)](https://en.wikipedia.org/wiki/American_Kennel_Club) breed categories

## Flat Classification
Initially, I trained a custom CNN to directly classify all 120 breeds.

### CNN Architecture
A deep convolutional model with 6 convolutional blocks:
- Each block includes: Conv2D (3x3) + BatchNorm + ReLU + MaxPool
- The number of filters increases from 64 → 512
- Ends with AdaptiveAvgPooling + Fully Connected layers

```text
Conv1: 64  -> Conv2: 128  -> Conv3: 256
Conv4-Conv6: 512 each
FC1: 256 neurons, ReLU + Dropout
FC2: Output layer for 120 classes
```

#### Outcome
- The model struggled to learn meaningful representations
- Training was extremely slow and validation accuracy stayed under 2%
- Conclusion: not enough data per class; architecture not deep or efficient enough

To validate this, we first trained the same architecture on a **subset of 10 classes**, which worked far better (~88% accuracy). This confirmed the architecture itself worked in simpler settings.

---

## Hierarchical Conditional CNN
To overcome the limitations, we moved to a **Hierarchical Conditional CNN (HCNN)**.

### What is Hierarchical and Conditional?
Instead of predicting 120 breeds directly, the model first classifies the image into a **group** (e.g., Toy, Hound, Working), then uses that information to predict the **breed**.

This conditional architecture reflects the way humans categorize: first broadly, then specifically. It also reduces class imbalance and model complexity at each decision point.

### HCNN Architecture
- Shared convolutional backbone (same as in flat CNN)
- Two fully connected heads:
  - Group classification head: predicts the broader category
  - Breed classification head: takes image features **plus softmax probabilities from the group**

### Model Forward Pass:
```python
x = self.conv_block(x)
x = self.flatten(x)
x = self.dropout(x)

# Group prediction
group_out = self.group_head(x)
group_probs = F.softmax(group_out, dim=1)

# Conditional input
x_combined = torch.cat([x, group_probs], dim=1)
breed_out = self.breed_head(x_combined)
```

---

## Transfer Learning with ResNet18
To further improve performance, we applied **transfer learning** using a pre-trained **ResNet18** model from ImageNet.

### Strategy
- All layers were initially frozen to retain low-level features.
- Then, we **unfroze layers 2, 3, 4 and the final FC layer**, allowing the model to adapt higher-level representations.
- The original FC layer was replaced with a new linear layer for 120 output classes.

### Training Setup
- Optimizer: `Adam` with learning rate `1e-4`
- Loss function: `CrossEntropyLoss`
- Epochs: 25
- Batch size: 32
- Achieved ~**94% accuracy** on validation set

This model demonstrated strong generalization and stability, and was therefore **selected as the final model for web demo**.

## 🖥️ Web Demo (BreedScope)

To illustrate how this model can be used in a real-world application, a prototype website called **BreedScope** was designed. The site allows users to upload a photo of a dog and receive instant breed predictions using the trained model.

Key Features:
- Upload an image and receive breed prediction
- View the top predicted breed alongside the uploaded image
- Designed for clarity, simplicity, and instant interaction

Screenshots from the demo:

## How to run?
1. Install Dependencies  
`pip install -r requirements.txt`
2. Clone the Repository  
`git clone https://github.com/issaravas/Classification-of-the-breeds-of-dogs.git`
`cd Classification-of-the-breeds-of-dogs`
3. Download the dataset  **[Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)**
4. Create & Run Virtual Environment  
`python -m venv venv `
`source venv/bin/activate  # macOS/Linux`
`venv\Scripts\activate  # Windows`
5. Run Jupyter Notebook  
`jupyter notebook`
