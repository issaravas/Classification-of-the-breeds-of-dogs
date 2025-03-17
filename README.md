# 🐶 Classification of Dog Breeds

## Project Overview
A computer vision project focused on classifying different dog breeds using deep learning.
We will use **transfer learning with Convolutional Neural Networks (CNNs)** to achieve high accuracy in breed identification.

### **Dataset Structure**  
The dataset is based on the **[Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)**, containing images of various dog breeds.  
The data has been **cleaned, resized, and augmented** to improve model performance.

## 🛠️ **Project Workflow**
**1. Data Preprocessing**
- Analyzed the dataset: the number of images per class, the distribution of image sizes.
- Resized all images to **224x224** while preserving aspect ratio (padding).
- Augmented images to prevent overfitting.
- Visualized sample images from the dataset before augmentation and resizing, as well as after the applied transformations.

**2. Data Loading**
- Converted images into **PyTorch tensors**.
- Created **train, validation, and test DataLoaders** for efficient batch processing.

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
