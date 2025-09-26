# Cotton Disease Classification using Transfer Learning

A deep learning project for classifying cotton plant diseases using transfer learning with pre-trained convolutional neural networks. This project was developed as a college mini-project to demonstrate the effectiveness of transfer learning in agricultural image classification.

## Project Overview

This project aims to automatically identify cotton plant diseases by leveraging the power of transfer learning. Transfer learning is a machine learning technique where we reuse a pre-trained model as the starting point for a model on a new task, allowing us to build accurate models with less data and computational resources.

### Problem Statement

Manual identification of cotton plant diseases is time-consuming and requires expert knowledge. This project addresses this challenge by developing an automated system that can classify whether cotton plants and leaves are healthy or diseased.

## Dataset

You can access the dataset used in this project from the following Google Drive link:
[Dataset](https://drive.google.com/drive/folders/1TchYRgFXw4dQWIpBTx9tNv7xJOhzBZPS?usp=sharing)

The dataset contains images of cotton plants and leaves in four categories:
- **Fresh Cotton Leaf** - Healthy cotton leaves
- **Fresh Cotton Plant** - Healthy cotton plants
- **Diseased Cotton Leaf** - Cotton leaves with diseases
- **Diseased Cotton Plant** - Cotton plants with diseases

**Dataset Statistics:**
- Training Images: 1,951
- Validation Images: 324
- Image Size: 224x224 pixels
- Total Classes: 4

### Data Preprocessing and Image Augmentation

Extensive data preprocessing and image augmentation techniques were implemented to improve model performance and address dataset imbalances:

1. **Image Preprocessing:**
   - Normalization (pixel values rescaled to 0-1)
   - Resizing to 224x224 pixels for model compatibility
   - Image denoising techniques
   - Morphological operations for noise reduction

2. **Data Augmentation:**
   - Shear transformation (range: 0.2)
   - Zoom augmentation (range: 0.2)
   - Width shift (range: 0.1)
   - Height shift (range: 0.1)
   - Rotation (range: 10 degrees)
   - Horizontal flip

3. **Dataset Balancing:**
   - Addressed uneven distribution between healthy and diseased samples
   - Applied sampling techniques to balance the dataset

## Models Architecture

Three state-of-the-art CNN architectures were implemented and compared:

### 1. ResNet152V2

ResNet, short for "Residual Networks," is a deep neural network architecture known for its exceptional performance in image classification tasks. It introduced the concept of residual connections, which allows the network to learn and optimize extremely deep architectures.

- **Architecture:** 152-layer residual network (Version 2)
- **Key Features:** Deep residual connections that solve vanishing gradient problems
- **Transfer Learning:** Pre-trained on ImageNet, frozen layers with custom classification head
- **Performance:** **96% mean accuracy**
- **Parameters:** Frozen pre-trained weights with trainable classification layers

### 2. InceptionV3

InceptionNet, also known as GoogLeNet, is a convolutional neural network architecture developed by Google. It is famous for its efficiency and ability to capture features at multiple scales.

- **Architecture:** Google's Inception architecture with factorized convolutions
- **Key Features:** Efficient feature extraction using inception modules with multiple kernel sizes in parallel
- **Transfer Learning:** Pre-trained on ImageNet, frozen layers with custom classification head
- **Performance:** **94% mean accuracy**
- **Advantages:** Computational efficiency while maintaining high accuracy

### 3. DenseNet201

DenseNet, short for "Densely Connected Convolutional Networks," is a neural network architecture that emphasizes feature reuse. Unlike traditional CNNs, where layers are connected sequentially, DenseNet connects each layer to every other layer in a feed-forward fashion.

- **Architecture:** 201-layer densely connected network
- **Key Features:** Dense connectivity pattern promoting feature reuse and efficient gradient flow
- **Transfer Learning:** Pre-trained on ImageNet, frozen layers with custom classification head
- **Performance:** **97% mean accuracy** (Best performing model)
- **Advantages:** Parameter efficiency due to feature reuse

## Implementation Details

### Transfer Learning Approach

We freeze the parameters from the pre-trained models and add custom classification layers:

1. **Frozen Base Model:** Pre-trained weights are frozen to preserve learned ImageNet features
2. **Custom Classification Head:** Added new fully connected layers for cotton disease classification
3. **Fine-tuning:** Only the final classification layers are trained on our cotton dataset

### Model Configuration

```python
# Common architecture for all models
base_model = PretrainedModel(input_shape=(224, 224, 3), 
                            weights='imagenet', 
                            include_top=False)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = Flatten()(base_model.output)
x = Dense(4, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=x)
```

### Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Size:** 16
- **Epochs:** 20
- **Input Shape:** (224, 224, 3)
- **Hardware:** GPU acceleration enabled

## Results and Performance

| Model | Mean Accuracy | Training Details | Performance Ranking |
|-------|---------------|------------------|--------------------|
| **DenseNet201** | **97%** | Achieved through feature reuse | 1st (Best) |
| **ResNet152V2** | **96%** | Deep residual learning | 2nd |
| **InceptionV3** | **94%** | Multi-scale feature extraction | 3rd |

### Key Findings

1. **DenseNet201** achieved the highest accuracy (97%), demonstrating the effectiveness of dense connectivity for feature reuse
2. **ResNet152V2** performed excellently (96%) with its deep residual architecture solving vanishing gradient problems
3. **InceptionV3** showed strong performance (94%) with computational efficiency
4. All models successfully learned to distinguish between healthy and diseased cotton plants/leaves
5. Transfer learning proved highly effective, achieving high accuracy with limited training data

## Project Structure

```
Cotton_disease_classification/
├── README.md
├── cotton_disease_classification_RESNET152V2.ipynb
├── cotton_disease_classification_Inception_v3.ipynb
└── dataset/ (available via Google Drive link)
    ├── train/
    │   ├── fresh_cotton_leaf/
    │   ├── fresh_cotton_plant/
    │   ├── diseased_cotton_leaf/
    │   └── diseased_cotton_plant/
    └── val/
        ├── fresh_cotton_leaf/
        ├── fresh_cotton_plant/
        ├── diseased_cotton_leaf/
        └── diseased_cotton_plant/
```

## Technical Requirements

```python
# Key dependencies
tensorflow >= 2.x
keras
numpy
matplotlib
pandas
pillow
scikit-learn
glob
```

## Usage

### Training a Model

1. **Download the dataset** from the Google Drive link provided above
2. **Organize the dataset** according to the folder structure shown
3. **Update the data paths** in the notebook:
   ```python
   train_path = '/path/to/train'
   valid_path = '/path/to/val'
   ```
4. **Run the notebook** for your chosen architecture (ResNet152V2 or InceptionV3)
5. **Monitor training** through loss and accuracy plots
6. **Save the trained model** for inference

### Making Predictions

```python
# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('model_name.h5')

# Preprocess new images
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path/to/image', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = model.predict(img_array)
class_names = ['diseased cotton leaf', 'diseased cotton plant', 
               'fresh cotton leaf', 'fresh cotton plant']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)
```

## Key Insights and Learnings

1. **Transfer Learning Effectiveness:** Pre-trained models significantly reduced training time and improved accuracy compared to training from scratch

2. **Data Augmentation Impact:** Image augmentation techniques helped improve model generalization and reduced overfitting

3. **Architecture Comparison:** DenseNet201's dense connectivity made it most effective for this agricultural classification task

4. **Hyperparameter Tuning:** Careful selection of batch size, learning rate, and epochs was crucial for optimal performance

5. **Practical Application:** The high accuracy rates (94-97%) demonstrate the feasibility of automated cotton disease detection in real-world agricultural scenarios

## Model Performance Analysis

### Training Characteristics
- All models showed stable training with minimal overfitting
- Validation accuracy closely followed training accuracy
- Loss curves demonstrated good convergence
- Data augmentation helped prevent overfitting

### Computational Efficiency
- **InceptionV3:** Most computationally efficient
- **DenseNet201:** Moderate computational requirements with best accuracy
- **ResNet152V2:** Highest computational requirements but strong performance

## Future Improvements

- **Dataset Expansion:** Increase dataset size with more diverse samples and disease types
- **Disease-Specific Classification:** Extend to identify specific types of cotton diseases
- **Mobile Deployment:** Optimize models for mobile applications using TensorFlow Lite
- **Real-time Processing:** Implement real-time disease detection for field applications
- **Multi-crop Support:** Extend the approach to other crop diseases
- **Ensemble Methods:** Combine multiple models for improved accuracy

## Applications

- **Precision Agriculture:** Automated crop health monitoring systems
- **Early Disease Detection:** Rapid identification of plant diseases for timely intervention
- **Agricultural IoT:** Integration with drone and sensor technologies
- **Educational Tools:** Training resources for agricultural students and farmers
- **Research Support:** Facilitating agricultural research and disease studies

## Contributing

Contributions to improve the project are welcome! Areas for contribution:

- Adding new disease categories
- Implementing additional CNN architectures
- Improving data preprocessing techniques
- Optimizing model performance
- Creating mobile or web applications
- Adding model interpretability features

## Acknowledgments

- **ImageNet Dataset:** For providing pre-trained model weights
- **TensorFlow/Keras:** For the comprehensive deep learning framework
- **Google Colab:** For providing free GPU resources for training
- **Academic Institution:** For supporting this research project
- **Open Source Community:** For various tools and libraries used

## Citation

If you use this project in your research, please cite:

```
Cotton Disease Classification using Transfer Learning
[Your Institution], [Year]
Available at: https://github.com/Abhishek-1804/Cotton_disease_classification
```

---

**Note:** This project demonstrates the power of transfer learning in agricultural applications and serves as a foundation for more advanced crop monitoring systems. The high accuracy achieved (94-97%) across all three architectures validates the approach for practical agricultural use cases.