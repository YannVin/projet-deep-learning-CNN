Basé sur la structure du projet et les fichiers fournis, je vais créer un exemple de fichier `README.md` pour votre projet de classification d'images CIFAR-10 et CIFAR-100 avec différents modèles de deep learning.

```markdown
# Deep Learning Project - CIFAR-10 & CIFAR-100 Image Classification

## Overview
This project aims to classify images from the CIFAR-10 and CIFAR-100 datasets using various deep learning architectures. Each folder contains different models trained on the respective datasets.

## Project Structure
```
├── models
│   ├── cifar10
│   │   ├── AlexNet.h5
│   │   ├── CNN_simple.h5
│   │   ├── DenseNet121.h5
│   │   ├── DenseNet201.h5
│   │   ├── LeNet.h5
│   │   ├── ResNet50.h5
│   │   └── VGGNet.h5
│   └── cifar100
│       ├── AlexNet.h5
│       ├── CNN_simple.h5
│       ├── DenseNet121.h5
│       ├── LeNet.h5
│       └── ResNet50.h5
└── scripts
    ├── cnn_cifar10.ipynb
    └── cnn_cifar100.ipynb
```
## Datasets
- **CIFAR-10**: Contains 60,000 32x32 color images in 10 different classes.
- **CIFAR-100**: Contains 60,000 32x32 color images in 100 different classes.

## Models
A variety of models have been trained on CIFAR-10 and CIFAR-100 datasets:
- **AlexNet**: One of the pioneer deep learning architectures for image classification.
- **CNN Simple**: A baseline convolutional neural network model for comparison purposes.
- **DenseNet121 & DenseNet201**: Feature-dense models that connect each layer to every other layer in a feed-forward fashion.
- **LeNet**: A classic CNN architecture that is smaller and quicker to train.
- **ResNet50**: Implements residual learning to facilitate training of deeper network architectures.
- **VGGNet**: Known for its simplicity and depth.

## Scripts
- `cnn_cifar10.ipynb`: Jupyter notebook containing the code for training and evaluating models on CIFAR-10.
- `cnn_cifar100.ipynb`: Jupyter notebook containing the code for training and evaluating models on CIFAR-100.

## Requirements
This project requires Python 3 and the following Python libraries installed:
- NumPy
- Matplotlib
- TensorFlow

## Usage
To use the trained models, load them using TensorFlow Keras API:
```python
from tensorflow.keras.models import load_model

# Example for CIFAR-10 AlexNet model
model = load_model('models/cifar10/AlexNet.h5')
```

To re-train or evaluate the models, run the Jupyter notebooks:
- For CIFAR-10: `scripts/cnn_cifar10.ipynb`
- For CIFAR-100: `scripts/cnn_cifar100.ipynb`

## Results
The results of the model classification can be visualized in the notebooks, including accuracy and loss metrics.

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
```

Ce `README.md` donne un aperçu général du projet, y compris la structure du projet, les datasets utilisés, une description des modèles, comment utiliser les modèles, et comment contribuer au projet. Vous pouvez l'ajuster en fonction de vos besoins spécifiques, ajouter des sections pour des instructions d'installation plus détaillées, des exemples de code supplémentaires, des résultats spécifiques de classification, ou toute autre information pertinente que vous souhaitez inclure.
