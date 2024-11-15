# Classify_That_Fruit

## Fruit Classification Using k-NN

This project demonstrates the use of the **k-Nearest Neighbors (k-NN)** algorithm for classifying fruits based on their physical properties such as mass, width, height, and color score. After evaluating multiple machine learning models, k-NN was chosen as the optimal classifier for this problem.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Project Overview
The project uses the **k-Nearest Neighbors (k-NN)** algorithm to classify fruit samples into four categories: **apple**, **mandarin**, **orange**, and **lemon**. The dataset includes fruit properties like mass, width, height, and color score.

## Technologies Used
- Python
- Jupyter Notebook
- Libraries: `numpy`, `matplotlib`, `scikit-learn`, `pandas`

## Features
- Train/test split for evaluating the k-NN model.
- Visualization of decision boundaries.
- Ability to adjust k-NN parameters like the number of neighbors and weight function.

## Dataset
The dataset consists of fruit samples with the following features:
- **Mass**: Weight of the fruit.
- **Width**: Width of the fruit.
- **Height**: Height of the fruit.
- **Color Score**: Numerical representation of the fruit's color.
- **Label**: Category of fruit (apple, mandarin, orange, or lemon).

## Model Selection
Initially, several classifiers were tested:

- **Logistic Regression**: This model showed poor performance on the test set, with an accuracy of only 42%.
- **Decision Tree Classifier**: The decision tree achieved perfect accuracy on the training set (1.0) but overfitted the data, resulting in poor test set performance (42% accuracy).
  
After evaluating these models, **k-NN** was selected as the optimal model due to its better generalization performance on the test set (83%) compared to the other models.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required libraries:
   ```bash
   pip install numpy matplotlib scikit-learn pandas
   ```
3. Open the Jupyter Notebook file:
   ```bash
   jupyter notebook ClassifyThatFruit.ipynb
   ```

## Usage
1. Run the cells in the notebook to load the dataset, preprocess the data, and split it into training and testing sets.
2. Visualize the decision boundaries by calling:
   ```python
   plot_fruit_knn(X_train, y_train, n_neighbors=5, weights='uniform')
   ```
3. Experiment with different `n_neighbors` and `weights` to observe their impact.

## Future Enhancements
- Add more features like fruit texture or ripeness.
- Experiment with other classification models for comparison.
- Deploy the model using a web framework (e.g., Flask, Django).

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
