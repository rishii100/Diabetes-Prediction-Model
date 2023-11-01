# Dibetes-Prediction-Model

## Overview

This machine learning project aims to predict the risk of diabetes based on health-related features. The project includes the development of multiple classifiers, such as Logistic Regression, K Nearest Neighbors, Support Vector Classifier, Naive Bayes, Decision Tree, and Random Forest.

## Files

- **diabetes.csv:** Dataset containing health-related features and diabetes outcomes.
- **your_script.py:** Python script for model training, evaluation, and user input prediction.
- **knn_model.pkl:** Pickled file containing the trained K Nearest Neighbors model.
- **scaler.pkl:** Pickled file containing the MinMaxScaler used for feature scaling during training.

## Dependencies

- **Python (>=3.6)**
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

## Usage

1. Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Run the `your_script.py` script to train the models and perform evaluations.

```bash
jupyter notebook diabetes_prediction.ipynb
```

3. Make predictions on diabetes risk using the user input function.

```python
# Example usage in a Python script or interactive environment
from sklearn.externals import joblib

# Load the K Nearest Neighbors model and scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Use the user input function to make predictions
predict_diabetes_user_input(knn_model, scaler)
```

## Project Structure

- **data:** Directory containing datasets used in the project.
- **models:** Directory for storing trained machine learning models.
- **scripts:** Directory containing project scripts and utilities.

## Contributing

If you'd like to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m 'Add your feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Special thanks to contributors and open-source libraries that made this project possible.

Feel free to customize and expand upon this template based on the specifics of your project!
