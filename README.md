Netflix Content Classification using Random Forest & Decision Tree







Overview

This project demonstrates an end-to-end machine learning workflow for classifying Netflix titles as Movies or TV Shows. It includes data preprocessing, feature engineering, model training, evaluation, and comparison using Random Forest and Decision Tree classifiers.

The notebook is designed to showcase a practical classification pipeline using real-world Netflix metadata.

Objectives
Explore and understand the Netflix dataset.
Clean and preprocess the data.
Handle missing values effectively.
Perform feature engineering.
Encode categorical variables.
Train and compare multiple classification models.
Evaluate model performance using standard metrics.
Save the trained model for future predictions.
Dataset

The dataset contains information such as:

Title
Type
Director
Country
Date Added
Release Year
Rating
Duration
Genre
Description

The target variable is Type, where the model predicts whether a title is a Movie or a TV Show.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Pickle
Jupyter Notebook

Project Workflow

Import libraries
Load the dataset
Explore the data
Handle missing values
Perform feature engineering
Encode categorical features
Split the dataset
Train Random Forest model
Train Decision Tree model
Evaluate and compare both models
Save the best model
Evaluation Metrics

The models are evaluated using:
Accuracy Score
Confusion Matrix
Classification Report

These metrics help measure the overall performance of each classifier and compare their prediction quality.

Project Structure
Netflix-Content-Classification/
│
├── random_and_decision_using_netflix.ipynb
├── model.pkl
├── requirements.txt
├── README.md
└── dataset.csv
How to Run
Clone the repository
git clone https://github.com/your-username/Netflix-Content-Classification.git
Navigate to the project
cd Netflix-Content-Classification
Install dependencies
pip install -r requirements.txt
Launch Jupyter Notebook
jupyter notebook

Open the notebook and run all cells.

Future Improvements
Hyperparameter tuning
Cross-validation
Additional machine learning algorithms
Interactive Streamlit web application
Cloud deployment
Contributing

Contributions are welcome. Feel free to fork the repository, submit issues, or create pull requests to improve the project.

License

This project is created for educational and learning purposes.
