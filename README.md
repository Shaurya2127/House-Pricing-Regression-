# 🏡 House Price Prediction Project
# 📚 Problem Statement
Predict the final selling prices of residential homes based on various features like area, number of bathrooms, location, and other house attributes.
The goal is to build a high-performing regression model that generalizes well to unseen data.

# 🛠️ Project Workflow
Data Cleaning
Handled missing values
Filled numeric columns with median values
Feature Engineering
Created new features:
TotalBathrooms
TotalSF
HouseAge
RemodelAge
IsNewHouse
One-hot encoding for categorical variables
Model Building
Tried multiple models:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest
K-Nearest Neighbors
Support Vector Regressor
Gradient Boosting
XGBoost
Model Evaluation
Compared models using RMSE (Root Mean Squared Error)
Cross-validated using 5-Fold CV
Hyperparameter Tuning
Tuned Gradient Boosting and XGBoost using GridSearchCV
Selected the best-performing model based on cross-validation RMSE
Final Model
Retrained the best model on full training data
Saved the trained model using joblib
Predictions
Predicted on new unseen test data
Saved results into a CSV file house_price_predictions.csv

### 📊 Libraries Used
Python
Pandas
Numpy
Scikit-learn
XGBoost
Seaborn
Matplotlib
TQDM
Joblib

## 🧠 Best Model Results
Model	CV RMSE Score
Gradient Boosting-	25724.71741377381
XGBoost-	25461.998046875



# ✅ Final Selected Model: Gradient Boosting or XGBoost (based on your result)

## 🚀 How to Run
Clone this repository
Install required libraries:
in bash:
pip install -r requirements.txt
Run the House_Price_Prediction.ipynb notebook step-by-step

To predict on new data, load the trained model:

python
import joblib
model = joblib.load('house_price_gradient_boosting_model.pkl') # or xgboost
predictions = model.predict(new_data)

### 📦 File Structure
css
Copy code
├── train.csv
├── test.csv
├── House_Price_Prediction.ipynb
├── house_price_predictions.csv
├── house_price_[your_model]_model.pkl
├── README.md
## ✨ Project Screenshots
![image](https://github.com/user-attachments/assets/5c752730-746b-45df-bff5-adba93ae3a8c)

![image](https://github.com/user-attachments/assets/381c83cc-e21a-4167-8306-b1b6872c97ce)

