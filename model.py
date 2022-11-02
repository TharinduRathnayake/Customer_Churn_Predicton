import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("cleaned_churn_dataset.csv")

print(df.head())

# Select independent and dependent variable
X = df[["region_category", "age", "points_in_wallet", "feedback", "gender", "used_special_discount", "past_complaint", "complaint_status"]]
y = df["churn_risk_score"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = DecisionTreeClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))