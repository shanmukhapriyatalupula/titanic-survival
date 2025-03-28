1. Understand the Goal

The aim is to predict whether a passenger on the Titanic survived or not. This is a binary classification problem where:

1 = Survived

0 = Did Not Survive



---

2. Set Up Your Environment

First, ensure you have the necessary tools installed. You’ll need Python and a few libraries like pandas, numpy, scikit-learn, matplotlib, and seaborn. If they aren’t installed, run:

pip install pandas numpy matplotlib seaborn scikit-learn


---

3. Get the Data

The Titanic dataset is publicly available on Kaggle. Download the dataset—it usually contains two files:

train.csv – For training your model

test.csv – For making predictions



---

4. Explore the Data

Load the dataset and take a look at it.

import pandas as pd

df = pd.read_csv('train.csv')
print(df.head())

Check for:

Missing values

Data types (numbers, categories, etc.)

The target variable (Survived)



---

5. Clean the Data

Real-world data is messy, so we need to clean it.

1. Handle Missing Data:



Age: Replace missing ages with the median age.

Embarked (Port of boarding): Fill missing values with the most common port.

Cabin: Many are missing – either drop it or mark as “Unknown”.


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)  # Dropping the Cabin column

2. Drop Useless Columns:
Some columns like PassengerId, Name, and Ticket don’t add much value to prediction. We’ll remove them.



df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


---

6. Feature Engineering

We can create new, useful features:

1. Family Size: Combine SibSp (siblings/spouses) and Parch (parents/children).



df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

2. Title Extraction: Pull out titles (Mr., Mrs., etc.) from the passenger names.




---

7. Convert Categorical Data

Machine learning models can’t work directly with text, so we convert categories to numbers:

1. Gender: Convert male to 0 and female to 1.


2. Embarked: Use One-Hot Encoding to create separate columns for each port.



df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'])


---

8. Split the Data

We divide the dataset into two parts:

Training data (to teach the model)

Test data (to evaluate how well it learned)


from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)  # Features (everything except Survived)
y = df['Survived']               # Target (what we want to predict)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


---

9. Train the Model

Let’s use a Random Forest model—a strong baseline for classification problems.

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


---

10. Evaluate the Model

After training, let’s see how well the model performs.

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

Key metrics:

Accuracy: Overall correctness of predictions.

Precision: How many predicted survivors actually survived.

Recall: How well the model identifies actual survivors.

F1-Score: A balance between precision and recall.



---

11. Improve the Model

If accuracy is low, try:

Hyperparameter tuning: Adjust model settings.

Feature selection: Use only the most useful features.

Try other models: Logistic Regression, Gradient Boosting (XGBoost), etc.


