'''
LOAN DATASET
'''

# Required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
data = pd.read_csv('loan.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

# Label encode the target variable
encode = LabelEncoder()
data.Loan_Status = encode.fit_transform(data.Loan_Status)

# Drop the null values
data.dropna(how='any', inplace=True)

# Train-test-split
train, test = train_test_split(data, test_size=0.2, random_state=0)

# Separate the target and independent variable
train_x = train.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
train_y = train['Loan_Status']

test_x = test.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
test_y = test['Loan_Status']

# Encode the data (handle categorical variables)
train_x = pd.get_dummies(train_x)
test_x  = pd.get_dummies(test_x)

print('Shape of training data : ', train_x.shape)
print('Shape of testing data : ', test_x.shape)

# Create the object of the model
model = LogisticRegression()

model.fit(train_x, train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data', predict)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y, predict))

# -------- Bar Graph: Loan Status Based on Education --------
plt.figure(figsize=(8, 6))
sns.countplot(x='Education', hue='Loan_Status', data=data, palette='Set1')
plt.title('Loan Status Based on Education')
plt.xlabel('Education')
plt.ylabel('Count')
plt.legend(title='Loan Status', loc='upper right', labels=['Rejected', 'Approved'])
plt.grid(True)
plt.show()

# -------- Box Plot: Loan Amount Distribution Based on Education --------
plt.figure(figsize=(8, 6))
sns.boxplot(x='Education', y='LoanAmount', data=data, palette='Set2')
plt.title('Loan Amount Distribution Based on Education')
plt.xlabel('Education')
plt.ylabel('Loan Amount')
plt.grid(True)
plt.show()

# -------- Bar Graph 2: Mean LoanAmount by Property Area --------
mean_loan_by_area = data.groupby('Property_Area')['LoanAmount'].mean()

plt.figure(figsize=(8, 6))
mean_loan_by_area.plot(kind='bar', color='lightgreen')
plt.title('Mean Loan Amount by Property Area')
plt.xlabel('Property Area')
plt.ylabel('Mean Loan Amount')
plt.grid(True)
plt.show()

# -------- Bar Graph 1: Distribution of Loan Status --------
loan_status_counts = data['Loan_Status'].value_counts()
labels = ['Rejected', 'Approved']

plt.figure(figsize=(8, 6))
plt.bar(labels, loan_status_counts, color=['skyblue', 'salmon'])
plt.title('Distribution of Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()
