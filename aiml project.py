import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# create file
data_dict = {
    "hours": [1,2,3,4,5,6,7,8],
    "attendance": [50,60,65,70,75,80,85,90],
    "previous_marks": [40,45,50,55,60,65,70,80],
    "result": ["fail","fail","fail","pass","pass","pass","pass","pass"]
}

df = pd.DataFrame(data_dict)
df.to_csv("student.csv", index=False)

# read file
data = pd.read_csv("student.csv")
print(data)

# Features & Target
X = data[['hours', 'attendance', 'previous_marks']]
y = data['result']

# Convert result (pass/fail → 1/0)
y = y.map({'fail': 0, 'pass': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Prediction
sample = pd.DataFrame([[4, 75, 60]], columns=['hours','attendance','previous_marks'])
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Pass")
else:
    print("Fail")

plt.scatter(data['attendance'], data['previous_marks'])
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.title("Attendance vs Marks")
plt.show()