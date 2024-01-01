import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('Practical_Implementaion\LogisticRegression_TitanicDataset', 'rb'))


# Get user input for passenger details
pclass = int(input("Enter passenger class (1, 2, or 3): "))
sex = int(input("Enter sex (0 for female, 1 for male): "))
age = float(input("Enter age: "))
sibsp = int(input("Enter number of siblings/spouses aboard: "))
parch = int(input("Enter number of parents/children aboard: "))
fare = float(input("Enter passenger fare: "))

# Create a DataFrame from user inputs
input_data = {
    'pclass': [pclass],
    'sex': [sex],
    'age': [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare]
}

input_df = pd.DataFrame(input_data)

# Make predictions using the model
predicted_survival = model.predict(input_df)

# Display the predicted survival outcome
if predicted_survival[0] == 1:
    print("The passenger is predicted to have survived.")
else:
    print("The passenger is predicted to have not survived.")
