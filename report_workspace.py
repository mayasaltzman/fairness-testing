import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#read csv file
df = pd.read_csv('recruitment_data.csv')
#print(df.head)

#clean data by invalid experience 
exp_condition = df['Age'] - df['ExperienceYears'] >= 15
df_cleaned = df[exp_condition]

#scale data for mean of 0 and standard deviation 1
#only apply scaling to continuous variables not the target variable (hiring decision) or binary variables (ex. Gender)

ct = ColumnTransformer([
        ('num', StandardScaler(), ['Age','ExperienceYears','PreviousCompanies','DistanceFromCompany','InterviewScore','SkillScore','PersonalityScore'])
    ], remainder='passthrough').set_output(transform="pandas")

df_scaled = ct.fit_transform(df_cleaned)

X = df_scaled[['num__Age', 'num__ExperienceYears', 'num__PreviousCompanies',
       'num__DistanceFromCompany', 'num__InterviewScore', 'num__SkillScore',
       'num__PersonalityScore', 'remainder__Gender',
       'remainder__EducationLevel', 'remainder__RecruitmentStrategy']]
y = df_scaled['remainder__HiringDecision']

#split to 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, train_size=0.90, shuffle=True)

#creates random forest classifer 
#n_estimators=100 creates 100 trees
#trains on training data and predicts on testing data
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#check of accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

feature_importances = classifier.feature_importances_

# Use the feature names from X, not df_scaled
features = X.columns

# Create a DataFrame for better visualization
importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)

# Plot nicely
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.tight_layout()
plt.show()

# #plot feature importance
# feature_importances = classifier.feature_importances_

# plt.barh(df_scaled.columns, feature_importances)
# plt.xlabel('Feature Importance')
# plt.title('Feature Importance in Random Forest Classifier')
# plt.show()
