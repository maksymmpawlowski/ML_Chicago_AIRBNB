#Import Modules
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


#Import file
file = pd.read_csv("availabilities.csv")
df = pd.DataFrame(file)


#Prepare Data Frame
# remove useless
df = df.drop(['index'], axis=1)

# change binary to numeric
df['available'].replace({'t': 1, 'f': 0}, inplace=True)

# remove dollar sign and convert to float
df['price'] = df['price'].str.replace('[^0-9.]', '', regex=True).astype(float)
df['price'] = df['price'].astype(int)

# remake the date column
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')

# Extract the month, day, and year from the 'date' column
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['year'] = df['date'].dt.year


#Build the model
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Define the independent and dependent variables
X = df[['month', 'day', 'year', 'price', 'minimum_nights', 'maximum_nights']]
y = df['available']

#Crossvalidation
# Define the TimeSeriesSplit 
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)


# Create empty lists to store the metric values
results = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    # Append the results to the list
    results.append({
        'dependent_variables': 'available',
        'independent_variables': ['month', 'day', 'year', 'price', 'minimum_nights', 'maximum_nights'],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'specificity': specificity,
        'npv': npv
    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Calculate the mean metric values across all folds
mean_metrics = results_df.mean()

# Print the mean metric values
print(mean_metrics)


#Visualisation of the result
# Plot the ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes')
plt.legend(loc="lower right")
plt.show()