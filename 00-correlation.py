import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
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


#correlaion
#correlations sorted decreasing between all their unique pairs of values
corr_matrix = df.corr()
corr_df = pd.DataFrame(corr_matrix.stack(), columns=['Correlation Coefficient'])
corr_df.index.names = ['Feature 1', 'Feature 2']
corr_df = corr_df.reset_index().sort_values('Correlation Coefficient', ascending=False)
corr_df['Features'] = corr_df[['Feature 1', 'Feature 2']].apply(lambda x: frozenset(x), axis=1)
corr_df = corr_df[(corr_df['Feature 1'] != corr_df['Feature 2']) & (corr_df['Feature 1'] != corr_df['Feature 2'])]
corr_df = corr_df.drop_duplicates(subset=['Features'])
corr_df = corr_df[['Correlation Coefficient', 'Feature 1', 'Feature 2']].reset_index(drop=True)
print(corr_df)


#Barchart with correlation between different pairs of features
# Create bar plot using Seaborn with swapped x and y axes
plt.figure(figsize=(10, 8))
sns.barplot(x=corr_df['Correlation Coefficient'],
            y=[f"{feat1} & {feat2}" for feat1, feat2 in zip(corr_df['Feature 1'], corr_df['Feature 2'])],
            color='#ADD8E6')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature Pairs')
plt.title('Correlation Between Features')
plt.tight_layout()
plt.show()



print(df.head())