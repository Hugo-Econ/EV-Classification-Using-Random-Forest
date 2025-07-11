# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:06:02 2025

@author: Hugo
"""


# %% Model 2017 and Import 2011-2016 datas
from sklearn.preprocessing import LabelEncoder
data_2017 = data_frames.get(2017)


data_2017 = data_2017[data_2017['CLAS'].isin(['PAU', 'CAU'])]

#Mapping of brand
data_2017['MARQ_VEH'] = data_2017['MARQ_VEH'].map(brand_mapping).fillna(data_2017['MARQ_VEH'])

data_2017['Marq_Model'] = data_2017['MARQ_VEH'] + ' ' + data_2017['MODEL_VEH']

data_2011_2016 = data_frames.get(2011)

years_to_import = range(2012, 2017)
for year in years_to_import:
    data_year = data_frames.get(year)
    data_2011_2016 = pd.concat([data_2011_2016, data_year], ignore_index=True)

data_2011_2016 = data_2011_2016[data_2011_2016['CLAS'].isin(['PAU', 'CAU'])]

#Mapping of brand
data_2011_2016['MARQ_VEH'] = data_2011_2016['MARQ_VEH'].map(brand_mapping).fillna(data_2011_2016['MARQ_VEH'])

data_2011_2016['Marq_Model'] = data_2011_2016['MARQ_VEH'] + ' ' + data_2011_2016['MODEL_VEH']
data_2011_2016.columns
data_2011_2016['AN'].unique()

# Labels
# Combine the unique model names from both datasets
all_models = list(data_2017['MODEL_VEH'].unique()) + list(data_2011_2016['MODEL_VEH'].unique())
unique_models = list(set(all_models))

# Create and fit the LabelEncoder
label_encoder_model = LabelEncoder()
label_encoder_model.fit(unique_models)

all_brand = list(data_2017['MARQ_VEH'].unique()) + list(data_2011_2016['MARQ_VEH'].unique())
unique_brand = list(set(all_brand))

# Create and fit the LabelEncoder
label_encoder_brand = LabelEncoder()
label_encoder_brand.fit(unique_brand)


all_brand_models = list(data_2017['Marq_Model'].unique()) + list(data_2011_2016['Marq_Model'].unique())
unique_brand_model = list(set(all_brand_models))

# Create and fit the LabelEncoder
label_encoder_brand_model = LabelEncoder()
label_encoder_brand_model.fit(unique_brand_model)

# %% Prep Datas
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
#NAs to 0.
data_2017['CYL_VEH'] = data_2017['CYL_VEH'].fillna(0).astype(int)
data_2017['CYL_VEH'].value_counts()

data_2017['NB_CYL'] = data_2017['NB_CYL'].fillna(0).astype(int)
data_2017['NB_CYL'].value_counts()

data_2017['NB_ESIEU_MAX'] = data_2017['NB_ESIEU_MAX'].fillna(0).astype(int)
data_2017['NB_ESIEU_MAX'].value_counts()

#Label encoding. Convert categorical variables (MODEL_VEH, MARQ_VEH) to numerical values.
data_2017_AI = data_2017

data_2017_AI['MODEL_VEH'] = label_encoder_model.transform(data_2017_AI['MODEL_VEH'])

data_2017_AI['MARQ_VEH'] = label_encoder_brand.transform(data_2017_AI['MARQ_VEH'])

data_2017_AI['Marq_Model'] = label_encoder_brand_model.transform(data_2017_AI['Marq_Model'])

# Features and target variable
#L== Electric, W== plug-in hybrid, H=hybrid. 
data_2017_AI['TYP_CARBU'].value_counts()

# Define the mapping for relabeling
relabel_mapping = {
    'P': 'OTHER',
    'N': 'OTHER',
    'A': 'OTHER',
    'S': 'OTHER',
    'nan': 'OTHER',  # Assuming 'nan' is a string representation of NaN
    'C': 'OTHER'
}

# Replace the specified values with 'OTHER'
data_2017_AI['TYP_CARBU'] = data_2017_AI['TYP_CARBU'].replace(relabel_mapping)

# If there are actual NaN values, also replace them with 'OTHER'
data_2017_AI['TYP_CARBU'] = data_2017_AI['TYP_CARBU'].replace(np.nan, 'OTHER')

#Works fine!
data_2017_AI['TYP_CARBU'].value_counts()


# Ensure the target variable is string type
data_2017_AI['TYP_CARBU'] = data_2017_AI['TYP_CARBU'].astype(str)


features = ['MODEL_VEH', 'MARQ_VEH','Marq_Model', 'NB_CYL', 'CYL_VEH', 'NB_ESIEU_MAX','ANNEE_MOD']
X = data_2017_AI[features]
y = data_2017_AI['TYP_CARBU']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% Random Forest: model 2017.
import time
# Start chrono for relabeling
start_time_rf = time.time()

# w and l underrepresented
class_weights = {'D': 1, 'E': 1, 'H': 1, 'L': 2, 'OTHER': 1, 'W': 2}

# Initialize the classifier with parallel processing, fewer estimators for speed, and class weights
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1, class_weight=class_weights)

#Heavier weight on L and W.

# Cross-validation with parallel processing
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
end_time_rf = time.time()
duration_rf = end_time_rf - start_time_rf
print(duration_rf/60)

# Fit the Random Forest classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the training set
y_train_pred = rf_classifier.predict(X_train)

# Predict on the test set
y_test_pred = rf_classifier.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

#If train_accuracy is much higher than test_accuracy, it might indicate overfitting.
# Not the case, good news ! 99%.
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Print classification report for the test set
Table_to_Maximise_F1=classification_report(y_test, y_test_pred)
print("Classification Report on Test Set:")
print(Table_to_Maximise_F1)
# I REALLY care about W and L
##Precision tells you how many of the items that the model labeled as positive are actually positive
# Formula: Precision = TP / (TP + FP)

## Recall tells you how many of the actual positive items were correctly labeled by the model.
# Recall = TP / (TP + FN)
# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## F1-Score is the balance between precision and recall. It gives a single metric that considers both precision and recall.

# Cross-validation scores already computed
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

end_time_rf_all = time.time()
duration_rf_all = end_time_rf_all - start_time_rf
print(duration_rf_all/60)

# %% Generalize 2011-2016.
start_time_rf_11_16 = time.time()
#Import
data_2011_2016['CYL_VEH'] = data_2011_2016['CYL_VEH'].fillna(0).astype(int)
data_2011_2016['CYL_VEH'].value_counts()

data_2011_2016['NB_CYL'] = data_2011_2016['NB_CYL'].fillna(0).astype(int)
data_2011_2016['NB_CYL'].value_counts()

data_2011_2016['NB_ESIEU_MAX'] = data_2011_2016['NB_ESIEU_MAX'].fillna(0).astype(int)
data_2011_2016['NB_ESIEU_MAX'].value_counts()

#Label encoding. Convert categorical variables (MODEL_VEH, MARQ_VEH) to numerical values.
data_2011_2016_AI = data_2011_2016


data_2011_2016_AI['MODEL_VEH'] = label_encoder_model.transform(data_2011_2016_AI['MODEL_VEH'])

data_2011_2016_AI['MARQ_VEH'] = label_encoder_brand.transform(data_2011_2016_AI['MARQ_VEH'])

data_2011_2016_AI['Marq_Model'] = label_encoder_brand_model.transform(data_2011_2016_AI['Marq_Model'])



X_2011_2016 = data_2011_2016_AI[features]

# Predict on the training set
#from sklearn.impute import SimpleImputer

# Impute missing values with the mean
#imputer = SimpleImputer(strategy='mean')
#X_2011_2016_imputed = imputer.fit_transform(X_2011_2016)

# Now use the imputed data for prediction


y_pred_2011_2016 = rf_classifier.predict(X_2011_2016_imputed)
data_2011_2016_AI['pred_TYP_CARBU']= y_pred_2011_2016
end_time_rf_all_11_16 = time.time()
duration_rf_all_11_16 = end_time_rf_all_11_16 - start_time_rf_11_16
print(duration_rf_all_11_16/60)

# %% Verification of 2011-2015 with real world datas.
#Look at agregare EV fleet. 11k in 2011. 

#Sounds about right.
abs_pred_TYP_CARBU = data_2011_2016_AI.groupby(['AN', 'pred_TYP_CARBU']).size().unstack(fill_value=0)
abs_pred_TYP_CARBU_2017 = data_2017.groupby(['AN', 'TYP_CARBU']).size().unstack(fill_value=0)
abs_pred_TYP_CARBU_2018 = data_2018.groupby(['AN', 'TYP_CARBU']).size().unstack(fill_value=0)

# Calculate the total number of vehicles per year
total_per_year = data_2011_2016_AI.groupby('AN').size()

# Calculate the percentage of pred_TYP_CARBU to the whole fleet
percentage_pred_TYP_CARBU = abs_pred_TYP_CARBU.div(total_per_year, axis=0)

# Display the percentage data
print(percentage_pred_TYP_CARBU)

# Sum columns 'L' and 'W' to create 'ZEV'
abs_pred_TYP_CARBU['ZEV'] = abs_pred_TYP_CARBU['L'] + abs_pred_TYP_CARBU['W']
abs_pred_TYP_CARBU_2017['ZEV'] = abs_pred_TYP_CARBU_2017['L'] + abs_pred_TYP_CARBU_2017['W']
abs_pred_TYP_CARBU_2018['ZEV'] = abs_pred_TYP_CARBU_2018['L'] + abs_pred_TYP_CARBU_2018['W']

zev_data = abs_pred_TYP_CARBU[['ZEV']].reset_index()

zev_data_2017 = abs_pred_TYP_CARBU_2017[['ZEV']].reset_index()
zev_data_2018 = abs_pred_TYP_CARBU_2018[['ZEV']].reset_index()

zev_data= pd.concat([zev_data, zev_data_2017], ignore_index=True)
zev_data=pd.concat([zev_data, zev_data_2018], ignore_index=True)

colors = ['blue'] * len(zev_data)
colors[zev_data[zev_data['AN'] == 2017].index[0]] = 'green'
colors[zev_data[zev_data['AN'] == 2018].index[0]] = 'green'

# Plot the graph
ax = zev_data.plot(x='AN', y='ZEV', kind='bar', color=colors, stacked=True, figsize=(10, 7))
plt.title('Percentage of ZEV per Year')
plt.xlabel('Year')
plt.ylabel('Percentage of Fleet')
plt.legend(title='pred_TYP_CARBU')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save datas.
# Decode the labels back to original values
data_2011_2016_AI['MODEL_VEH'] = label_encoder_model.inverse_transform(data_2011_2016_AI['MODEL_VEH'])
data_2011_2016_AI['MARQ_VEH'] = label_encoder_brand.inverse_transform(data_2011_2016_AI['MARQ_VEH'])
data_2011_2016_AI['Marq_Model'] = label_encoder_brand_model.inverse_transform(data_2011_2016_AI['Marq_Model'])
data_2011_2016_AI

file_path = os.path.join(base_path, f'Data\\raw\\Vehicule_En_Circulation\\data_2011_2016_AI.csv')

# Save the data to the file path
data_2011_2016_AI.to_csv(file_path, index=False)


# %% Random forest: Train 2018 and fit on 2017: GOOD RESULT. (So likely to be true on 2011-2016.)
data_2018 = data_frames.get(2018)
data_2018 = data_2018[data_2018['CLAS'].isin(['PAU', 'CAU'])]

#Mapping of brand
data_2018['MARQ_VEH'] = data_2018['MARQ_VEH'].map(brand_mapping).fillna(data_2018['MARQ_VEH'])

data_2018['Marq_Model'] = data_2018['MARQ_VEH'] + ' ' + data_2018['MODEL_VEH']

#NAs to 0.
data_2018['CYL_VEH'] = data_2018['CYL_VEH'].fillna(0).astype(int)
data_2018['CYL_VEH'].value_counts()

data_2018['NB_CYL'] = data_2018['NB_CYL'].fillna(0).astype(int)
data_2018['NB_CYL'].value_counts()

data_2018['NB_ESIEU_MAX'] = data_2018['NB_ESIEU_MAX'].fillna(0).astype(int)
data_2018['NB_ESIEU_MAX'].value_counts()

#Label encoding. Convert categorical variables (MODEL_VEH, MARQ_VEH) to numerical values.
# Labels
# Combine the unique model names from both datasets
all_models = list(data_2017['MODEL_VEH'].unique()) + list(data_2018['MODEL_VEH'].unique())
unique_models = list(set(all_models))

# Create and fit the LabelEncoder
label_encoder_model = LabelEncoder()
label_encoder_model.fit(unique_models)

all_brand = list(data_2017['MARQ_VEH'].unique()) + list(data_2018['MARQ_VEH'].unique())
unique_brand = list(set(all_brand))

# Create and fit the LabelEncoder
label_encoder_brand = LabelEncoder()
label_encoder_brand.fit(unique_brand)


all_brand_models = list(data_2017['Marq_Model'].unique()) + list(data_2018['Marq_Model'].unique())
unique_brand_model = list(set(all_brand_models))

# Create and fit the LabelEncoder
label_encoder_brand_model = LabelEncoder()
label_encoder_brand_model.fit(unique_brand_model)

#Models-brand to numbers
data_2018_AI = data_2018

data_2018_AI['MODEL_VEH'] = label_encoder_model.transform(data_2018_AI['MODEL_VEH'])

data_2018_AI['MARQ_VEH'] = label_encoder_brand.transform(data_2018_AI['MARQ_VEH'])

data_2018_AI['Marq_Model'] = label_encoder_brand_model.transform(data_2018_AI['Marq_Model'])

# Features and target variable
data_2018_AI['TYP_CARBU'].value_counts()

# Define the mapping for relabeling
relabel_mapping = {
    'P': 'OTHER',
    'N': 'OTHER',
    'A': 'OTHER',
    'S': 'OTHER',
    'nan': 'OTHER',  # Assuming 'nan' is a string representation of NaN
    'C': 'OTHER'
}

# Replace the specified values with 'OTHER'
data_2018_AI['TYP_CARBU'] = data_2018_AI['TYP_CARBU'].replace(relabel_mapping)

# If there are actual NaN values, also replace them with 'OTHER'
data_2018_AI['TYP_CARBU'] = data_2018_AI['TYP_CARBU'].replace(np.nan, 'OTHER')

#Works fine!
data_2018_AI['TYP_CARBU'].value_counts()


# Ensure the target variable is string type
data_2018_AI['TYP_CARBU'] = data_2018_AI['TYP_CARBU'].astype(str)


features = ['MODEL_VEH', 'MARQ_VEH','Marq_Model', 'NB_CYL', 'CYL_VEH', 'NB_ESIEU_MAX','ANNEE_MOD']
X = data_2018_AI[features]
y = data_2018_AI['TYP_CARBU']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)




# Random Forest
import time
# Start chrono for relabeling
start_time_rf = time.time()

# w and l underrepresented
class_weights = {'D': 1, 'E': 1, 'H': 1, 'L': 2, 'OTHER': 1, 'W': 2}

# Initialize the classifier with parallel processing, fewer estimators for speed, and class weights
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1, class_weight=class_weights)

#Heavier weight on L and W.

# Cross-validation with parallel processing
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
end_time_rf = time.time()
duration_rf = end_time_rf - start_time_rf
print(duration_rf/60)

# Fit the Random Forest classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the training set
y_train_pred = rf_classifier.predict(X_train)

# Predict on the test set
y_test_pred = rf_classifier.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

#If train_accuracy is much higher than test_accuracy, it might indicate overfitting.
# Not the case, good news ! 99%.
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Print classification report for the test set
Table_to_Maximise_F1=classification_report(y_test, y_test_pred)
print("Classification Report on Test Set:")
print(Table_to_Maximise_F1)
# I REALLY care about W and L
##Precision tells you how many of the items that the model labeled as positive are actually positive
# Formula: Precision = TP / (TP + FP)

## Recall tells you how many of the actual positive items were correctly labeled by the model.
# Recall = TP / (TP + FN)
# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## F1-Score is the balance between precision and recall. It gives a single metric that considers both precision and recall.

# Cross-validation scores already computed
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

end_time_rf_all = time.time()
duration_rf_all = end_time_rf_all - start_time_rf
print(duration_rf_all/60)


## Test on 2017 

data_2017 = data_frames.get(2017)
data_2017 = data_2017[data_2017['CLAS'].isin(['PAU', 'CAU'])]

#Mapping of brand
data_2017['MARQ_VEH'] = data_2017['MARQ_VEH'].map(brand_mapping).fillna(data_2017['MARQ_VEH'])

data_2017['Marq_Model'] = data_2017['MARQ_VEH'] + ' ' + data_2017['MODEL_VEH']

#Import
data_2017['CYL_VEH'] = data_2017['CYL_VEH'].fillna(0).astype(int)
data_2017['CYL_VEH'].value_counts()

data_2017['NB_CYL'] = data_2017['NB_CYL'].fillna(0).astype(int)
data_2017['NB_CYL'].value_counts()

data_2017['NB_ESIEU_MAX'] = data_2017['NB_ESIEU_MAX'].fillna(0).astype(int)
data_2017['NB_ESIEU_MAX'].value_counts()

#Label encoding. Convert categorical variables (MODEL_VEH, MARQ_VEH) to numerical values.
data_2017_AI = data_2017

data_2017_AI['TYP_CARBU'] = np.where(data_2017_AI['TYP_CARBU'] == 'Plug-in Hybrid', 'W', data_2017_AI['TYP_CARBU'])
data_2017_AI['TYP_CARBU'] = np.where(data_2017_AI['TYP_CARBU'] == 'Hybrid', 'H', data_2017_AI['TYP_CARBU'])
data_2017_AI['TYP_CARBU'] = np.where(data_2017_AI['TYP_CARBU'] == 'Electric', 'L', data_2017_AI['TYP_CARBU'])

data_2017_AI['TYP_CARBU'].value_counts()


data_2017_AI['MODEL_VEH'] = label_encoder_model.transform(data_2017_AI['MODEL_VEH'])

data_2017_AI['MARQ_VEH'] = label_encoder_brand.transform(data_2017_AI['MARQ_VEH'])

data_2017_AI['Marq_Model'] = label_encoder_brand_model.transform(data_2017_AI['Marq_Model'])

# Features and target variable
#L== Electric, W== plug-in hybrid, H=hybrid. 
data_2017_AI['TYP_CARBU'].unique()
data_2017_AI['TYP_CARBU'].value_counts()

# Define the mapping for relabeling
relabel_mapping = {
    'P': 'OTHER',
    'N': 'OTHER',
    'A': 'OTHER',
    'S': 'OTHER',
    'nan': 'OTHER',  # Assuming 'nan' is a string representation of NaN
    'C': 'OTHER'
}

# Replace the specified values with 'OTHER'
data_2017_AI['TYP_CARBU'] = data_2017_AI['TYP_CARBU'].replace(relabel_mapping)

# If there are actual NaN values, also replace them with 'OTHER'
data_2017_AI['TYP_CARBU'] = data_2017_AI['TYP_CARBU'].replace(np.nan, 'OTHER')

#Works fine!
data_2017_AI['TYP_CARBU'].value_counts()

# Ensure the target variable is string type
data_2017_AI['TYP_CARBU'] = data_2017_AI['TYP_CARBU'].astype(str)


X_2017 = data_2017_AI[features]
y_2017 = data_2017['TYP_CARBU']

X_train_2017 , X_test_2017 , y_train_2017 , y_test_2017 = train_test_split(X_2017, y_2017, test_size=0.5, random_state=42)

# Predict on the training set
y_train_pred_2017 = rf_classifier.predict(X_train_2017)

# Predict on the test set
y_test_pred_2017 = rf_classifier.predict(X_test_2017)

# Evaluate the model
train_accuracy_2017 = accuracy_score(y_train_2017, y_train_pred_2017)
test_accuracy_2017 = accuracy_score(y_test_2017, y_test_pred_2017)

#If train_accuracy is much higher than test_accuracy, it might indicate overfitting.
# Not the case, good news ! 99%.
print("Training Accuracy:", train_accuracy_2017)
print("Test Accuracy:", test_accuracy_2017)

# Print classification report for the test set
Table_to_Maximise_F1_2017 = classification_report(y_test_2017, y_test_pred_2017)
print("Classification Report on Test Set:")
print(Table_to_Maximise_F1_2017)
# I REALLY care about W and L
##Precision tells you how many of the items that the model labeled as positive are actually positive
# Formula: Precision = TP / (TP + FP)

## Recall tells you how many of the actual positive items were correctly labeled by the model.
# Recall = TP / (TP + FN)
# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## F1-Score is the balance between precision and recall. It gives a single metric that considers both precision and recall.

# Cross-validation scores already computed
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))



