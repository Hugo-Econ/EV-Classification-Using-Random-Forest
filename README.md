🔋 EV Type Inference from Vehicle Registration Data (2011–2018)
Classifying Electric, Hybrid, and Plug-in Hybrid Vehicles Before Labels Existed Using Machine Learning

🧠 Project Overview
Many vehicle registration datasets prior to 2017 in Ontario do not explicitly label vehicle energy type (i.e., Electric, Plug-in Hybrid, Hybrid, or Internal Combustion). This project aims to retroactively classify EVs in the 2011–2016 datasets by training a machine learning model on labeled data from 2017 and 2018.

The resulting model allows us to:

Identify zero-emission vehicle (ZEV) trends over time

Quantify historical growth of EVs by type

Support policy and infrastructure planning with historical EV penetration estimates

⚙️ Methodology
Data Cleaning & Normalization

Standardize make/model using custom mappings

Fill missing values for engine specs (cylinders, electric capacity)

Label encode categorical features

Training on 2017 and 2018 Data

Use vehicle features to predict fuel type (TYP_CARBU)

Handle class imbalance (EVs are rare) with class_weight in Random Forest

Model Evaluation

Accuracy, F1-scores, and cross-validation

Special focus on W = Plug-in Hybrid, L = Electric classes

Apply Model to 2011–2016

Predict fuel type labels for historical records

Aggregate and visualize inferred EV adoption

📊 Features Used
MODEL_VEH — vehicle model

MARQ_VEH — vehicle make (brand)

NB_CYL, CYL_VEH — cylinder data

NB_ESIEU_MAX — number of electric motors (proxy for hybrid/EV)

ANNEE_MOD — model year

🔍 Results & Outputs
Inferred EV labels for 2011–2016 vehicles

ZEV (W + L) share plotted over time

Exported labeled dataset as CSV

💡 Why It Matters
Accurately understanding past EV penetration helps:

Quantify infrastructure needs

Evaluate effectiveness of past policy

Benchmark future EV adoption scenarios
