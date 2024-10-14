# Multi-classification Assessment of Bank Personal Credit Risk Based on Multi-source Information Fusion

This project implements the methods described in the paper:

**"Multi-classification assessment of bank personal credit risk based on multi-source information fusion"**

By Tianhui Wang, Renjing Liu, Guohua Qi

Reference: Expert Systems With Applications 191 (2022) 116236

## Author

Kurnia Cahya Febryanto

## Introduction

This project aims to replicate the MIFCA (Multi-source Information Fusion Credit Assessment) model as described in the referenced paper. The MIFCA model integrates the outputs of six different classifiers using Dempster-Shafer evidence theory to improve the accuracy and robustness of personal credit risk assessment.

## Workflow

The analysis workflow consists of the following steps:

1. **Data Loading and Preprocessing**
   - Load the dataset from `Dataset-Research.xlsx`.
   - Remove irrelevant or redundant variables.
   - Handle missing values by imputing numerical and categorical columns.
   - Handle outliers in the data.
   - Encode categorical variables into numerical ones.
   - Split the data into training and testing sets.
   - Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   - Standardize numerical features.

2. **Training Base Classifiers**
   - Train six base classifiers:
     - Decision Tree (DT)
     - Random Forest (RF)
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - BP Neural Network (implemented as Multi-Layer Perceptron)
     - XGBoost
   - Obtain predicted probabilities from each classifier.

3. **Dempster-Shafer Evidence Theory Fusion**
   - Define the `MassFunction` class to handle mass functions and combine them using Dempster's rule.
   - Prepare belief assignments by converting classifier probabilities into mass functions.
   - Combine mass functions from all classifiers for each instance.
   - Make final predictions based on the combined mass functions.

4. **Model Evaluation**
   - Evaluate the performance of each base classifier using metrics such as accuracy, precision, recall, and F1-score.
   - Evaluate the performance of the fused MIFCA model.
   - Plot confusion matrices for visual analysis.
   - Visualize feature importance from the Random Forest classifier.
   - Plot learning curves to assess model performance over varying training set sizes.

5. **Results and Analysis**
   - Compare the performance of the MIFCA model with individual classifiers.
   - Analyze the results to understand the benefits of multi-source information fusion.
   - Discuss the implications of the findings and potential areas for improvement.

6. **Conclusion**
   - Summarize the key findings of the project.
   - Highlight the effectiveness of the MIFCA model in improving credit risk assessment.
   - Provide insights into future work or enhancements.

## References

- Wang, T., Liu, R., & Qi, G. (2022). Multi-classification assessment of bank personal credit risk based on multi-source information fusion. *Expert Systems With Applications*, 191, 116236.

