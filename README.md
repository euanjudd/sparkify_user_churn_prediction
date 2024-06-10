# Sparkify User Churn Prediction Model

This project develops a machine learning model to accurately predict churn among users of Sparkify, a music streaming platform. By analysing patterns in user activity and interaction data, the model identifies patterns that signal potential churn. This approach aims to help Sparkify deploy retention offers, increasing revenue by retaining valuable customers while minimising costs associated with misclassifying satisfied users. The insights gained can also be used to guide improvements in platform features to enhance overall user satisfaction and reduce churn rates.

## Baseline Model

The baseline model always predicts the majority class (non-churn) as the churn class is much less common than the non-churn class. 
- AUC: 0.5
- F1 Score: 0.9503
An AUC of 0.5 means it performs no better than randomly guessing the label. However, the high F1 Score suggests that the metric is skewed by the imbalanced dataset, where the vast majority of predictions are correct because most users do not churn. We will use this baseline as a reference point for evaluating the machine learning models. Clearly improving AUC will be the key performance indicator going forward.

## Model Selection

Below is a comparison of several machine learning algorithms performance based on the Average Area Under the Curve (AUC) and Average F1 Score. AUC is particularly useful for this unbalanced dataset as we saw in the baseline model. A high AUC means the model does a good job distinguishing between churn and non-churn users, even if churned users are rare. Here, the default Logistic Regression seems sufficient to distinguish between churning and non-churning users suggesting the relationship between the features and churn label could be linear. We used cross-validation to ensure the model generalised across the entire dataset rather than just perform well on a specific subset of the data.

| Model                    | Average AUC        | Average F1 Score   |
|--------------------------|--------------------|--------------------|
| Logistic Regression      | 0.8569 ± 0.1248    | 0.9867 ± 0.0015    |
| Random Forest Classifier | 0.6926 ± 0.1128    | 0.9598 ± 0.0078    |
| Gradient Boosted Trees   | 0.5448 ± 0.1354    | 0.9661 ± 0.0088    |

## Hyperparameter Tuning

We systematically varied the hyperparameters to determine which combination had the best performance. Regularization strength (regParam), the mix between L1 and L2 regularization (elasticNetParam), and the number of iterations (maxIter) were tested. The default settings (regParam=0.0, elasticNetParam=0.0, maxIter=100) had the highest average AUC and F1 Score, i.e. minimal regularization was required to achieve a good performance for this dataset.

| regParam | elasticNetParam | maxIter |
|----------|-----------------|---------|
| 0.0      | 0.0             | 50      |
| 0.01     | 50              | 100     |
| 0.1      |                 | 200     |

## Final Model
The final model was trained on the full training dataset. The performance metrics at the default classification threshold can be seen below. The confusion matrix shows the model predicted 336 non-churning users correctly and correctly identified 9 users as likely to churn. However, there were 3 false positives (non-churners incorrectly labeled as churners) and 5 false negatives (churners incorrectly labeled as non-churners). We will adjust the classification threshold in the next section to reduce the number of false negatives, potentially capturing more true churners at the expense of increasing false positives. 

| Metric     | Value   |
|------------|---------|
| AUC        | 0.9008  |
| F1 Score   | 0.9765  |
| Precision  | 0.9760  |
| Recall     | 0.9773  |

Confusion Matrix
+-----+---+---+
|churn|0.0|1.0|
+-----+---+---+
|    0|336|  3| 
|    1|  5|  9|
+-----+---+---+

### Feature Importances

| Feature                    | Importance         |
|----------------------------|--------------------|
| 307                        | 106.0409           |
| Thumbs Up                  | -62.2967           |
| Add Friend                 | -25.5407           |
| Thumbs Down                | -15.5114           |
| Logout                     | -15.1594           |
| 200                        | 13.8928            |
| avg_songs_per_session      | -10.4434           |
| avg_session_length         | 9.3805             |
| deviation_from_avg_length  | -8.9789            |
| length                     | -4.8372            |
| deviation_from_avg_songs   | 4.0845             |
| itemInSession              | -3.8239            |
| Save Settings              | -3.7290            |
| Home                       | -2.0182            |
| Submit Downgrade           | -1.5569            |
| Submit Upgrade             | -1.5043            |
| Help                       | -1.2636            |
| Upgrade                    | -0.8712            |
| Settings                   | 0.5643             |
| normalized_length          | 0.5042             |
| stddev_session_interval    | 0.4554             |
| genderVec                  | -0.4268            |
| levelVec                   | 0.4268             |
| userAgentVec               | 0.3224             |
| total_sessions             | 0.2589             |
| tenure                     | -0.2135            |
| numberOfSongs              | -0.2072            |
| NextSong                   | -0.2072            |
| Roll Advert                | -0.0855            |
| activity_recency           | -0.0850            |
| avg_session_interval       | 0.0840             |
| Add to Playlist            | -0.0815            |
| Downgrade                  | 0.0759             |
| Error                      | -0.0574            |
| 404                        | -0.0574            |
| About                      | 0.0211             |

- The HTTP code for temporary redirects (307) is a strong predictor of churn, suggesting users get frustrated due to navigation issues.
- Increased user engagement and community, e.g. Thumbs Up, Add Friends, Thumbs Down, significanrtly reduce the likelihood of churn.
- A user retention strategy could therefore focus on increased engagement by encouraging community building (Add Friends), user feedback to improve recommendations (Thumbs Up and Thumbs Down) and improving site performance to avoid user frustration.

## Optimising Retention Strategies

To optimise our churn prediction model, it's necessary to balance identifying churning users (True Positives) with minimising the incorrect classification of non-churning users as churners (False Positives). By varying the True Positive Rate (TPR), we can adjust the classification threshold to find the optimal point where the benefits of retaining additional at-risk users outweigh the costs of mistakenly offering retention incentives to users who would not have churned. This section explores how different TPR levels impact the model's performance and the associated financial costs.

### Confusion Matrix Analysis by TPR Thresholds

| TPR Threshold | TP | FP | TN | FN | Description |
|---------------|----|----|----|----|-------------|
| **>= 0.6**    |  9 |  0 |339 |  5 | High precision, zero false positives. |
| **>= 0.7**    | 10 | 27 |312 |  4 | Increase in false positives as TPR rises. |
| **>= 0.8**    | 12 | 57 |282 |  2 | Further increase in false positives. |
| **>= 0.9**    | 13 |130 |209 |  1 | High recall, significant false positives. |
| **= 1.0**     | 14 |226 |113 |  0 | Maximum recall, very high false positives. |

- **TPR >= 0.6**:
  - **Outcome**: Correctly predicts 9 churning users (True Positives, TP) while failing to predict 5 churning users (False Negatives, FN). Notably, this accuracy is achieved without misclassifying any non-churning users (0 False Positives, FP).
  - **Financial Impact**: If Sparkify offers a retention incentive ($10 off a $15 monthly subscription), they would spend $90 (9 TPs x $10) to potentially secure an additional $45 in revenue next month ($5 additional revenue per retained user). If all churning users had churned, Sparkify's revenue from the remaining users would be $5085 ($15 x 339). With this intervention, potential revenue could increase to $5130 assuming all identified users are retained.
  
- **TPR >= 0.7**:
  - **Outcome**: Increases TP to 10 but at a cost of significantly more false positives, jumping from 0 to 27.
  - **Financial Impact**: The cost of retention offers would be $370 ($10 x 37 users). The overall revenue, factoring in the offers, would be $4865 ($15 x 312 + $5 x 37), which is actually lower than if no preventive measures had been taken ($5085), showing a net loss compared to the no-action scenario despite retaining more users.
  
- **TPR >= 1.0**:
  - **Outcome**: Predicts all churning users correctly (14 TPs), but at a very high cost of 226 misclassified non-churning users.
  - **Financial Impact**: The cost in retention offers would be significantly high at $2400 ($10 x 240 users). This approach would yield a revenue of just $2895, a significant financial loss, despite retaining all churning users.

**Conclusion**:
Increasing TPR can initially seem beneficial by retaining more churning users. However, as shown, the financial downside of false positives, i.e. non-churning users incorrectly predicted to churn and offered retention incentives, can outweigh the benefits. Using this model, Sparkify should expect to identify no more than 60% of churning users to avoid the costs of retaining churning users from outweighing the benefits.

**Limitations and Potential Improvements**:
- The analysis did not account for the differing lifetime value (LTV) of each customer.
- It assumes a fixed retention offer. Dynamic pricing models could be developed that take into account the probability of churn and other factors such as the users LTV.


