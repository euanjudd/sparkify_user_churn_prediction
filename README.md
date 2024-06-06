# sparkify_user_churn_prediction

Precision and Recall

Precision indicates how many of the users predicted to churn actually do churn—it's crucial for avoiding costly mistakes by not falsely identifying satisfied users as churn risks. Recall shows how many of the actual churners are correctly identified—important for capturing as many at-risk users as possible. High precision ensures that money is spent effectively by targeting real churners without wasting resources on non-churners, while high recall ensures more churners are caught but may lead to spending on users who weren't planning to leave. Balancing these metrics helps in using the budget wisely, avoiding unnecessary costs while still retaining a significant number of at-risk users.

Model Selection Results

LogisticRegression fold=0 auc=0.7497354497354498 f1=0.9872
LogisticRegression fold=1 auc=0.8227259684361548 f1=0.9849912984570822
LogisticRegression fold=2 auc=0.9981108312342569 f1=0.9879450336476797
RandomForestClassifier fold=0 auc=0.7593378607809835 f1=0.9633985065944893
RandomForestClassifier fold=1 auc=0.7642543859649122 f1=0.9503510443299837
RandomForestClassifier fold=2 auc=0.5542803970223322 f1=0.9656071439206202
DecisionTreeClassifier fold=0 auc=0.2217427511545159 f1=0.9689644717522529
DecisionTreeClassifier fold=1 auc=0.4999067164179104 f1=0.9455491449598408
DecisionTreeClassifier fold=2 auc=0.5615598885793871 f1=0.9526203870307021
GradientBoostedTrees fold=0 auc=0.4687323146576117 f1=0.9778969927023659
GradientBoostedTrees fold=1 auc=0.4509793485203323 f1=0.9643205472667195
GradientBoostedTrees fold=2 auc=0.7146943691659139 f1=0.9560108281038514

Results for Logistic Regression:
  Average AUC:		0.8569 ± 0.1248
  Average F1 Score:	0.9867 ± 0.0015
Results for Random Forest Classifier:
  Average AUC: 		0.6926 ± 0.1128
  Average F1 Score: 	0.9598 ± 0.0078
Results for Decision Tree Classifier:
  Average AUC: 		0.4277 ± 0.1743
  Average F1 Score: 	0.9557 ± 0.0096
Results for Gradient Boosted Trees:
  Average AUC: 		0.5448 ± 0.1354
  Average F1 Score: 	0.9661 ± 0.0088

Assumptions:

- $10 to retain an existing user
- $50 to acquire a new user
- $15 monthly subscription per user