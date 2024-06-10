from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col, count, from_unixtime, date_format, regexp_replace, datediff, lag, to_date, unix_timestamp, coalesce, lit, first, avg, sum, min, max, countDistinct, stddev, lead, expr, monotonically_increasing_id, udf
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.sql.window import Window
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import roc_curve, precision_score, recall_score, confusion_matrix, f1_score, classification_report
import numpy as np

# Create a spark session
spark = SparkSession.builder \
    .appName("Sparkify Data Analysis") \
    .getOrCreate()

# Read in sparkify dataset
mini_data_path = "s3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json"

mini_df = spark.read.json(mini_data_path)

def data_cleaning(input_df: DataFrame) -> DataFrame:
    """
    Perform data cleaning operations on the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame to be cleaned.

    Returns:
        DataFrame: The cleaned DataFrame.

    """
    # Remove unnecessary columns and columns that are too high dimensional to be both OneHotEncoded and useful.
    cleaned_df = input_df.drop("auth", "firstName", "lastName", "method", "artist", "location")
    
    # Only keep OS information from 'userAgent'
    cleaned_df = cleaned_df.withColumn('userAgent', 
                   when(col('userAgent').contains('Mac'), 'Mac')
                   .when(col('userAgent').contains('Windows'), 'Windows')
                   .when(col('userAgent').contains('Linux'), 'Linux')
                   .otherwise('Other'))
    
    # Fill null values as the absence of a value can sometimes be predictive.
    fill_values = {
        "gender": "Missing",
        "userAgent": "Missing",
        "length": 0
    }
    cleaned_df = cleaned_df.na.fill(fill_values)

    # Remove the small number of users that do not have "registration" data.
    cleaned_df = cleaned_df.filter(col("registration").isNotNull())

    ## Keep non-null and users who have been on the paid tier at least once. We are only interested in the churn of paid users.
    paid_users_filter = (col("level") == 'paid') & col("userId").isNotNull()
    df_paid_users = cleaned_df.filter(paid_users_filter).select("userId").distinct()

    result_df = cleaned_df.join(df_paid_users, on="userId", how="inner")

    return result_df


def aggregate_session_data(input_df: DataFrame) -> DataFrame:
    """
    Aggregates session data based on the sessionId.

    Args:
        input_df (DataFrame): The input DataFrame containing session data.

    Returns:
        DataFrame: The aggregated DataFrame with session data.

    """
    return input_df.groupBy("sessionId").agg(
        first("gender").alias("gender"),
        max("itemInSession").alias("itemInSession"),
        sum("length").alias("length"),
        first("level").alias("level"),
        first("registration").alias("registration"),
        min("ts").alias("ts"),
        first("userAgent").alias("userAgent"),
        first("userId").alias("userId"),
        count(when(col("song").isNotNull(), True)).alias("numberOfSongs")
    )

def add_user_session_stats(input_df: DataFrame) -> DataFrame:
    """
    Calculates session statistics for each user and adds them to the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame containing user session data.

    Returns:
        DataFrame: The input DataFrame with additional columns for session statistics.

    """
    user_session_stats = input_df.groupBy("userId").agg(
        avg("length").alias("avg_session_length"),
        countDistinct("sessionId").alias("total_sessions"),
        avg("numberOfSongs").alias("avg_songs_per_session")
    )
    return input_df.join(user_session_stats, "userId")

def calculate_deviation_features(input_df: DataFrame) -> DataFrame:
    """
    Calculates deviation features based on the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame containing the necessary columns.

    Returns:
        DataFrame: The DataFrame with additional columns for deviation features.
    """
    return input_df.withColumn("deviation_from_avg_length", col("length") - col("avg_session_length"))\
             .withColumn("deviation_from_avg_songs", col("numberOfSongs") - col("avg_songs_per_session"))

def pivot_page_and_status_counts(input_df: DataFrame) -> DataFrame:
    """
    Calculates the counts of pages and statuses for each session in the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame containing session data.

    Returns:
        Tuple[DataFrame, DataFrame]: A tuple containing two DataFrames - page_counts and status_counts.
            - page_counts: DataFrame with counts of pages for each session.
            - status_counts: DataFrame with counts of statuses for each session.
    """
    page_counts = input_df.groupBy("sessionId").pivot("page").count().na.fill(0)
    status_counts = input_df.groupBy("sessionId").pivot("status").count().na.fill(0)
    return page_counts, status_counts

def add_human_readable_dates(input_df: DataFrame) -> DataFrame:
    """
    Adds human-readable dates to the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame containing the 'registration' and 'ts' columns.

    Returns:
        DataFrame: The input DataFrame with additional 'registration_date' and 'activity_date' columns.

    """
    return input_df.withColumn('registration_date', from_unixtime(col('registration') / 1000).cast('timestamp'))\
             .withColumn('activity_date', from_unixtime(col('ts') / 1000).cast('timestamp'))\
             .drop("ts", "registration")

def calculate_tenure_and_recency(input_df: DataFrame) -> DataFrame:
    """
    Calculates the tenure and recency of user activity based on the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame containing user activity data.

    Returns:
        DataFrame: The input DataFrame with additional columns 'tenure' and 'activity_recency'.

    """
    user_window = Window.partitionBy("userId").orderBy("activity_date")
    input_df = input_df.withColumn('tenure', datediff(to_date(col('activity_date')), to_date(col('registration_date'))))
    input_df = input_df.withColumn('previous_activity_date', lag('activity_date').over(user_window))
    input_df = input_df.withColumn('activity_recency', (unix_timestamp('activity_date') - unix_timestamp('previous_activity_date')) / 60)
    input_df = input_df.drop('previous_activity_date')
    return input_df.withColumn('activity_recency', coalesce(col('activity_recency'), lit(0)))

def calculate_session_interval_stats(input_df: DataFrame) -> DataFrame:
    """
    Calculates the average and standard deviation of session intervals for each user.

    Args:
        input_df (DataFrame): The input DataFrame containing user activity data.

    Returns:
        DataFrame: A DataFrame with the average and standard deviation of session intervals for each user.
    """
    return input_df.groupBy("userId").agg(
        avg(when(col("activity_recency") != 0, col("activity_recency"))).alias("avg_session_interval"),
        stddev(when(col("activity_recency") != 0, col("activity_recency"))).alias("stddev_session_interval")
    )

def replace_nulls_with_global_stats(input_df: DataFrame, global_avg_interval: float) -> DataFrame:
    """
    Replaces null values in the input DataFrame with global statistics.

    Args:
        input_df (DataFrame): The input DataFrame containing null values.
        global_avg_interval (float): The global average session interval.

    Returns:
        DataFrame: The input DataFrame with null values replaced by global statistics.
    """
    return input_df.na.fill({
        'avg_session_interval': global_avg_interval,
        'stddev_session_interval': 0
    })

def normalize_length(input_df: DataFrame) -> DataFrame:
    """
    Normalize the length column in the input DataFrame by dividing it by the average session length.

    Args:
        input_df (DataFrame): The input DataFrame containing the length and avg_session_length columns.

    Returns:
        DataFrame: The input DataFrame with an additional column 'normalized_length' that contains the normalized length values.
    """
    return input_df.withColumn(
        "normalized_length",
        when(col("avg_session_length") == 0, 0)  # Replace zero division with 0
        .otherwise(col("length") / col("avg_session_length"))
    ).na.fill({"normalized_length": 0})

def feature_engineering(input_df: DataFrame) -> DataFrame:
    """
    Perform feature engineering on the input DataFrame.

    Args:
        input_df (DataFrame): The input DataFrame containing the raw data.

    Returns:
        DataFrame: The resulting DataFrame after feature engineering.
    """
    # Aggregate session data
    grouped_df = aggregate_session_data(input_df)
    
    # Add user session statistics: (1) average session length per user, (2) total sessions per user, (3) average number of songs per session per user.
    grouped_df = add_user_session_stats(grouped_df)
    
    # Calculate deviation features for each session: deviation from the users average for (1) the session length and (2) the number of songs.
    grouped_df = calculate_deviation_features(grouped_df)
    
    # Pivot 'page' and 'status' column counts. Each category has a column containing the count of visits to that page per session.
    page_counts, status_counts = pivot_page_and_status_counts(input_df)
    grouped_df = grouped_df.join(page_counts, on="sessionId", how="left")
    grouped_df = grouped_df.join(status_counts, on="sessionId", how="left")

    # Drop 'Cancel' as it is essentially equivalent to 'Cancellation Confirmation' and could cause data leakage.
    grouped_df = grouped_df.drop("Cancel")
    
    # Add human-readable dates
    tenure_df = add_human_readable_dates(grouped_df)
    
    # Calculate tenure and activity recency
    tenure_df = calculate_tenure_and_recency(tenure_df)
    
    # Calculate session interval statistics
    session_interval_stats = calculate_session_interval_stats(tenure_df)
    intervals_df = tenure_df.join(session_interval_stats, on="userId", how="left")
    
    # Calculate global average interval for users with more than one session
    global_avg_interval = tenure_df.filter(col("total_sessions") > 1).agg(
        {'activity_recency': 'avg'}
    ).collect()[0][0]
    
    # Replace nulls with global stats
    intervals_df = replace_nulls_with_global_stats(intervals_df, global_avg_interval)
    
    # Normalize session lengths
    result_df = normalize_length(intervals_df)
    
    return result_df

def adjust_churn_labels(input_df: DataFrame) -> DataFrame:
    """
    Adjusts the churn labels in the input DataFrame. 
    The churn label is set to 1 if the user has churned in the current or any of the next 3 sessions.

    Args:
        input_df (DataFrame): The input DataFrame containing the churn data.

    Returns:
        DataFrame: The resulting DataFrame with adjusted churn labels.
    """
    # Rename churn column
    churn_df = input_df.withColumnRenamed("Cancellation Confirmation", "churn")

    # Create a window partitioned by userId and ordered descending by activity_date
    user_window = Window.partitionBy("userId").orderBy(col("activity_date").desc())

    # Assuming df has a 'churn' column marked 1 at the actual churn session
    churn_df = churn_df.withColumn("label_1", lead("churn", 1).over(user_window))
    churn_df = churn_df.withColumn("label_2", lead("churn", 2).over(user_window))
    churn_df = churn_df.withColumn("label_3", lead("churn", 3).over(user_window))

    # Use coalesce to treat nulls as 0 in the churn calculation
    churn_df = churn_df.withColumn("churn", when(
        (col("churn") == 1) | 
        (coalesce(col("label_1"), lit(0)) == 1) | 
        (coalesce(col("label_2"), lit(0)) == 1) | 
        (coalesce(col("label_3"), lit(0)) == 1),
        1
    ).otherwise(0))

    # Clean up temporary columns
    result_df = churn_df.drop("label_1", "label_2", "label_3")

    return result_df

def encode_categorical_columns(input_df: DataFrame) -> DataFrame:
    """
    Encodes categorical columns in the input DataFrame using OneHotEncoder.

    Args:
        input_df (DataFrame): The input DataFrame containing the categorical columns.

    Returns:
        DataFrame: The DataFrame with the categorical columns encoded.

    """
    categorical_columns = ["gender", "level", "userAgent"]

    # Generate the names of the indexed and encoded columns
    index_columns = [col + "Index" for col in categorical_columns]
    vec_columns = [col + "Vec" for col in categorical_columns]
    
    # Indexing all categorical columns
    indexer = StringIndexer(inputCols=categorical_columns, outputCols=index_columns)
    indexed_df = indexer.fit(input_df).transform(input_df)
    
    # Applying OneHotEncoder
    encoder = OneHotEncoder(inputCols=index_columns, outputCols=vec_columns, dropLast=False)
    ohe_df = encoder.fit(indexed_df).transform(indexed_df)
    
    # Dropping original and indexed columns
    all_cols_to_drop = categorical_columns + index_columns
    ohe_df = ohe_df.drop(*all_cols_to_drop)
    
    return ohe_df

def stratified_split_churn_data(input_df: DataFrame, churn_column: str = "churn", train_fraction: float = 0.8, seed: int = 42):
    """
    Perform stratified split of churn data into training and testing datasets. 
    Each userId will appear in either the training or testing dataset but not both.

    Args:
        input_df (DataFrame): The input DataFrame containing churn data.
        churn_column (str, optional): The name of the churn column. Defaults to "churn".
        train_fraction (float, optional): The fraction of data to be used for training. Defaults to 0.8.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        train_df (DataFrame): The training DataFrame containing stratified churn data.
        test_df (DataFrame): The testing DataFrame containing stratified churn data.
    """
    # Ensure users who have churn=1 in at least one session are only included in the churned_users data
    user_churn_status = input_df.groupBy("userId").agg(max(churn_column).alias(churn_column))
    
    # Split into churned and non-churned DataFrames
    churned_users = user_churn_status.filter(col(churn_column) == 1).select("userId")
    non_churned_users = user_churn_status.filter(col(churn_column) == 0).select("userId")
    
    # Perform stratified split for both churned and non-churned users
    train_churned, test_churned = churned_users.randomSplit([train_fraction, 1.0 - train_fraction], seed=seed)
    train_non_churned, test_non_churned = non_churned_users.randomSplit([train_fraction, 1.0 - train_fraction], seed=seed)
    
    # Combine training and testing datasets
    train_users = train_churned.union(train_non_churned)
    test_users = test_churned.union(test_non_churned)
    
    # Join back to the original data
    train_df = input_df.join(train_users, ["userId"], "inner")
    test_df = input_df.join(test_users, ["userId"], "inner")
    
    return train_df, test_df

def custom_stratified_cross_val(df_scv: DataFrame, k: int, seed: int = 42):
    """
    Perform custom stratified cross-validation on a DataFrame.
    Each userId will appear in one fold only.

    Args:
        df_scv (DataFrame): The input DataFrame containing the data to be cross-validated.
        k (int): The number of folds to create.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        list: A list of DataFrames, where each DataFrame represents a fold.

    """
    # Mark users as churned if any of their sessions are marked as churned
    user_churn_status = df_scv.groupBy("userId").agg(max("churn").alias("churn"))
    
    # Split users into churned and non-churned
    churned_users = user_churn_status.filter(col("churn") == 1).select("userId")
    non_churned_users = user_churn_status.filter(col("churn") == 0).select("userId")
    
    # Assign each user to a fold
    churned_users = churned_users.withColumn('fold', (monotonically_increasing_id() % k))
    non_churned_users = non_churned_users.withColumn('fold', (monotonically_increasing_id() % k))
    
    # Union and join back to the original data
    users_fold = churned_users.union(non_churned_users)
    df_scv = df_scv.join(users_fold, on='userId', how='inner')
    
    # Generate folds
    folds = [df_scv.filter(col('fold') == i).drop('fold') for i in range(k)]
    
    return folds

# Cross-validation function
def cross_validate_model(pipeline, folds, auc_evaluator, f1_evaluator, verbose=True):
    """
    Cross-validates a machine learning model using k-fold cross-validation.

    Args:
        pipeline (Pipeline): The machine learning pipeline to be trained and evaluated.
        folds (list): A list of data folds for cross-validation.
        auc_evaluator (Evaluator): The evaluator used to calculate the AUC metric.
        f1_evaluator (Evaluator): The evaluator used to calculate the F1 metric.
        verbose (bool, optional): Whether to print the evaluation metrics for each fold. Defaults to True.

    Returns:
        tuple: A tuple containing the average AUC and average F1 metrics across all folds.
    """
    auc_metrics = []
    f1_metrics = []
    
    for i in range(len(folds)):
        cv_train = [folds[j] for j in range(len(folds)) if j != i]
        cv_test = folds[i]
        
        # Combine training folds
        cv_train_df = cv_train[0]
        for fold in cv_train[1:]:
            cv_train_df = cv_train_df.union(fold)
        
        # Fit the pipeline on the training set
        model = pipeline.fit(cv_train_df)
        
        # Make predictions on the test set
        predictions = model.transform(cv_test)
        
        # Evaluate the model
        auc = auc_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)
        
        # Store metrics
        auc_metrics.append(auc)
        f1_metrics.append(f1)
        
        if verbose:
            print(f"  Fold {i}: AUC={auc}, F1={f1}")
    
    # Calculate average metrics
    average_auc = __builtins__.sum(auc_metrics) / len(auc_metrics)
    average_f1 = __builtins__.sum(f1_metrics) / len(f1_metrics)

    print(f"  Average: AUC={average_auc}, F1={average_f1}")
    
    return average_auc, average_f1

cleaned_df = data_cleaning(mini_df)

engineered_df = feature_engineering(cleaned_df)

labelled_df = adjust_churn_labels(engineered_df)

encoded_df = encode_categorical_columns(labelled_df)

train_df, test_df = stratified_split_churn_data(encoded_df)

folds = custom_stratified_cross_val(train_df, k=3)

# Feature processing stages
assembler = VectorAssembler(
    inputCols=[col for col in train_df.columns if col not in ["userId", "sessionId", "churn", "registration_date", "activity_date"]],
    outputCol="features_unscaled")

scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)

# Evaluation metrics
auc_evaluator = BinaryClassificationEvaluator(labelCol='churn', metricName='areaUnderROC')
f1_evaluator = MulticlassClassificationEvaluator(labelCol='churn', metricName='f1')

####################################################################
# Baseline classifier
####################################################################

# Baseline classifier that always predicts 'not churn' (0)
baseline_predictions = test_df.withColumn("prediction", lit(0).cast("double"))

# Evaluators
bl_auc_evaluator = BinaryClassificationEvaluator(labelCol='churn', rawPredictionCol='prediction', metricName='areaUnderROC')
bl_f1_evaluator = MulticlassClassificationEvaluator(labelCol='churn', predictionCol='prediction', metricName='f1')

# Calculate AUC
bl_auc = bl_auc_evaluator.evaluate(baseline_predictions)

# Calculate F1 score
bl_f1 = bl_f1_evaluator.evaluate(baseline_predictions)

# Print baseline evaluation metrics
print(f"Baseline Model AUC: {bl_auc}")
print(f"Baseline Model F1 Score: {bl_f1}")

####################################################################
# Model Selection
####################################################################

classifiers = {
    "LogisticRegression": LogisticRegression(featuresCol='features', labelCol='churn'),
    "RandomForestClassifier": RandomForestClassifier(featuresCol='features', labelCol='churn'),
    "GradientBoostedTrees": GBTClassifier(featuresCol='features', labelCol='churn')
}

results = {}
for name, classifier in classifiers.items():
    print(f"Starting {name}.")
    pipeline = Pipeline(stages=[assembler, scaler, classifier])
    average_auc, average_f1 = cross_validate_model(pipeline, folds, auc_evaluator, f1_evaluator)

    results[name] = {
        "Average AUC": average_auc,
        "Average F1 Score": average_f1
    }

# Print results for classifiers
for model_name, metrics in results.items():
    print(f"Results for {model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

####################################################################
# Hyperparameter tuning
####################################################################

# Hyperparameter tuning for Logistic Regression
parameter_grid = [
    (regParam, elasticNetParam, maxIter) for regParam in [0.0, 0.01, 0.1]
    for elasticNetParam in [0.0, 50]
    for maxIter in [50, 100, 200]
]

best_model_metrics = {"Average AUC": 0, "Average F1 Score": 0, "Best Params": None}

for params in parameter_grid:
    print(f"Starting {params}.")
    regParam, elasticNetParam, maxIter = params
    logistic_regression = LogisticRegression(featuresCol='features', labelCol='churn', regParam=regParam, elasticNetParam=elasticNetParam, maxIter=maxIter)
    pipeline = Pipeline(stages=[assembler, scaler, logistic_regression])
    
    average_auc, average_f1 = cross_validate_model(pipeline, folds, auc_evaluator, f1_evaluator)

    if average_auc > best_model_metrics["Average AUC"]:
        best_model_metrics = {
            "Average AUC": average_auc,
            "Average F1 Score": average_f1,
            "Best Params": params
        }

# Print results for the best logistic regression model
print("Best Model Parameters:", best_model_metrics["Best Params"])
print("Best Model Average AUC:", best_model_metrics["Average AUC"])
print("Best Model Average F1 Score:", best_model_metrics["Average F1 Score"])

####################################################################
# Train Final Model
####################################################################

print(f"Training Final Model.")

regParam, elasticNetParam, maxIter = best_model_metrics["Best Params"]

# Logistic regression model with the best parameters
best_lr = LogisticRegression(featuresCol='features', labelCol='churn', regParam=regParam, elasticNetParam=elasticNetParam, maxIter=maxIter)

# Create the pipeline
pipeline = Pipeline(stages=[assembler, scaler, best_lr])

# Train the model on the full training dataset
final_model = pipeline.fit(train_df)

# Evaluate on the test dataset
test_predictions = final_model.transform(test_df)
test_predictions.cache()

# Calculate AUC and F1 score
auc = auc_evaluator.evaluate(test_predictions)
f1 = f1_evaluator.evaluate(test_predictions)

# Evaluate precision and recall using MulticlassClassificationEvaluator
precision_evaluator = MulticlassClassificationEvaluator(labelCol='churn', predictionCol='prediction', metricName='weightedPrecision')
recall_evaluator = MulticlassClassificationEvaluator(labelCol='churn', predictionCol='prediction', metricName='weightedRecall')

precision = precision_evaluator.evaluate(test_predictions)
recall = recall_evaluator.evaluate(test_predictions)

# Confusion Matrix
conf_matrix = test_predictions.groupBy('churn').pivot('prediction').count().fillna(0).orderBy('churn')

# Print the evaluation metrics
print(f"Final Model Average AUC: {auc}")
print(f"Final Model Average F1 Score: {f1}")
print(f"Final Model Precision: {precision}")
print(f"Final Model Recall: {recall}")
print(f"Confusion Matrix:\n")
conf_matrix.show()

####################################################################
# Feature Importance
####################################################################

print("Feature Importances (sorted):")

# Last stage in the pipeline is the LogisticRegressionModel
best_model = final_model.stages[-1]

# Create a list of (feature, importance) 
feature_importances = [(feature, importance) for feature, importance in zip(assembler.getInputCols(), best_model.coefficients)]

# Sort the features by the absolute value of their importances, descending
feature_importances_sorted = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)

for feature, importance in feature_importances_sorted:
    print(f"  {feature}: {importance}")

####################################################################
# Metrics for different TPR
####################################################################

test_predictions_new = test_predictions.withColumn("probability_array", vector_to_array("probability"))
test_predictions_new.cache()
test_predictions_new = test_predictions_new.withColumn("probability_1", col("probability_array")[1])

print(f"Count before threshold adjustment: {test_predictions_new.count()}")

# Calculate precision and recall
data_collected = test_predictions_new.select("churn", "probability_1").collect()
y_true = [row['churn'] for row in data_collected]
y_prob = [row['probability_1'] for row in data_collected]

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Find thresholds for TPR levels
tpr_levels = [0.6, 0.7, 0.8, 0.9, 1.0]

# Adjusted code to apply threshold and calculate metrics
for tpr_level in tpr_levels:
    threshold_idx = np.where(tpr >= tpr_level)[0][0]
    threshold = thresholds[threshold_idx]

    # Apply the threshold to the 'probability' column
    adjusted_predictions = test_predictions_new.withColumn(
        "adjusted_prediction", 
        when(col("probability_1") >= threshold, 1).otherwise(0)
    )

    print(f"Count after threshold adjustment: {adjusted_predictions.count()}")
    
    # Evaluate adjusted predictions
    data_collected = adjusted_predictions.select("churn", "adjusted_prediction").collect()
    y_true_adj = [row['churn'] for row in data_collected]
    y_pred_adj = [row['adjusted_prediction'] for row in data_collected]   

    precision_adj = precision_score(y_true_adj, y_pred_adj)
    recall_adj = recall_score(y_true_adj, y_pred_adj)
    f1_adj = f1_score(y_true_adj, y_pred_adj)
    conf_matrix_adj = confusion_matrix(y_true_adj, y_pred_adj)

    # Calculate the short-term financial impact
    tp_count = conf_matrix_adj[1, 1]  # True Positives
    fp_count = conf_matrix_adj[0, 1]  # False Positives
    retention_cost = 10
    monthly_subscription = 15
    # Short-term gain for an intervention that retains an existing user
    net_savings_per_tp = monthly_subscription - retention_cost
    # We want to maximise the total_benefit
    total_benefit = (tp_count * net_savings_per_tp) - (fp_count * retention_cost)

    # Calculate the longer term financial impact
    customer_lifetime = 12  # months
    annual_discount_rate = 0.05
    monthly_discount_rate = (1 + annual_discount_rate) ** (1/12) - 1
    # Calculate lifetime value
    ltv = monthly_subscription * (customer_lifetime / (1 + monthly_discount_rate))
    total_ltv = (tp_count * ltv) - (fp_count * retention_cost)

    print(f"\nFor TPR >= {tpr_level}:")
    print(f"Threshold: {threshold}")
    print(f"Precision: {precision_adj}")
    print(f"Recall: {recall_adj}")
    print(f"F1 Score: {f1_adj}")
    print(f"Confusion Matrix:\n{conf_matrix_adj}")
    print(f"Short-term Benefit: ${total_benefit}")
    print(f"LTV-based Benefit: ${total_ltv}")
    print(classification_report(y_true_adj, y_pred_adj))

test_predictions.unpersist()
test_predictions_new.unpersist()