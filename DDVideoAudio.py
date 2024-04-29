#Audio

import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Function to extract audio features
def extract_audio_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # n_mfcc is usually between 12 and 40
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(e)

# Collect participant IDs and extract features
participant_ids = [i for i in range(300, 377)]  # Replace with actual IDs
audio_features_list = []

for participant_id in participant_ids:
    audio_file = f"D:/DAIC_Dataset/{participant_id}_AUDIO.wav"  
    features = extract_audio_features(audio_file)
    if features is not None:
        audio_features_list.append(features)
    else:
        audio_features_list.append([0] * 13)  # Handle case where features are None

# Create a DataFrame for the audio features
audio_features_df = pd.DataFrame(audio_features_list, columns=[f'mfcc_{i}' for i in range(13)])
audio_features_df['Participant_ID'] = participant_ids

# Load the dataset
df = pd.read_csv("participant_transcript_data.csv")
transcripts = df['Transcript'].tolist()
labels = df['PHQ8_Binary'].tolist()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(audio_features_df.iloc[:, :-1], labels, test_size=0.2, random_state=42)

# Train a RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'audio_feature_model.pkl')



#VGG

# import numpy as np
# import pandas as pd
# import librosa
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib
# # Make sure you have the vggish module available in your environment
# from vggish import VGGish, vggish_input

# # Function to extract audio embeddings using VGGish
# def extract_vggish_embeddings(audio_file):
#     try:
#         # Load the audio file
#         y, sr = librosa.load(audio_file, sr=None)
#         # Produce a batch of log mel spectrogram examples.
#         input_batch = vggish_input.waveform_to_examples(y, sr)
#         # Initialize the VGGish model.
#         vggish_model = VGGish()
#         vggish_model.load_state_dict(torch.load('vggish_pretrained.ckpt'))
#         vggish_model.eval()
#         # Extract embeddings
#         with torch.no_grad():
#             embeddings = vggish_model.forward(input_batch)
#         return embeddings.numpy()
#     except Exception as e:
#         print(f"Error processing {audio_file}: {e}")
#         return None

# # Collect participant IDs and extract features
# participant_ids = [i for i in range(300, 377)]  # Replace with actual IDs
# audio_embeddings_list = []

# for participant_id in participant_ids:
#     audio_file = f"D:/DAIC_Dataset/{participant_id}_AUDIO.wav"  
#     embeddings = extract_vggish_embeddings(audio_file)
#     if embeddings is not None:
#         audio_embeddings_list.append(embeddings.mean(axis=0))
#     else:
#         audio_embeddings_list.append([0] * 128)  # VGGish embeddings have a size of 128

# # Create a DataFrame for the audio embeddings
# audio_embeddings_df = pd.DataFrame(audio_embeddings_list)
# audio_embeddings_df['Participant_ID'] = participant_ids

# # Load the labels
# labels_df = pd.read_csv("participant_transcript_data.csv")
# labels = labels_df['PHQ8_Binary'].values

# # Merge embeddings DataFrame with labels
# data_df = audio_embeddings_df.merge(labels_df, left_on='Participant_ID', right_on='Participant')
# X = data_df.drop(columns=['Participant_ID', 'PHQ8_Binary'])
# y = data_df['PHQ8_Binary']

# # Split data into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a RandomForest classifier on the embeddings
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Evaluate the RandomForest model
# y_pred = rf_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Save the RandomForest model trained on VGGish embeddings
# joblib.dump(rf_model, 'vggish_random_forest_model.pkl')


## Video



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import joblib

# # Function to load a single feature file
# def load_feature_file(filepath):
#     try:
#         return pd.read_csv(filepath, delimiter=',', skiprows=1, header=None)
#     except FileNotFoundError:
#         print(f"File not found: {filepath}")
#         return None

# # Initialize an empty DataFrame to hold all features
# video_features_df = pd.DataFrame()

# # Load features for each participant
# for participant_id in range(300, 377):  # 77 participants from ID 300 to 377
#     participant_features = []
#     missing_any_file = False
#     for feature_name in ['CLNF_AUs', 'CLNF_features', 'CLNF_features3D', 'CLNF_gaze', 'CLNF_pose']:
#         filepath = f"D:/DAIC_Dataset/{participant_id}_{feature_name}.txt"
#         try:
#             # Attempt to load the feature file
#             feature_data = pd.read_csv(filepath, delimiter=',', skiprows=1, header=None).values.flatten()
#             participant_features.append(feature_data)
#         except FileNotFoundError:
#             # If a file is missing, mark the flag and break the loop
#             missing_any_file = True
#             print(f"Missing file for participant {participant_id}: {feature_name}")
#             break  # Skipping the rest of the files for this participant

#     if not missing_any_file:
#         # If all files are present, append the features to the DataFrame
#         all_features = np.concatenate(participant_features)
#         video_features_df = video_features_df.append(pd.Series(all_features), ignore_index=True)
#     else:
#         # If any file is missing, append a row of zeros (or NaNs)
#         # The length of the row should be the sum of lengths of all feature arrays
#         feature_lengths = [len(feature) for feature in participant_features if isinstance(feature, np.ndarray)]
#         all_features = np.zeros(sum(feature_lengths))  # Replace with np.nan if you prefer NaNs
#         video_features_df = video_features_df.append(pd.Series(all_features), ignore_index=True)



# # Load labels from CSV
# labels_df = pd.read_csv("D:/DAIC_Dataset/participant_transcript_data.csv")
# labels = labels_df['PHQ8_Binary'].values


# # Ensure the labels correspond to the loaded feature vectors
# video_features_df['Participant_ID'] = labels_df['Participant'][video_features_df.index]
# labels = labels_df['PHQ8_Binary'][video_features_df.index].values

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(video_features_df.drop('Participant_ID', axis=1), labels, test_size=0.2, random_state=42)

# # Create the Random Forest Classifier
# rf_clf = RandomForestClassifier(random_state=42)
# rf_clf.fit(X_train, y_train)

# # Predict and evaluate the model
# predictions = rf_clf.predict(X_test)
# print(classification_report(y_test, predictions))

# # Save the model
# joblib.dump(rf_clf, 'D:/DAIC_Dataset/video_rf_model.pkl')

#Spark

# from pyspark.sql import SparkSession
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.classification import RandomForestClassifier, LinearSVC
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# import pyspark.sql.functions as F

# # Initialize Spark Session
# spark = SparkSession.builder.appName("DepressionAnalysis").getOrCreate()

# # Function to load a single feature file into a Spark DataFrame
# def load_feature_file_spark(filepath, feature_name):
#     try:
#         df = spark.read.csv(filepath, header=False, inferSchema=True).toDF(*[f"{feature_name}_{i}" for i in range(len(df.columns))])
#         return df
#     except Exception as e:
#         print(f"Error loading file {filepath}: {e}")
#         return None

# # Read labels into a Spark DataFrame
# labels_df = spark.read.csv("D:/DAIC_Dataset/participant_transcript_data.csv", header=True, inferSchema=True)

# # Prepare a list to hold DataFrames for each participant's features
# features_dfs = []

# # Load features for each participant
# for participant_id in range(300, 377):  # Assuming 77 participants from ID 300 to 377
#     participant_features = []
#     missing_any_file = False
#     for feature_name in ['CLNF_AUs', 'CLNF_features', 'CLNF_features3D', 'CLNF_gaze', 'CLNF_pose']:
#         filepath = f"D:/DAIC_Dataset/{participant_id}_{feature_name}.txt"
#         feature_df = load_feature_file_spark(filepath, feature_name)
#         if feature_df is not None:
#             participant_features.append(feature_df)
#         else:
#             missing_any_file = True
#             print(f"Missing file for participant {participant_id}: {feature_name}")
#             break
    
#     if not missing_any_file:
#         # Merge all feature DataFrames into one DataFrame per participant
#         for i, feature_df in enumerate(participant_features):
#             if i == 0:
#                 participant_df = feature_df
#             else:
#                 participant_df = participant_df.join(feature_df)
#         features_dfs.append(participant_df.withColumn("Participant_ID", F.lit(participant_id)))

# # Union all participant DataFrames into one DataFrame
# video_features_df = features_dfs[0]
# for feature_df in features_dfs[1:]:
#     video_features_df = video_features_df.union(feature_df)

# # Join the features DataFrame with labels
# data_df = video_features_df.join(labels_df, "Participant_ID")

# # Assemble features into a feature vector
# feature_columns = data_df.columns[:-2]  # Exclude Participant_ID and label column
# assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
# data_df = assembler.transform(data_df)

# # Split data into training and test sets
# train_df, test_df = data_df.randomSplit([0.8, 0.2], seed=42)

# # Train a RandomForest classifier
# rf = RandomForestClassifier(featuresCol="features", labelCol="PHQ8_Binary", seed=42)
# rf_model = rf.fit(train_df)

# # Predict and evaluate the RandomForest model
# rf_predictions = rf_model.transform(test_df)
# rf_evaluator = BinaryClassificationEvaluator(labelCol="PHQ8_Binary")
# print("RandomForest AUC:", rf_evaluator.evaluate(rf_predictions))

# # Train a Linear Support Vector Machine classifier
# svm = LinearSVC(featuresCol="features", labelCol="PHQ8_Binary")
# svm_model = svm.fit(train_df)

# # Predict and evaluate the SVM model
# svm_predictions = svm_model.transform(test_df)
# svm_evaluator = BinaryClassificationEvaluator(labelCol="PHQ8_Binary")
# print("SVM AUC:", svm_evaluator.evaluate(svm_predictions))

# # Save the RandomForest model
# rf_model.save('D:/DAIC_Dataset/video_rf_model')

# # Save the SVM model
# svm_model.save('D:/DAIC_Dataset/video_svm_model')

# # Stop the Spark session
# spark.stop()





