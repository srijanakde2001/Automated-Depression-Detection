import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from vggish import VGGish


def extract_audio_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # n_mfcc is usually between 12 and 40
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(e)
# Instantiate the VGGish model and load pre-trained weights
vggish_model = joblib.load('audio_model.pkl')
vggish_model.eval()
participant_ids = [i for i in range(300, 377)]  # Replace with actual IDs
audio_file_paths = []
for participant_id in participant_ids:
    audio_file_paths.append(f"D:/DAIC_Dataset/{participant_id}_AUDIO.wav")
# Load your audio files and process them with VGGish
# You need to write the `load_audio` function to load the audio file and preprocess it as required by VGGish
audio_embeddings = []
for audio_path in audio_file_paths:  # List of paths to audio files
    audio_input = extract_audio_features(audio_path)  # Replace with actual preprocessing function
    with torch.no_grad():
        embedding = vggish_model(audio_input)  # Get the embeddings from VGGish
    audio_embeddings.append(embedding.cpu().numpy())

# At this point, `audio_embeddings` contains the embeddings from the VGGish model
# You might need to flatten or reshape them as necessary for your RandomForest model
from sklearn import svm

# Load your trained SVM model
svm_model = joblib.load('video_svm.pkl')
# Function to load a single feature file
def extract_video_features(filepath):
    try:
        return pd.read_csv(filepath, delimiter=',', skiprows=1, header=None)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
# Extract features from the video as required by your SVM model
# This typically involves frame extraction, feature calculation, and flattening
video_embeddings = []
video_file_paths = []
for participant_id in participant_ids:
    for feature_name in ['CLNF_AUs', 'CLNF_features', 'CLNF_features3D', 'CLNF_gaze', 'CLNF_pose']:
         video_file_paths.append(f"D:/DAIC_Dataset/{participant_id}_{feature_name}.txt")# Replace with actual video feature extraction function
for video_path in video_file_paths:  # List of paths to video files
    video_features = extract_video_features(video_path)
    video_embedding = svm_model.predict(video_features)  # Here we get the decision function as the embedding
    video_embeddings.append(video_embedding)

# Now, `video_embeddings` should contain the decision scores or embeddings from the SVM
from transformers import BertModel, BertTokenizer

# Load pre-trained model tokenizer and model
bert_model = torch.load('text_model.pth')
bert_model.eval()
text_data = []
# Tokenize and encode sentences in the BERT format
# You need to write the `load_text_data` function to load and preprocess your text data as required by BERT
text_embeddings = []
for text in text_data:  # List or array of text strings
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the pooled output for embeddings
    text_embeddings.append(outputs.pooler_output.cpu().numpy())

# `text_embeddings` now contains the embeddings from BERT
import torch
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained models
audio_model = joblib.load('audio_model.pkl')
video_model = joblib.load('video_model.pkl')
text_model = torch.load('text_model.pth')  # Adjust based on how your text model was saved.
text_model.eval()  # If it's a PyTorch model

# Combine predictions/embeddings from each modality-specific model into one feature set
combined_features = np.concatenate((audio_embeddings, video_embeddings, text_embeddings), axis=1)

# Standardize the features
scaler = StandardScaler()
combined_features_scaled = scaler.fit_transform(combined_features)

# Labels need to be provided as a numpy array, replace 'your_labels.npy' with the path to your labels
labels = np.load('your_labels.npy')

# Split the data into training and testing sets (adjust the splits as needed)
X_train, X_test, y_train, y_test = train_test_split(combined_features_scaled, labels, test_size=0.2, random_state=42)

# Train the MLP classifier on the combined features
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Save the MLP model
joblib.dump(mlp, 'multi_modal_mlp_model.pkl')

# Load the multi-modal model for evaluation or further use
multi_modal_mlp_model = joblib.load('multi_modal_mlp_model.pkl')

# Evaluate the model if you have a test set
predictions = multi_modal_mlp_model.predict(X_test)
print(classification_report(y_test, predictions))


