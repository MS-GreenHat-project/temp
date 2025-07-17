import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, help='azureml:/subscriptions/ca3a7121-8c69-4769-a3c4-01dbafb3872d/resourceGroups/Green-Hat/providers/Microsoft.MachineLearningServices/workspaces/greenhat-ai/data/labeling-data/versions/5')
parser.add_argument('--output-dir', type=str, help='/mnt/azureml/cr/j/2c9d42b6649043a881fd02c079f5d1ca/cap/data-capability/wd/model_output')
args = parser.parse_args()

data_path = os.path.join(args.data_folder, 'labels.csv')
df = pd.read_csv(data_path)
X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs(args.output_dir, exist_ok=True)
joblib.dump(model, os.path.join(args.output_dir, 'model.pkl')) 