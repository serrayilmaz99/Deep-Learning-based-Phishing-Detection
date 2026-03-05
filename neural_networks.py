import pandas as pd
import numpy as np
from feature_main import training_df2

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import shap


training_df2 = shuffle(training_df2)
phishing_t = training_df2['phishing'] 
training_df_t = training_df2.drop(columns=['phishing'])

X_train, X_test, y_train, y_test = train_test_split(training_df_t, phishing_t, train_size=0.8, random_state=42)


scaler = MinMaxScaler()
X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

print(X_train.shape, X_test.shape)
print(f'X_train shape: {X_train_scaled.shape}')
print(f'y_train shape: {y_train_tensor.shape}')


class PhishingDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(PhishingDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


input_dim = X_train_scaled.shape[1]
model = PhishingDetectionNN(input_dim)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 20
batch_size = 32

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_scaled.size()[0])
    
    for i in range(0, X_train_scaled.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_scaled[indices], y_train_tensor[indices]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).float()

y_test_np = y_test_tensor.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()

accuracy = accuracy_score(y_test_np, y_pred_np)
f1 = f1_score(y_test_np, y_pred_np)
cm = confusion_matrix(y_test_np, y_pred_np)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(colorbar=False)
plt.show()



def model_predict(X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

explainer = shap.Explainer(model_predict, X_train_scaled.numpy())
shap_values = explainer(X_test_scaled.numpy())

shap.summary_plot(shap_values, X_test_scaled.numpy(), feature_names=X_train.columns)
