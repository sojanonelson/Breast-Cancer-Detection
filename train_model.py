import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
import pygame
import tkinter as tk
from threading import Thread

# Initialize pygame mixer
pygame.mixer.init()

# Sound function
def play_sound(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# GUI setup
window = tk.Tk()
window.title("Breast Cancer Model Trainer")
window.geometry("450x300")

status_label = tk.Label(window, text="Waiting to start training...", font=("Helvetica", 14))
status_label.pack(pady=20)

progress = tk.Text(window, height=10, width=50)
progress.pack(pady=10)

def update_status(text):
    progress.insert(tk.END, text + "\n")
    progress.see(tk.END)
    window.update()

def train_model():
    update_status("🔄 Starting training process...")
    play_sound("sounds/start.mp3")

    # Load dataset
    update_status("📥 Loading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                    'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
                    'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    df = pd.read_csv(url, names=column_names)

    # Preprocessing
    update_status("🧹 Cleaning data...")
    df = df[df['bare_nuclei'] != '?']
    df['bare_nuclei'] = df['bare_nuclei'].astype(int)
    df.drop('id', axis=1, inplace=True)

    X = df.drop('class', axis=1)
    y = df['class']

    # Split
    update_status("✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    update_status("🚀 Training RandomForestClassifier...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    update_status(f"✅ Training Accuracy: {train_accuracy * 100:.2f}%")
    update_status(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    update_status("💾 Model saved to model.pkl")

    # Done
    update_status("🎉 Training complete!")
    play_sound("sounds/success.mp3")

# Run training in a separate thread to keep UI responsive
def start_training():
    Thread(target=train_model).start()

# Button to trigger training
train_button = tk.Button(window, text="Start Training", font=("Helvetica", 12), command=start_training)
train_button.pack(pady=10)

# Start the GUI loop
window.mainloop()
