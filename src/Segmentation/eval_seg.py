from pathlib import Path
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configure paths
current_file = Path(__file__).resolve()
project_path = current_file.parents[2]
frames_without_mask_path = Path(project_path, "Data", "Output_files", "HTB5-170122", "Frames without mask")

def extract_bg_patches():
    background_patches_path = Path(project_path, "Data", "Output_files", "HTB5-170122", "background patches")
    # Ensure output directory exists
    background_patches_path.mkdir(parents=True, exist_ok=True)

    # Load frames
    sliding_window = [128, 128]
    window_height, window_width = sliding_window
    stride = 80
    frames_without_mask = []

    for frame_idx, frame_file in enumerate(sorted(frames_without_mask_path.glob("*.png"))):
        frame = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue  # Skip unreadable files

        frame_height, frame_width = frame.shape

        # Slide window
        patch_idx = 0
        for y in range(0, frame_height - window_height + 1, stride):
            for x in range(0, frame_width - window_width + 1, stride):
                patch = frame[y:y + window_height, x:x + window_width]
                # Save patch
                patch_filename = background_patches_path / f"frame{frame_idx:04d}_patch{patch_idx:05d}.png"
                cv2.imwrite(str(patch_filename), patch)
                patch_idx += 1

def load_images_from_folder(folder, label):
    data, labels = [], []
    for file in folder.glob("*.png"):
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {file}")
            continue
        if img.shape != (128, 128):
            print(f"Warning: Unexpected shape {img.shape} in {file}")
            continue
        data.append(img.flatten())  # Ensure it's 1D, consistent shape
        labels.append(label)
    return np.array(data), np.array(labels)

def prepare_data(cell_dir, bg_dir):
    bg_data, bg_labels = load_images_from_folder(bg_dir, 0)
    cell_data, cell_labels = load_images_from_folder(cell_dir, 1)

    # Combine
    X = np.concatenate([bg_data, cell_data], axis=0)
    y = np.concatenate([bg_labels, cell_labels], axis=0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def run_knn(X_train, X_test, y_train, y_test):
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))

def run_simple_nn(X_train, X_test, y_train, y_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Dataset & DataLoader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Model
    model = nn.Sequential(
        nn.Linear(128 * 128, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(10):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training accuracy
            predicted = pred.argmax(dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        train_accuracy = correct / total
        print(f"Epoch {epoch + 1} complete - Train Accuracy: {train_accuracy:.2%}")

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print(f"Test Accuracy: {correct / total:.2%}")


cells_path = Path(project_path, "Data", "Output_files", "HTB5-170122", "Dataset")
background_patches_path = Path(project_path, "Data", "Output_files", "HTB5-170122", "background patches")
X_train, X_test, y_train, y_test = prepare_data(cells_path, background_patches_path)

run_knn(X_train, X_test, y_train, y_test)
run_simple_nn(X_train, X_test, y_train, y_test)
