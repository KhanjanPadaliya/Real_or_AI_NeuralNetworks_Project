import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))

# Download Kaggle dataset via kagglehub
path = kagglehub.dataset_download('xhlulu/140k-real-and-fake-faces')
print('Path to dataset files:', path)
print('Directory contents:', os.listdir(path))

# Load CSV with labels and paths
df = pd.read_csv(os.path.join(path, 'train.csv'))
print('Columns:', df.columns.tolist())

# Build full image paths
image_base = os.path.join(path, 'real_vs_fake', 'real-vs-fake')
df['full_path'] = df['path'].apply(lambda p: os.path.join(image_base, p))
df[['full_path', 'label', 'label_str']].head()

sample = df.sample(16, random_state=42)
plt.figure(figsize=(10, 10))
for i, row in enumerate(sample.itertuples()):
    img_path = row.full_path
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path)
    plt.subplot(4, 4, i + 1)
    plt.imshow(img)
    plt.title(row.label_str)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Train/Val/Test split
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['label'],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['label'],
    random_state=42,
)

print('Train size:', len(train_df))
print('Val size  :', len(val_df))
print('Test size :', len(test_df))

IMG_SIZE = 224
BATCH_SIZE = 32

def df_to_dataset(dataframe, shuffle=True):
    paths = dataframe['full_path'].values
    labels = dataframe['label'].values.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=42)
    return ds

train_ds_raw = df_to_dataset(train_df, shuffle=True)
val_ds_raw   = df_to_dataset(val_df,   shuffle=False)
test_ds_raw  = df_to_dataset(test_df,  shuffle=False)

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name='data_augmentation',
)

def load_and_preprocess_image(path, label, training=True):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = data_augmentation(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds_raw
    .map(lambda p, y: load_and_preprocess_image(p, y, training=True),
         num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds_raw
    .map(lambda p, y: load_and_preprocess_image(p, y, training=False),
         num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds_raw
    .map(lambda p, y: load_and_preprocess_image(p, y, training=False),
         num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

"""## CNN Model – DenseNet121

We first train a **DenseNet121-based CNN** on the tf.data pipeline.
"""

from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Build DenseNet121 CNN
cnn_base = DenseNet121(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

cnn_model = Sequential([
    cnn_base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

cnn_model.summary()

early_stop_cnn = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr_cnn = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
checkpoint_cnn = ModelCheckpoint('cnn_densenet_best.keras', save_best_only=True, monitor='val_loss')

EPOCHS_CNN = 10

history_cnn = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_CNN,
    callbacks=[early_stop_cnn, reduce_lr_cnn, checkpoint_cnn],
)

# Evaluate CNN on test set
test_loss_cnn, test_acc_cnn = cnn_model.evaluate(test_ds)
print(f'CNN Test Accuracy: {test_acc_cnn:.4f}')

# Collect predictions
y_true = []
y_pred_probs = []
for batch_images, batch_labels in test_ds:
    preds = cnn_model.predict(batch_images, verbose=0)
    y_pred_probs.extend(preds.squeeze().tolist())
    y_true.extend(batch_labels.numpy().tolist())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs >= 0.5).astype(int)

cm_cnn = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print('CNN Classification Report:')
print(classification_report(y_true, y_pred, digits=4))

"""## Vision Transformer (ViT) Model

Now we define and train a **custom Vision Transformer (ViT)** using the same tf.data pipeline.
"""

cnn_model.save('cnn_densenet_real_fake.keras')

print('Saved cnn_densenet_real_fake.keras and vit_real_fake.keras')