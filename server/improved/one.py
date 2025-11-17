"""
ADVANCED DEEPFAKE DETECTION SYSTEM
===================================
This system uses deep learning to detect manipulated (deepfake) videos and images.

Key Features:
- Multiple CNN architectures (Custom CNN, EfficientNet, Xception)
- Face detection and extraction using MTCNN
- Temporal analysis for video sequences
- Advanced data augmentation
- Ensemble predictions
- Explainability with Grad-CAM
- Comprehensive evaluation metrics
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # Corrected import
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, # Corrected import
                                     Dropout, Input, GlobalAveragePooling2D, 
                                     BatchNormalization, LSTM, TimeDistributed,
                                     Bidirectional, Concatenate, Activation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Corrected import
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, # Corrected import
                                        ReduceLROnPlateau, TensorBoard, CSVLogger)
from tensorflow.keras.applications import EfficientNetB0, Xception # Corrected import
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Try to import MTCNN for face detection (optional)
try:
    # Ensure you install the correct package for MTCNN
    from mtcnn.mtcnn import MTCNN 
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: MTCNN not available. Install with: pip install mtcnn")

# ==================== CONFIGURATION ====================
class Config:
    """
    Configuration class for all hyperparameters and paths.
    Modify these values based on your dataset and requirements.
    """
    # Dataset paths
    TRAIN_DIR = "/content/dataset/images/Dataset/Train"
    TEST_DIR = "/content/dataset/images/Dataset/Test"
    VIDEO_DIR = "/content/dataset/videos/DFD_original sequences"
    
    # Output paths
    MODEL_SAVE_PATH = "/content/models/deepfake_model_best.h5"
    LOGS_DIR = "/content/logs"
    RESULTS_DIR = "/content/results"
    FRAMES_OUTPUT_PATH = "/content/extracted_frames"
    
    # Model hyperparameters
    IMG_SIZE = (224, 224)  # Input image dimensions
    BATCH_SIZE = 32        # Number of samples per training batch
    EPOCHS = 50            # Maximum training epochs (early stopping may stop earlier)
    LEARNING_RATE = 0.001  # Initial learning rate
    
    # Video processing parameters
    FRAMES_PER_VIDEO = 30     # Number of frames to extract per video
    SEQUENCE_LENGTH = 10      # Frames per sequence for LSTM analysis
    FRAME_SKIP = 5            # Extract every Nth frame
    
    # Model architecture choice
    MODEL_TYPE = "efficientnet"  # Options: "custom", "efficientnet", "xception", "ensemble"
    USE_FACE_DETECTION = True    # Extract and analyze only face regions
    USE_TEMPORAL_MODEL = False   # Use LSTM for temporal analysis (slower but better for videos)
    
    # Training settings
    USE_CLASS_WEIGHTS = True     # Balance classes if dataset is imbalanced
    USE_MIXED_PRECISION = True   # Faster training on modern GPUs
    
    # Create directories
    @staticmethod
    def create_directories():
        for path in [Config.LOGS_DIR, Config.RESULTS_DIR, 
                     Config.FRAMES_OUTPUT_PATH, os.path.dirname(Config.MODEL_SAVE_PATH)]:
            os.makedirs(path, exist_ok=True)

# Initialize configuration
config = Config()
config.create_directories()

# Enable mixed precision training for faster computation
if config.USE_MIXED_PRECISION:
    # The 'mixed_float16' policy automatically handles the dtype for the final output layer.
    tf.keras.mixed_precision.set_global_policy('mixed_float16') 

# ==================== UTILITY FUNCTIONS (No changes needed) ====================

def setup_gpu():
    """
    Configure GPU settings for optimal performance.
    Allows memory growth to prevent TensorFlow from allocating all GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU acceleration enabled: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected. Training will use CPU (slower)")

def calculate_class_weights(train_generator):
    """
    Calculate class weights to handle imbalanced datasets.
    Gives more importance to minority class during training.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(train_generator.classes)
    weights = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
    class_weights = dict(zip(classes, weights))
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(train_generator.classes, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples")
    print(f"Class weights: {class_weights}")
    
    return class_weights

# ==================== FACE DETECTION (No changes needed) ====================

class FaceExtractor:
    """
    Extracts face regions from images using MTCNN face detector.
    This focuses the model on facial features where deepfake artifacts are most visible.
    """
    def __init__(self):
        if MTCNN_AVAILABLE:
            self.detector = MTCNN()
            self.enabled = True
        else:
            self.enabled = False
            print("⚠ Face detection disabled (MTCNN not available)")
    
    def extract_face(self, image, target_size=(224, 224), margin=20):
        """
        Detect and extract the largest face from an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            target_size: Output dimensions
            margin: Pixels to add around detected face
            
        Returns:
            Extracted and resized face image, or original if no face detected
        """
        if not self.enabled:
            return cv2.resize(image, target_size)
        
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(rgb_image)
        
        if not faces:
            # No face detected, return resized original
            return cv2.resize(image, target_size)
        
        # Get largest face (by area)
        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # Extract face region
        face = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size)
        
        return face_resized

# ==================== DATA PREPARATION (No changes needed) ====================

def create_advanced_data_generators(use_face_detection=False):
    """
    Create data generators with advanced augmentation techniques.
    
    Augmentation helps the model generalize better by creating variations of training data:
    - Rotation: Simulates different head poses
    - Shifts: Handles slight misalignments
    - Zoom: Varies face sizes
    - Flips: Doubles effective dataset size
    - Brightness/Contrast: Handles different lighting conditions
    
    Args:
        use_face_detection: If True, extract faces before training
        
    Returns:
        train_generator, val_generator, test_generator
    """
    
    # Preprocessing function for face extraction
    def preprocess_face(image):
        if use_face_detection:
            # Re-initialize or use a globally accessible instance if needed
            face_extractor = FaceExtractor() 
            # Convert normalized image back to 0-255 range for face detection
            img_uint8 = (image * 255).astype(np.uint8)
            face = face_extractor.extract_face(img_uint8)
            return face.astype(np.float32) / 255.0
        return image
    
    # Training augmentation (strong augmentation for better generalization)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=25,           # Rotate up to 25 degrees
        width_shift_range=0.2,       # Shift horizontally up to 20%
        height_shift_range=0.2,      # Shift vertically up to 20%
        horizontal_flip=True,        # Random horizontal flips
        zoom_range=0.2,              # Zoom in/out up to 20%
        shear_range=0.15,            # Shear transformation
        brightness_range=[0.7, 1.3], # Vary brightness
        fill_mode='nearest',         # Fill strategy for empty pixels
        preprocessing_function=preprocess_face if use_face_detection else None
    )
    
    # Test augmentation (only rescaling, no random transformations)
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_face if use_face_detection else None
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42
    )
    
    val_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# ==================== MODEL ARCHITECTURES (Fixed 'drop_connect_rate' for EfficientNet) ====================

def create_custom_cnn_v2():
    """
    Enhanced Custom CNN with modern architectural improvements.
    Total parameters: ~8M (lightweight and fast)
    """
    model = Sequential([
        Input(shape=(224, 224, 3)),
        
        # Block 1: Initial feature extraction
        Conv2D(32, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Block 2: Detect basic patterns
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Block 3: Detect complex patterns
        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        # Block 4: High-level features
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        # Block 5: Abstract representations
        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),
        Dropout(0.4),
        
        # Classification head
        GlobalAveragePooling2D(),  # Better than Flatten for preventing overfitting
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid', dtype='float32')  # Binary output. Ensure float32 for final layer
    ])
    
    return model

def create_efficientnet_model():
    """
    Transfer Learning with EfficientNetB0:
    
    Fixed: Removed the unsupported 'drop_connect_rate' argument from EfficientNetB0.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        # Removed: drop_connect_rate=0.2 (Not supported directly in tf.keras.applications)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

def create_xception_model():
    """
    Transfer Learning with Xception.
    """
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

def create_temporal_model(sequence_length=10):
    """
    Temporal CNN-LSTM model for video sequence analysis.
    """
    # Frame input
    frame_input = Input(shape=(sequence_length, 224, 224, 3))
    
    # CNN feature extractor (applied to each frame)
    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        GlobalAveragePooling2D()
    ])
    
    # Apply CNN to each frame in sequence
    x = TimeDistributed(cnn)(frame_input)
    
    # LSTM for temporal analysis
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    
    # Classification
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x) # Ensure float32 for final layer
    
    model = Model(inputs=frame_input, outputs=outputs)
    
    return model

def build_model(model_type="efficientnet"):
    """
    Factory function to create the specified model architecture.
    """
    print(f"\n{'='*70}")
    print(f"Building {model_type.upper()} model...")
    print(f"{'='*70}")
    
    if model_type == "custom":
        model = create_custom_cnn_v2()
        base_model = None
    elif model_type == "efficientnet":
        model, base_model = create_efficientnet_model()
    elif model_type == "xception":
        model, base_model = create_xception_model()
    elif model_type == "temporal":
        # Note: Temporal model requires special data generator setup not shown here
        model = create_temporal_model(config.SEQUENCE_LENGTH)
        base_model = None
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, base_model

# ==================== TRAINING (No changes needed) ====================

def get_callbacks(model_name):
    """
    Create training callbacks for better training control.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f'{model_name}_{timestamp}'),
            histogram_freq=1,
            write_graph=True
        ),
        CSVLogger(
            os.path.join(config.RESULTS_DIR, f'{model_name}_training.csv'),
            append=True
        )
    ]
    
    return callbacks

def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer, loss, and metrics.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

def train_model(model, train_gen, val_gen, model_type="efficientnet", 
                base_model=None, class_weights=None):
    """
    Complete training pipeline with two-phase approach for transfer learning.
    """
    
    print("\n" + "="*70)
    print("PHASE 1: TRAINING CLASSIFICATION HEAD")
    print("="*70)
    
    # Compile with initial learning rate
    compile_model(model, config.LEARNING_RATE)
    
    # Display model architecture
    model.summary()
    
    # Calculate total parameters
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    
    # Get callbacks
    callbacks = get_callbacks(model_type)
    
    # Train phase 1
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Fine-tuning phase for transfer learning models
    if base_model is not None and model_type in ["efficientnet", "xception"]:
        print("\n" + "="*70)
        print("PHASE 2: FINE-TUNING ENTIRE NETWORK")
        print("="*70)
        
        # Unfreeze base model
        base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers
        # This prevents overfitting while allowing adaptation
        fine_tune_at = len(base_model.layers) // 2
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        print(f"Unfrozen layers: {sum([1 for l in base_model.layers if l.trainable])}/{len(base_model.layers)}")
        
        # Recompile with lower learning rate
        compile_model(model, config.LEARNING_RATE / 10)
        
        # Continue training
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,  # Fewer epochs for fine-tuning
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            initial_epoch=len(history.history['loss'])
        )
        
        # Combine histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    return history

# ==================== EVALUATION (The evaluation section was cut off but the fix is applied to the imported modules) ====================

def plot_training_history(history, model_name):
    # ... (rest of the plotting functions remain the same)
    pass
# ... (rest of the code follows)