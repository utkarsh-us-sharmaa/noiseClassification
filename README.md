## Noise Classification System: Air Conditioner vs. Copy Machine

This repository contains the implementation of a deep learning-based noise classification system that distinguishes between air conditioner and copy machine sounds. The project adapts a Convolutional Neural Network (CNN) architecture originally designed for bird sound classification to the task of mechanical noise classification using the Microsoft Scalable Noisy Speech Dataset (MS-SNSD).

Environmental noise classification has significant applications in smart buildings, industrial monitoring, and acoustic scene analysis. This project demonstrates how deep learning techniques can effectively differentiate between common noise sources with high accuracy, even with limited training data.

# Technical Overview
Dataset
The Microsoft Scalable Noisy Speech Dataset (MS-SNSD) provides a collection of environmental noise recordings, including air conditioners and copy machines. The dataset includes:

High-quality audio recordings at 16 kHz sampling rate

Multiple recordings per noise type

Varying acoustic conditions and device types

For this project, we focus exclusively on two noise classes:

Air conditioner noise

Copy machine noise

# Feature Extraction
Audio signals are converted to Mel-Frequency Cepstral Coefficients (MFCCs), which provide a compact representation of the spectral characteristics of sound:

def extract_and_aggregate_mfcc(audio_file, sr=16000, n_mfcc=40):
    Load audio file
    audio, sample_rate = librosa.load(audio_file, sr=sr)
    Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    Aggregate features by taking mean across time dimension
    aggregated_mfccs = np.mean(mfccs, axis=1)
    Convert to tensor
    mfccs_tensor = tf.convert_to_tensor(aggregated_mfccs, dtype=tf.float32)
    return mfccs_tensor

MFCCs are particularly effective for audio classification tasks as they model the human auditory system's response to sound, emphasizing perceptually significant aspects while de-emphasizing less relevant information.

# Data Augmentation
To overcome the limited size of the dataset, we implemented several data augmentation techniques:

Time stretching: Slowing down or speeding up the audio without affecting pitch

Pitch shifting: Raising or lowering the pitch without affecting duration

Noise addition: Adding random Gaussian noise to increase robustness

These techniques effectively expanded our training dataset by a factor of 6, which significantly improved model generalization and performance.

Model Architecture
We adapted a CNN architecture with the following structure:


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Input                       [(None, 40, 1)]           0         
                                                                 
 Conv1D (64 filters)         (None, 38, 64)            256       
                                                                 
 BatchNormalization          (None, 38, 64)            256       
                                                                 
 MaxPooling1D                (None, 19, 64)            0         
                                                                 
 Conv1D (128 filters)        (None, 17, 128)           24704     
                                                                 
 BatchNormalization          (None, 17, 128)           512       
                                                                 
 MaxPooling1D                (None, 9, 128)            0         
                                                                 
 Conv1D (256 filters)        (None, 7, 256)            98560     
                                                                 
 BatchNormalization          (None, 7, 256)            1024      
                                                                 
 MaxPooling1D                (None, 4, 256)            0         
                                                                 
 Flatten                     (None, 1024)              0         
                                                                 
 Dense (256 units)           (None, 256)               262400    
                                                                 
 Dropout (0.3)               (None, 256)               0         
                                                                 
 Dense (128 units)           (None, 128)               32896     
                                                                 
 Dropout (0.3)               (None, 128)               0         
                                                                 
 Dense (2 units)             (None, 2)                 258       
                                                                 
=================================================================


# Key architectural elements:

1D Convolutional layers: Capture temporal and spectral patterns

BatchNormalization: Stabilizes training and improves convergence

MaxPooling: Reduces dimensionality while preserving important features

Dropout and L2 regularization: Prevent overfitting

Dense layers: Learn high-level representations for classification

This architecture was inspired by successful models in audio classification tasks but adapted for our binary classification problem.

Performance Results
Our model achieved excellent performance on the test dataset:

Classification Metrics
![image](https://github.com/user-attachments/assets/438bd981-ae0b-4e84-b9ce-055cee64d16f)


# Confusion Matrix
The confusion matrix shows:

9 correctly classified Air Conditioner samples

1 Air Conditioner misclassified as Copy Machine

2 correctly classified Copy Machine samples

0 Copy Machine misclassified as Air Conditioner

# Training History
The training process showed:

Rapid convergence to high accuracy (>95% within the first 50 epochs)

Steady decrease in loss over training

No significant gap between training and validation metrics, indicating good generalization

Implementation Details
Data Pipeline
We implemented an optimized data pipeline using TensorFlow's data API:
//Optimized data pipeline
train_ds = train_ds.cache().shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.cache().shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

This approach:

Caches data in memory after loading

Shuffles data to improve training stability

Prefetches data to overlap computation and I/O operations

# Model Training
The model was trained with:

Adam optimizer with learning rate 1e-4

Sparse categorical crossentropy loss

100 epochs (early stopping was available but not triggered)

Batch size of 32


# Challenges and Solutions
Limited Dataset Size
Challenge: MS-SNSD provides only about 10 recordings per noise type.

# Solution: Extensive data augmentation techniques were implemented, creating 5 additional variations of each recording through time stretching, pitch shifting, and noise addition.

# Class Imbalance
Challenge: Uneven distribution of samples between classes (visible in the test set with 10 air conditioner vs. 2 copy machine samples).

Solution: Performance was evaluated using precision, recall, and F1-score to account for class imbalance. Future work could implement class weighting in the loss function.

# Feature Representation
Challenge: Determining the optimal audio representation for mechanical noise classification.

Solution: MFCCs were chosen as they capture spectral characteristics effectively. The temporal aggregation (taking the mean across time) provided a fixed-length representation regardless of audio duration.

# Conclusion
This project successfully demonstrates that a CNN-based approach can effectively classify mechanical noise types with high accuracy (92%) even with limited training data. The combination of MFCCs for feature extraction, data augmentation techniques, and a carefully designed CNN architecture provides a robust solution to the noise classification problem.

The model shows excellent performance for air conditioner classification (F1-score of 0.95) and good performance for copy machine classification (F1-score of 0.80), with the lower performance likely due to the limited number of copy machine samples in the test set.

# Future Work
Several directions for future improvement include:

Enhanced Feature Extraction: Experimenting with additional audio features such as spectral contrast, chroma features, and Delta-MFCCs to capture more detailed acoustic characteristics.

Advanced Architectures: Implementing attention mechanisms or recurrent neural networks (RNNs) to better capture temporal dependencies in the audio signals.

Transfer Learning: Leveraging pre-trained audio models (e.g., VGGish, YAMNet) and fine-tuning them for noise classification to benefit from knowledge learned on larger datasets.

Multi-Class Extension: Expanding the system to classify additional noise types available in the MS-SNSD dataset, such as babble, music, or traffic.

Real-Time Processing: Optimizing the inference pipeline for low-latency applications in embedded systems or edge devices.

Explainable AI: Implementing visualization techniques to understand which spectral-temporal regions contribute most to classification decisions.

# References:
Davis, S., & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366. Salamon, J., & Bello, J. P. (2017). "Deep convolutional neural networks and data augmentation for environmental sound classification." IEEE Signal Processing Letters, 24(3), 279-283. Hershey, S., Chaudhuri, S., Ellis, D. P., Gemmeke, J. F., Jansen, A., Moore, R. C., ... & Wilson, K. (2017). "CNN architectures for large-scale audio classification." In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 131-135). Gemmeke, J. F., Ellis, D. P., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., ... & Ritter, M. (2017). "Audio set: An ontology and human-labeled dataset for audio events." In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 776-780). Pons, J., & Serra, X. (2019). "Randomly weighted CNNs for (music) audio classification." In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 336-340). Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880-2894. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking model scaling for convolutional neural networks." In International Conference on Machine Learning (pp. 6105-6114). Mishra, S., Sturm, B. L., & Dixon, S. (2018). "Local Interpretable Model-Agnostic Explanations for Music Content Analysis." In ISMIR (pp. 537-543).


