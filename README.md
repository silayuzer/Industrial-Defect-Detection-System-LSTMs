# Industrial Defect Detection & Root Cause Analysis (RNN/LSTM)

## 📌 Project Overview
This project implements a Deep Learning framework to detect and localize defects in a candy production line using multi-sensor time series data. The system is designed to handle 50,000 samples of synthetic industrial data, addressing both classification and explainability requirements.

The core objective is to move beyond "black-box" AI by providing **Root Cause Analysis (RCA)**, identifying not just *if* a defect occurred, but *where* and *why*.

## 🧠 Model Architecture: Seq2Seq Autoencoder
The system utilizes a **Sequence-to-Sequence (Seq2Seq)** blueprint optimized for temporal data:
* **Feature Embedding:** Raw sensor signals are projected into a 32-dimensional feature space using a `TimeDistributed` Dense layer to capture complex non-linear interactions.
* **Encoder:** A Stacked Bidirectional LSTM compresses the multidimensional input sequences into a latent representation.
* **Decoder:** A Bidirectional LSTM "unfolds" this representation to reconstruct the original signal.



## 🔬 Root Cause Analysis (Explanation)
The project satisfies the requirement for "Explanation" through **Reconstruction Error Distribution**:
* **Mechanism:** The model is trained to reconstruct "normal" sinusoidal production patterns. When it encounters an anomaly (e.g., sensor dropout, spike, or level shift), the reconstruction fails.
* **Localization:** The difference between the input and output (Mean Squared Error) is plotted as a **Temporal Error Map** (Red areas in the plots).
* **Proof:** These error spikes provide mathematical evidence of the defect's exact timestamp and magnitude, allowing for precise root cause localization.

## 🛠️ Technical Features
- **Sequence Handling:** Implemented Post-Padding for variable-length sequences (40-60 timesteps).
- **Gradient Stability:** Used **Gradient Normalization (`clipnorm=1.0`)** and **Orthogonal Initialization** to stabilize training and prevent exploding gradients in deep recurrent layers.
- **Multi-Task Learning:** Simultaneously performs multi-label classification and signal reconstruction to identify concurrent defects.
- **Extra Task:** Successfully detects correlated anomalies occurring across multiple sensors simultaneously using the temporal context of Bidirectional RNNs.

## 📊 Results
The model achieves high accuracy in identifying defect types while providing clear visual localization of anomalies on the time axis. This makes it a suitable tool for industrial predictive maintenance.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `ProjectData.ipynb` in Google Colab or Jupyter Notebook and run all cells.
