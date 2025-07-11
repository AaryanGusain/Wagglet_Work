# WaggleNet: Bee Colony Queen Status Classification

A machine learning project for analyzing bee colony audio recordings to classify queen status using acoustic features. This project uses audio signal processing and various ML algorithms to distinguish between "Queen Right" (QR) and "Queen Less" (QL) bee colonies.

## 🐝 Project Overview

This project analyzes bee hive audio recordings to automatically detect the presence or absence of a queen bee in the colony. The classification is based on acoustic features extracted from WAV audio files, leveraging the fact that queenless colonies produce distinctly different acoustic patterns compared to colonies with a queen.

## 📊 Features

- **Audio Feature Extraction**: Mel-spectrograms, MFCCs (Mel-frequency Cepstral Coefficients), and Delta MFCCs
- **Multiple ML Models**: Comparison of 6 different classification algorithms
- **Caching System**: Efficient feature caching to avoid recomputation
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## 🛠 Technologies Used

- **Python 3.x**
- **Audio Processing**: librosa
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib
- **Caching**: joblib

## 🏗 Project Structure

```
wagglenet/
├── data_analysis.py          # Main ML pipeline and model comparison
├── data_format.py           # Data preprocessing utilities
├── audio_file_labels_asof.csv  # Labeled dataset with audio paths and queen status
├── audio_file_labels.csv    # Additional labeled data
├── inspections_2021.csv     # Hive inspection records for 2021
├── inspections_2022.csv     # Hive inspection records for 2022
├── requirements.txt         # Python dependencies
└── README.md               # This file

# Note: Audio files and cache files are excluded from Git due to size
```

## 🎯 Machine Learning Models

The project compares 6 different classification algorithms:

1. **K-Nearest Neighbors (KNN)**
2. **Random Forest**
3. **XGBoost**
4. **Logistic Regression**
5. **LightGBM**
6. **Multi-Layer Perceptron (MLP)**

### Model Performance
Based on test results, the models achieve the following accuracies:
- **LightGBM**: 98.4% (Best performing)
- **Logistic Regression**: 96.8%
- **XGBoost**: 96.8%
- **MLP Neural Network**: 95.5%
- **KNN**: 94.6%
- **Random Forest**: 93.3%

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AaryanGusain/Wagglet_Work.git
   cd Wagglet_Work
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Data Requirements

Due to file size limitations, audio files are not included in this repository. To run the analysis, you'll need:

1. **Audio Files**: WAV recordings from bee hives (organized in `audio_2021_chunk_*` directories)
2. **Labels**: CSV files with audio paths and corresponding queen status labels

The expected data structure:
```
audio_2021_chunk_1/
├── 12-08-2021_00h45_Hive-6.wav
├── 12-08-2021_01h45_HIVE-3693.WAV
└── ...
```

## 🚀 Usage

1. **Ensure your data is properly structured** with audio files and labels
2. **Run the main analysis**:
   ```bash
   python data_analysis.py
   ```

The script will:
- Load and preprocess the labeled data
- Extract audio features (or load from cache)
- Train multiple ML models
- Display performance metrics and visualizations

## 📈 Feature Engineering

The project extracts the following acoustic features:

- **Mel-spectrograms**: Frequency representation of audio signals
- **MFCCs**: Cepstral coefficients capturing spectral characteristics
- **Delta MFCCs**: Temporal changes in MFCCs
- **Statistical aggregations**: Mean and standard deviation of features

## 📊 Data Description

- **Total samples**: ~1,562 audio recordings
- **Features**: Queen status (QR/QL), hive ID, timestamps, colony metrics
- **Time period**: 2021 bee season recordings
- **Binary classification**: Queen Right (QR) vs Queen Less (QL)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Aaryan Gusain** - Initial work and development

## 🙏 Acknowledgments

- Bee research community for providing insights into bee colony acoustics
- Open-source contributors of librosa, scikit-learn, and other libraries used

## 📞 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or contact the repository owner.

---

*This project contributes to the field of precision apiculture and automated bee colony monitoring.*
