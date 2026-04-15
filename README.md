# AI-Driven Optimisation of Radar Absorbing Materials for Stealth Aircraft Skin

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayant1309/ram_optimisation/blob/main/RAM_Optimisation.ipynb)

## Project Overview

This project implements a comprehensive AI-driven pipeline for the optimization of radar absorbing materials (RAM) for stealth aircraft applications. The system leverages the Materials Project database and advanced machine learning techniques to predict and identify promising materials with optimal electromagnetic absorption properties. Using a combination of classical regression, classification, and deep neural networks, the pipeline systematically processes materials data, engineers meaningful features, and evaluates candidate materials based on their dielectric properties.

The project integrates materials science domain knowledge with state-of-the-art machine learning to accelerate the discovery of novel radar absorbing materials that can enhance stealth capabilities in aerospace applications.

## Pipeline Architecture

### Data Flow

1. **Data Acquisition** - Query Materials Project API for dielectric materials and extract properties (dielectric constants, band gap, density, crystal structure)
2. **Feature Engineering** - Compute materials science features using pymatgen (composition analysis) and mendeleev (element properties)
3. **EDA & Visualization** - Exploratory analysis with correlation heatmaps, PCA, t-SNE, and K-Means clustering
4. **Baseline Regression Models** - Predict continuous dielectric response (e_total) using Linear, Polynomial, SVR, and Random Forest
5. **Classification Models** - Binary classification (high/low absorber) using Logistic Regression, SVM, and Decision Trees
6. **Deep Neural Network** - Multi-layer perceptron using PyTorch for binary classification with K-Fold cross-validation
7. **Evaluation & Comparison** - Model comparison, feature importance analysis, and top candidate material ranking

### Dual ML Approach

This pipeline uniquely combines **regression** and **classification** to provide both quantitative predictions and actionable recommendations:

- **Regression (Step 4)**: Predicts continuous dielectric response values, useful for understanding the full spectrum of material performance
- **Classification (Steps 5-6)**: Identifies high-potential candidates by splitting materials into high/low absorbers using the median e_total value
- **Benefits**: Regression provides precise predictions; classification accelerates experimental validation by flagging the most promising materials

## Tech Stack

- **Core ML**: Python 3.8+, scikit-learn 1.5+, PyTorch 2.3+
- **Materials Science**: pymatgen 2024.11+, mendeleev, mp-api
- **Visualization**: matplotlib, seaborn
- **Development**: Supports both local machine and Google Colab
- **Orchestration**: Automated pipeline execution via main.py

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-green.svg)
![pymatgen](https://img.shields.io/badge/pymatgen-2024.11+-orange.svg)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-yellow.svg)

## Quick Start

### ⚡ Get Your Materials Project API Key

1. Visit https://materialsproject.org/api
2. Register for a free account
3. Copy your API key from your dashboard

### 🔐 Set Up API Key (Choose ONE method)

#### Method 1: Environment Variable (Recommended for Command Line)
```bash
export MP_API_KEY="your_api_key_here"
```

#### Method 2: .env File (Recommended for IDEs)
```bash
cp .env.example .env
# Edit .env and add your API key on line 1
```

#### Method 3: Direct Config (Quick but Less Secure)
Edit `config.py` and replace `YOUR_KEY_HERE` with your API key (do not commit this change!)

### ▶️ Run the Pipeline

```bash
python main.py
```

This executes all 7 steps automatically. Individual steps can also be run directly:

```bash
# Run a specific step
python 1_data_acquisition.py
python 2_feature_engineering.py
# ... etc
```

### 🚀 Get Started with Google Colab

Click the "Open In Colab" badge at the top of this README. When the notebook opens:

1. Run the first cell to install dependencies  
2. Enter your Materials Project API key in the second cell
3. Run all cells to execute the pipeline
4. Check `MyDrive/ram_optimisation/` for all outputs

*All outputs and trained models will be automatically saved to your Google Drive in `MyDrive/ram_optimisation/`*

## Local Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

```bash
git clone https://github.com/jayant1309/ram_optimisation.git
cd ram_optimisation
pip install -r requirements.txt
```

### ⚠️ API Key Security

**IMPORTANT**: See [SECURITY.md](SECURITY.md) for secure API key handling best practices.

Never commit API keys to Git. This repository provides three secure methods:
- **Environment variables** (recommended for production)
- **`.env` file** (recommended for development, automatically ignored by Git)
- **Direct config edit** (quick testing only, requires caution)

Get your free Materials Project API key from: https://materialsproject.org/open

### Local Machine

```bash
git clone https://github.com/jayant1309/ram_optimisation.git
cd ram_optimisation
pip install -r requirements.txt
```

Set your Materials Project API key (choose one method):

**Method 1 (Recommended) - Environment variable:**
```bash
export MP_API_KEY="your_api_key_here"
```

**Method 2 - .env file:**
```bash
cp .env.example .env
# Edit .env and add your API key
```

**Method 3 - Direct edit:**
Edit `config.py` and replace `YOUR_KEY_HERE` with your API key (do not commit this change).

Get your free API key from: https://materialsproject.org/open

Run the pipeline:
```bash
python main.py
```

### Google Colab

Click the "Open In Colab" badge above. When the notebook opens:

1. Run the first cell to install dependencies
2. Enter your Materials Project API key in the second cell
3. Run all subsequent cells to execute the pipeline

All outputs and trained models will be automatically saved to your Google Drive in `MyDrive/ram_optimisation/`.

## Results

After running the pipeline, outputs are organized as follows:

### 📊 Visualizations (`plots/` directory)
- **EDA**: Target distributions, correlation heatmaps, PCA/t-SNE projections, K-means clustering
- **Regression**: Actual vs predicted plots for all baseline models  
- **Classification**: Confusion matrices with accuracy/precision/recall/F1 metrics
- **Model Comparison**: Bar plots comparing Accuracy and F1 scores across all models
- **Feature Importance**: Random Forest feature importance rankings

### 💾 Data & Models (`data/` directory)
- **Processed Data**: raw_materials.csv → features.csv → clustered_features.csv
- **Trained Models**: 
  - Regression: Linear, Polynomial, SVR, Random Forest (.pkl)
  - Classification: Logistic Regression, SVM, Decision Tree (.pkl)
  - Deep Learning: DNN model state dict (.pth)
- **Results**: Model metrics, predictions, and evaluation summaries (.pkl)
- **Preprocessing**: Saved scalers, imputers, and encoders for reproducibility

### 🎯 Key Outputs
- **Model Performance Table**: Side-by-side comparison of all models
- **Top Candidate Materials**: Ranked list of high-potential RAM candidates
- **Feature Engineering Pipeline**: Reusable preprocessing for new materials

## Troubleshooting

### Import Errors
If you encounter import errors, ensure all requirements are installed:

```bash
pip install -r requirements.txt
```

### API Connection Issues
- Verify your API key is correctly set
- Check internet connection
- Materials Project API may have rate limits for free tier

### Memory Issues (Large Datasets)
If you run out of memory with large material datasets:
- Reduce `nsites` filter in `1_data_acquisition.py`
- Run on Google Colab with GPU acceleration
- Increase batch size in DNN training

### Colab-Specific Issues
- Ensure drive is mounted before running pipeline
- Check that you're using the correct path in Colab: `/content/drive/MyDrive/ram_optimisation/`
- Run `!pip install -r requirements.txt` in first cell

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU available
```
If False, Colab: Runtime → Change runtime type → GPU

## Project Structure

```
ram_optimisation/
├── 1_data_acquisition.py      # Fetch materials from Materials Project
├── 2_feature_engineering.py   # Compute materials science features
├── 3_eda_and_viz.py           # Exploratory data analysis
├── 4_regression_models.py     # Baseline regression models
├── 5_classification.py        # Classification models
├── 6_deep_learning.py         # Deep neural network
├── 7_evaluation.py            # Model evaluation and ranking
├── main.py                    # Pipeline orchestrator
├── config.py                  # Hyperparameters and settings
├── requirements.txt           # Python dependencies
├── .env.example              # API key template
├── SECURITY.md               # Security best practices
├── data/                     # Processed datasets and models
├── plots/                    # Generated visualizations
└── README.md                 # This file
```

## Next Steps & Usage

After running the pipeline, you can:

1. **Analyze Results**: Review `plots/` for visual insights and `models/` for trained weights
2. **Predict New Materials**: Use saved models (`*.pkl`, `*.pth`) to predict properties of new material compositions
3. **Tune Hyperparameters**: Modify values in `config.py` and re-run specific steps
4. **Extend the Pipeline**: Add new models, features, or evaluation metrics
5. **Explore Candidates**: Use top-ranked materials from Step 7 for experimental validation

## References

### Materials & Data
- [Materials Project](https://materialsproject.org/) - Materials database and API documentation
- [Materials Project API](https://materialsproject.org/api) - REST API for materials data
- [pymatgen Documentation](https://pymatgen.org/) - Materials analysis library
- [mendeleev Documentation](https://mendeleev.readthedocs.io/) - Periodic table and element properties
- [mp-api Python Client](https://github.com/materialsproject/api) - Official Python client

### Machine Learning
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework
- [scikit-learn Documentation](https://scikit-learn.org/) - Machine learning toolkit
- [NumPy Documentation](https://numpy.org/doc/) - Numerical computing
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Data manipulation

### Visualization
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) - Plotting library
- [Seaborn Documentation](https://seaborn.pydata.org/) - Statistical data visualization

## License

This project is created for educational and research purposes in AI-driven materials science.

## Contributors

Built with Claude Code by Anthropic - AI-assisted development for scientific computing applications.

**Key Contributions:**
- Materials Project API integration
- pymatgen feature engineering pipeline
- Multi-model ensemble (Regression + Classification)
- Automated evaluation and candidate ranking system
