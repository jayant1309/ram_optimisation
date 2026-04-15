# AI-Driven Optimisation of Radar Absorbing Materials for Stealth Aircraft Skin

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayant1309/ram_optimisation/blob/main/RAM_Optimisation.ipynb)

## Project Overview

This project implements a comprehensive AI-driven pipeline for the optimization of radar absorbing materials (RAM) for stealth aircraft applications. The system leverages the Materials Project database and advanced machine learning techniques to predict and identify promising materials with optimal electromagnetic absorption properties. Using a combination of classical regression, classification, and deep neural networks, the pipeline systematically processes materials data, engineers meaningful features, and evaluates candidate materials based on their dielectric properties.

The project integrates materials science domain knowledge with state-of-the-art machine learning to accelerate the discovery of novel radar absorbing materials that can enhance stealth capabilities in aerospace applications.

## Pipeline Architecture

1. **Data Acquisition** - Query Materials Project API for dielectric materials
2. **Feature Engineering** - Extract materials science features using pymatgen and mendeleev
3. **EDA & Visualization** - Exploratory analysis with PCA, t-SNE, and clustering
4. **Baseline Regression** - Linear, Polynomial, SVR, and Random Forest models
5. **Classification** - High/low absorber classification using Logistic Regression, SVM, Decision Trees
6. **Deep Neural Network** - PyTorch-based deep learning with K-Fold validation
7. **Evaluation & Comparison** - Model comparison, feature importance, and candidate ranking

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-green.svg)
![pymatgen](https://img.shields.io/badge/pymatgen-2024.11+-orange.svg)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-yellow.svg)

## Setup

### ⚠️ API Key Security

**IMPORTANT**: See [SECURITY.md](SECURITY.md) for secure API key handling.

Never commit API keys to Git. This repository uses:
- Environment variables (recommended)
- `.env` file (not tracked by Git)
- Direct config edit (less secure)

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

After running the pipeline, check the `plots/` folder for visualizations and the `data/` folder for processed data, trained models, and evaluation metrics.

## References

- [Materials Project](https://materialsproject.org/)
- [pymatgen Documentation](https://pymatgen.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## License

This project is created for educational and research purposes in AI-driven materials science.
