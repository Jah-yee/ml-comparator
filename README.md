# ML Model Comparator

<p align="center">
  <img src="https://img.shields.io/badge/Machine-Learning-9%20Models-blue" alt="ML">
  <img src="https://img.shields.io/badge/Parameters-Experiment-orange" alt="Experiment">
</p>

Interactive platform for comparing machine learning models and their hyperparameters.

## Features

- **9 Models**: SVM, Logistic Regression, Decision Tree
- **Parameter Grid**: Multiple C values, tree depths
- **Real Training**: Actual sklearn training
- **Metrics**: Train/Test accuracy + overfitting detection
- **Visualization**: Comparative charts

## Models

| Model | Parameters |
|-------|------------|
| SVM (RBF) | C=0.1, 1, 10 |
| SVM (Linear) | C=1 |
| Logistic Regression | C=0.1, 1 |
| Decision Tree | depth=3, 5, 10 |

## Quick Start

```bash
pip install -r requirements.txt
python trainer.py
# Open index.html in browser
```

## Demo

🔗 **[ml-comparator.vercel.app](https://ml-comparator.vercel.app)**

## Tech Stack

- Python, scikit-learn, matplotlib
- HTML5, Tailwind CSS, Canvas API
- Vercel
