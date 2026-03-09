# ML Model Comparator

<p align="center">
  <img src="https://img.shields.io/badge/Machine-Learning-9%20Models-blue" alt="ML">
  <img src="https://img.shields.io/badge/Compare-SVM--LR--Tree-green" alt="Compare">
  <img src="https://img.shields.io/badge/Parameters-C--Depth-orange" alt="Params">
</p>

Interactive platform for comparing machine learning models and their parameters.

## Features

- **9 Models Compared**: SVM, Logistic Regression, Decision Tree
- **Parameter Grid**: Multiple C values, tree depths
- **Real Training**: Actual sklearn training on your data
- **Accuracy Metrics**: Train/Test accuracy + overfitting analysis
- **Visualization**: Comparative charts

## Models Included

| Model | Parameters |
|-------|------------|
| SVM (RBF) | C=0.1, 1, 10 |
| SVM (Linear) | C=1 |
| Logistic Regression | C=0.1, 1 |
| Decision Tree | depth=3, 5, 10 |

## Quick Start

```bash
# Local
pip install -r requirements.txt
python trainer.py
# Open index.html in browser
```

## Online Demo

🔗 **[ml-comparator.vercel.app](https://ml-comparator.vercel.app)**

## How It Works

1. Select dataset type (moons, circles, linear)
2. Adjust noise level and sample size
3. Click "Run Experiment"
4. Compare 9 models with different parameters
5. Analyze accuracy and overfitting

## Tech Stack

- **Backend**: Python, scikit-learn
- **Frontend**: HTML5, Tailwind CSS, Canvas
- **Deployment**: Vercel
