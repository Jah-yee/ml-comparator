"""
Machine Learning Comparator - Core Training Module
Compare multiple ML models with different parameters
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

class MLComparator:
    def __init__(self):
        self.results = {}
        self.scaler = StandardScaler()
        
    def generate_dataset(self, name='moons', n_samples=500, noise=0.3):
        """Generate training and test datasets"""
        generators = {
            'moons': make_moons,
            'circles': make_circles,
            'linear': make_classification
        }
        
        if name == 'linear':
            X, y = generators[name](n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_informative=2, random_state=42, flip_y=0.01)
        else:
            X, y = generators[name](n_samples=n_samples, noise=noise, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_models(self):
        """Define models to compare"""
        return {
            'svm_rbf_c1': SVC(kernel='rbf', C=1.0, random_state=42),
            'svm_rbf_c10': SVC(kernel='rbf', C=10.0, random_state=42),
            'svm_rbf_c01': SVC(kernel='rbf', C=0.1, random_state=42),
            'svm_linear_c1': SVC(kernel='linear', C=1.0, random_state=42),
            'logistic_l2_c1': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42),
            'logistic_l2_c01': LogisticRegression(penalty='l2', C=0.1, max_iter=1000, random_state=42),
            'tree_depth3': DecisionTreeClassifier(max_depth=3, random_state=42),
            'tree_depth5': DecisionTreeClassifier(max_depth=5, random_state=42),
            'tree_depth10': DecisionTreeClassifier(max_depth=10, random_state=42),
        }
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and get results"""
        models = self.get_models()
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting': train_acc - test_acc
            }
        
        return results
    
    def create_comparison_chart(self, results, dataset_name):
        """Create comparison visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        names = list(results.keys())
        train_accs = [results[n]['train_accuracy'] for n in names]
        test_accs = [results[n]['test_accuracy'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        # Accuracy comparison
        axes[0].bar(x - width/2, train_accs, width, label='Train', color='#3b82f6')
        axes[0].bar(x + width/2, test_accs, width, label='Test', color='#10b981')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Model Accuracy Comparison - {dataset_name}')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        
        # Overfitting analysis
        overfits = [results[n]['overfitting'] for n in names]
        colors = ['#ef4444' if o > 0.1 else '#10b981' for o in overfits]
        axes[1].bar(names, overfits, color=colors)
        axes[1].set_ylabel('Train - Test')
        axes[1].set_title('Overfitting Analysis')
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        from io import BytesIO
        import base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img
    
    def run_experiment(self, dataset='moons', noise=0.3):
        """Run complete experiment"""
        X_train, X_test, y_train, y_test = self.generate_dataset(dataset, noise=noise)
        results = self.train_and_evaluate(X_train, X_test, y_train, y_test)
        chart = self.create_comparison_chart(results, dataset)
        
        return {
            'dataset': dataset,
            'samples': len(X_train),
            'results': results,
            'chart': f'data:image/png;base64,{chart}'
        }

if __name__ == '__main__':
    comp = MLComparator()
    result = comp.run_experiment('moons')
    print("Experiment completed!")
    print(f"Best test accuracy: {max(r['test_accuracy'] for r in result['results'].values()):.2%}")
