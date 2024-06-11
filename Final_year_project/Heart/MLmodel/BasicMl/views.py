import pandas as pd
from django.shortcuts import render
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from mlxtend.classifier import StackingCVClassifier
import matplotlib.pyplot as plt
import io
import base64

def welcome(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    if request.method == 'GET':
        data = pd.read_csv(r'C:\Users\dhars\Downloads\Final_year_project__review\Final_year_project\Heart\MLmodel\heartdisease.csv')
        y = data["target"]
        X = data.drop('target', axis=1)
        y_binary = np.where(y > 0, 1, 0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(solver='liblinear')
        rf = RandomForestClassifier(random_state=2)
        knn = KNeighborsClassifier()
        svc = SVC(kernel='rbf', probability=True)
        gpc = GaussianProcessClassifier()
        adaboost = AdaBoostClassifier()
        classifiers = [lr, rf, knn, svc, gpc, adaboost]
        scv = StackingCVClassifier(
            classifiers=classifiers, meta_classifier=svc, random_state=42)
        scv.fit(X_scaled, y_binary)
        rf.fit(X_scaled, y_binary)

        val1 = float(request.GET.get('age'))
        val2 = float(request.GET.get('sex'))
        val3 = float(request.GET.get('cp'))
        val4 = float(request.GET.get('trestbps'))
        val5 = float(request.GET.get('chol'))
        val6 = float(request.GET.get('fbs'))
        val7 = float(request.GET.get('restecg'))
        val8 = float(request.GET.get('thalach'))
        val9 = float(request.GET.get('exang'))
        val10 = float(request.GET.get('oldpeak'))
        val11 = float(request.GET.get('slope'))

        user_input_scaled = scaler.transform(
            [[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11]])

        prediction = scv.predict(user_input_scaled)
        result = "POSITIVE" if prediction == 1 else "NEGATIVE"

        user_df = pd.DataFrame({
            'Age': [val1],
            'Sex': [val2],
            'Chest Pain Type': [val3],
            'Resting Blood Pressure': [val4],
            'Serum Cholesterol': [val5],
            'Fasting Blood Sugar': [val6],
            'Resting Electrocardiographic Results': [val7],
            'Maximum Heart Rate Achieved': [val8],
            'Exercise Induced Angina': [val9],
            'Old Peak': [val10],
            'Slope': [val11],
            'Prediction': [result]
        })

        normal_values = {
            'Age': None,
            'Sex': None,
            'Chest Pain Type': 0,
            'Resting Blood Pressure': 120,
            'Serum Cholesterol': 200,
            'Fasting Blood Sugar': 100,
            'Resting Electrocardiographic Results': 0,
            'Maximum Heart Rate Achieved': 80,
            'Exercise Induced Angina': 0,
            'Old Peak': 1,
            'Slope': 0
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        param_names = list(user_df.columns[:-1])
        param_values = user_df.iloc[0, :-1].values
        normal_vals = [normal_values[param] for param in param_names]

        ax.plot(param_names, normal_vals, label='Normal', marker='o')
        ax.plot(param_names, param_values, label='User Input', marker='x')

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')
        ax.set_title('Visualization Chart')
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()

        context = {'result2': result, 'chart_image': chart_image}
        return render(request, 'result.html', context)

def count_plot_age_sex_target(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', hue='sex', data=data)
    plt.title('Count Plot of Heart Disease by Sex')
    plt.ylabel('Count')
    plt.legend(title='Sex', labels=['Female', 'Male'])
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64

def generate_feature_histograms(data):
    plt.figure(figsize=(12, 10))
    features = data.columns.tolist()

    # Remove the target column if it exists
    if 'target' in features:
        features.remove('target')

    num_plots = len(features)
    rows = num_plots // 2 if num_plots % 2 == 0 else (num_plots // 2) + 1

    for i, feature in enumerate(features, start=1):
        plt.subplot(rows, 2, i)
        sns.histplot(data=data, x=feature, kde=True)
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    return image_base64
def generate_dummy_data():
    # Generate dummy test labels (replace this with your actual data)
    y_test = np.random.randint(0, 2, size=(100,))
    return y_test
def generate_violin_plot(data):
    plt.figure(figsize=(12, 8))
    features = data.columns.tolist()

    # Remove the target column if it exists
    if 'target' in features:
        features.remove('target')

    num_plots = len(features)
    rows = num_plots // 2 if num_plots % 2 == 0 else (num_plots // 2) + 1

    for i, feature in enumerate(features, start=1):
        plt.subplot(rows, 2, i)
        sns.violinplot(x='target', y=feature, data=data)
        plt.title(f'Violin Plot of {feature}')
        plt.xlabel('Target')
        plt.ylabel(feature)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    return image_base64
def roc_curve_comparison(X_train, y_train, X_test, y_test):
    classifiers = {
        'Logistic Regression': LogisticRegression(solver='liblinear'),
        'Random Forest Classifier': RandomForestClassifier(random_state=2),
        'K-Neighbors Classifier': KNeighborsClassifier(),
        'Support Vector Machine': SVC(kernel='rbf', probability=True),
        'Gaussian Process Classifier': GaussianProcessClassifier(),
        'AdaBoost Classifier': AdaBoostClassifier()
    }

    plt.figure(figsize=(10, 8))
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        if hasattr(clf, 'predict_proba'):
            y_pred_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = clf.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')

    # StackingCVClassifier
    scv = StackingCVClassifier(classifiers=list(classifiers.values()), meta_classifier=LogisticRegression(), random_state=42)
    scv.fit(X_train, y_train)
    y_pred_prob_scv = scv.predict_proba(X_test)[:, 1] if hasattr(scv, 'predict_proba') else scv.decision_function(X_test)
    fpr_scv, tpr_scv, thresholds_scv = roc_curve(y_test, y_pred_prob_scv)
    roc_auc_scv = auc(fpr_scv, tpr_scv)
    plt.plot(fpr_scv, tpr_scv, label=f'StackingCVClassifier (AUC = {roc_auc_scv:.2f})', linestyle='--')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    
    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Different Classifiers')
    
    # Show legend
    plt.legend()

    # Save the plot as base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return image_base64
def evaluation_comparison_plot(evaluation_metrics_train, evaluation_metrics_test, stacking_clf, X_test, y_test):
    classifiers = {
        'Logistic Regression': (evaluation_metrics_train['Logistic Regression'], evaluation_metrics_test['Logistic Regression']),
        'Random Forest Classifier': (evaluation_metrics_train['Random Forest Classifier'], evaluation_metrics_test['Random Forest Classifier']),
        'K-Neighbors Classifier': (evaluation_metrics_train['K-Neighbors Classifier'], evaluation_metrics_test['K-Neighbors Classifier']),
        'Support Vector Machine': (evaluation_metrics_train['Support Vector Machine'], evaluation_metrics_test['Support Vector Machine']),
        'Gaussian Process Classifier': (evaluation_metrics_train['Gaussian Process Classifier'], evaluation_metrics_test['Gaussian Process Classifier']),
        'AdaBoost Classifier': (evaluation_metrics_train['AdaBoost Classifier'], evaluation_metrics_test['AdaBoost Classifier']),
    }
    
    # Calculate evaluation metrics for StackingCVClassifier separately
    stacking_metrics_train = evaluate_metrics(y_test, stacking_clf.predict(X_test))
    stacking_metrics_test = evaluate_metrics(y_test, stacking_clf.predict(X_test))
    classifiers['StackingCVClassifier'] = (stacking_metrics_train, stacking_metrics_test)

    metric_names = list(classifiers['Logistic Regression'][0].keys())

    plt.figure(figsize=(14, 10))  # Adjust the size of the plot here

    for metric in metric_names:
        train_metric_values = [metrics_train[metric] for metrics_train, _ in classifiers.values()]
        test_metric_values = [metrics_test[metric] for _, metrics_test in classifiers.values()]

        plt.plot(train_metric_values, marker='o', label=f'{metric} (Training)', linestyle='--')
        plt.plot(test_metric_values, marker='o', label=f'{metric} (Testing)')

    plt.xlabel('Classifier', fontsize=12)  # Increase fontsize for better visibility
    plt.ylabel('Metric Value', fontsize=12)  # Increase fontsize for better visibility
    plt.title('Evaluation Metric Comparison', fontsize=14)  # Increase fontsize for better visibility
    plt.xticks(range(len(classifiers)), list(classifiers.keys()), rotation=45, ha='right', fontsize=10)  # Increase fontsize and rotate labels for better readability
    plt.yticks(fontsize=10)  # Increase fontsize for better visibility
    plt.grid(True)
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return image_base64

def evaluate_metrics(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    classification_error = (fp + fn) / (tp + tn + fp + fn)
    absolute_error = (fp + fn) / len(y_true)
    relative_error = absolute_error / (tp + tn + fp + fn)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    recall = tp / (tp + fn)
    train_acc_score = accuracy_score(y_true, y_pred)

    return {
        'accuracy': acc_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score,
        'classification_error': classification_error,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'correlation': correlation,
        'roc_auc': roc_auc,
        'recall': recall,
        'training_accuracy': train_acc_score
    }


def visualization(request):
    if request.method == 'GET':
        data = pd.read_csv(r'C:\Users\dhars\Downloads\Final_year_project__review\Final_year_project\Heart\MLmodel\heartdisease.csv')
        y_test = generate_dummy_data()
        count_plot_age_sex_target_image = count_plot_age_sex_target(data)
       
        # Generate histograms for each feature in the dataset
        feature_histograms_image = generate_feature_histograms(data)
        violin_plot_image = generate_violin_plot(data)  # Generating the violin plot image
  
        y = data["target"]
        X = data.drop('target', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(solver='liblinear'),
            'Random Forest Classifier': RandomForestClassifier(random_state=2),
            'K-Neighbors Classifier': KNeighborsClassifier(),
            'Support Vector Machine': SVC(kernel='rbf', probability=True),
            'Gaussian Process Classifier': GaussianProcessClassifier(),
            'AdaBoost Classifier': AdaBoostClassifier(),
        }

        evaluation_metrics_train = {}
        evaluation_metrics_test = {}
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            evaluation_metrics_train[clf_name] = {'accuracy': accuracy_train}
            evaluation_metrics_test[clf_name] = {'accuracy': accuracy_test}

        # Define StackingCVClassifier with classifiers
        stacking_clf = StackingCVClassifier(
            classifiers=list(classifiers.values()), 
            meta_classifier=SVC(kernel='rbf', probability=True), 
            random_state=42
        )

        # Fit StackingCVClassifier
        stacking_clf.fit(X_train, y_train)

        # Get training accuracy of StackingCVClassifier
        scv_train_acc_score = np.mean([accuracy_score(y_train, clf.predict(X_train)) for clf in stacking_clf.classifiers])

        # Evaluate and get confusion matrix
        y_pred_stacking = stacking_clf.predict(X_test)
        accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
        evaluation_metrics_train['StackingCVClassifier'] = {'accuracy': scv_train_acc_score}
        evaluation_metrics_test['StackingCVClassifier'] = {'accuracy': accuracy_stacking}

        roc_curve_comparison_image = roc_curve_comparison(X_train, y_train, X_test, y_test)

        # Pass y_train to the evaluation_comparison_plot function
        # evaluation_comparison_image = evaluation_comparison_plot(evaluation_metrics_train, evaluation_metrics_test, stacking_clf, X_test, y_test)
        evaluation_comparison_image = evaluation_comparison_plot(evaluation_metrics_train, evaluation_metrics_test, stacking_clf, X_test, y_test)

        context = {
            'count_plot_age_sex_target_image': count_plot_age_sex_target_image,
            'feature_histograms_image': feature_histograms_image,
            'roc_curve_comparison_image': roc_curve_comparison_image,
            'evaluation_comparison_image': evaluation_comparison_image,
            'violin_plot_image': violin_plot_image  # Adding the violin plot image to the context
        }

        return render(request, 'visualization.html', context)
