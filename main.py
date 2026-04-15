"""
Main Orchestrator Script

Pipeline for AI-Driven Optimisation of Radar Absorbing Materials:
1. Data Acquisition (Materials Project API)
2. Feature Engineering (pymatgen + mendeleev)
3. EDA & Visualization (PCA, t-SNE, K-Means)
4. Baseline Regression (Linear, Polynomial, SVR, Random Forest)
5. Classification (Logistic Regression, SVM, Decision Tree)
6. Deep Neural Network (PyTorch, K-Fold Validation)
7. Evaluation & Comparison (Model comparison, Feature importance, Candidate ranking)

Run: python main.py
"""

import subprocess
import sys
import time

# Pipeline steps configuration
STEPS = [
    ("DATA ACQUISITION", "1_data_acquisition.py"),
    ("FEATURE ENGINEERING", "2_feature_engineering.py"),
    ("EDA AND VISUALISATION", "3_eda_and_viz.py"),
    ("BASELINE REGRESSION", "4_regression_models.py"),
    ("CLASSIFICATION", "5_classification.py"),
    ("DEEP LEARNING (DNN)", "6_deep_learning.py"),
    ("EVALUATION", "7_evaluation.py"),
]

def run_step(step_title, script_name):
    """
    Execute a single pipeline step.

    Args:
        step_title (str): Display title for the step
        script_name (str): Python script filename

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {step_title}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            check=False
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[DONE] {step_title} completed in {elapsed_time:.1f}s")
            return True
        else:
            print(f"\n[ERROR] {step_title} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n[ERROR] {step_title} failed: {e}")
        return False


def main():
    """
    Main orchestration function that runs the entire pipeline.
    """
    print("=" * 70)
    print("AI-DRIVEN OPTIMISATION OF RADAR ABSORBING MATERIALS")
    print("Complete ML Pipeline")
    print("=" * 70)

    total_start = time.time()
    steps_completed = 0

    for step_title, script_name in STEPS:
        success = run_step(step_title, script_name)

        if success:
            steps_completed += 1
        else:
            print(f"\n{'='*70}")
            print("PIPELINE TERMINATED")
            print(f"{'='*70}")
            print(f"Failed at step {steps_completed + 1}: {step_title}")
            print("Please check the error above and re-run the pipeline.")
            sys.exit(1)

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"All {steps_completed} steps completed successfully!")
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print("\nResults:")
    print("  - Processed data: data/")
    print("  - Visualizations: plots/")
    print("  - Trained models: data/*.pkl, data/*.pth")
    print("\nNext steps:")
    print("  - Review the generated plots in the plots/ directory")
    print("  - Check the final summary and top candidates")
    print("  - Analyze feature importance for insights")
    print("  - Consider using the trained models for new material screening")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline terminated by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
