# Anti-Money Laundering Detection with Transaction Network Analytics

A machine-learning and graph-analysis pipeline for identifying illicit cryptocurrency transactions using the Elliptic dataset. The project combines transaction-level attributes with graph-derived behavioral features to test whether network structure improves illicit-activity detection.

## Why this project

Anti-money-laundering detection is a difficult classification problem because illicit activity is relatively rare, patterns evolve over time, and individual transactions may look ordinary in isolation. This project treats transactions as part of a network and asks whether relationship-level signals can improve detection.

The workflow is designed around an applied analytics question:

> Can transaction-network behavior help distinguish illicit activity from licit activity while avoiding obvious target leakage?

## What the pipeline does

1. Loads transaction features, labels, and transaction-network edges.
2. Standardizes class labels and removes unknown labels from supervised modeling.
3. Builds a transaction graph with NetworkX.
4. Engineers graph features including:
   - degree
   - degree centrality
   - clustering coefficient
   - neighbor illicit ratio
5. Uses a time-based train/test split when a valid time variable is available, with a stratified random split as fallback.
6. Restricts the neighbor-label feature to training IDs to reduce leakage risk.
7. Trains two baseline models:
   - Logistic Regression
   - Random Forest
8. Handles class imbalance with balanced class weights.
9. Evaluates models using:
   - confusion matrix
   - precision / recall / F1
   - ROC-AUC
   - PR-AUC
10. Saves model results and diagnostic visualizations to `outputs/`.

## Tech stack

- Python
- pandas
- NumPy
- NetworkX
- scikit-learn
- Matplotlib

## Repository structure

```text
elliptic-aml-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── README.md
├── src/
│   └── aml_detection.py
└── outputs/
```

## Data

This project uses the Elliptic cryptocurrency transaction dataset. The raw CSV files are **not included** in this repository because they are large and should be obtained from the original dataset source.

Place these files inside `data/` before running the analysis:

```text
data/txs_features.csv
data/txs_classes.csv
data/txs_edgelist.csv
```

## How to run

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline from the repository root:

```bash
python src/aml_detection.py
```

Generated artifacts are written to `outputs/`.

## Feature engineering

### Graph structure

Each transaction is treated as a node in a graph, with edges representing transaction relationships. The pipeline derives:

- **Degree:** number of directly connected transactions.
- **Degree centrality:** normalized connectivity relative to the graph.
- **Clustering coefficient:** degree to which a transaction's neighbors are connected with each other.

### Neighbor illicit ratio

For each transaction, the pipeline calculates the share of usable neighboring transactions labeled illicit. To reduce target leakage, only neighbors belonging to the training set are used when constructing this feature.

This feature is intended to capture whether illicit activity is locally concentrated in the transaction network.

## Modeling approach

### Logistic Regression

Used as an interpretable linear baseline with median imputation, feature scaling, and balanced class weights.

### Random Forest

Used to capture nonlinear relationships and interactions between transaction-level and graph-derived features. Balanced class weights are used to better handle the imbalanced target.

## Evaluation

Accuracy alone can be misleading when the positive class is rare, so this project emphasizes precision, recall, F1, ROC-AUC, and especially PR-AUC alongside the confusion matrix.

The script also generates feature-importance output for the Random Forest model to help identify which signals contribute most to predictions.

## Business relevance

Although this project focuses on financial crime, the analytical workflow is broadly transferable to trust, safety, privacy, fraud, and security problems:

- investigate large operational datasets
- identify why undesirable outcomes occur
- engineer behavioral signals
- quantify patterns and risk
- compare alternative analytical approaches
- translate findings into system or process improvements

That workflow is especially relevant to data teams working on privacy operations, abuse prevention, data quality, and automated decision systems.

## Current limitation

Model performance values are intentionally not hard-coded into this README. The repository should report metrics only after the full local dataset has been run successfully. This avoids presenting unverified results.

## Future improvements

Potential next steps include:

- threshold tuning for precision/recall tradeoffs
- temporal cross-validation
- calibration analysis
- graph embeddings
- gradient-boosted models
- SHAP-based interpretability
- deeper error analysis on false positives and false negatives

## Author

**Varisht Aggarwal**  
M.S. Business Analytics, Northeastern University
