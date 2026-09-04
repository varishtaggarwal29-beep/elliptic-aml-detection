import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

FEATURES_FILE = "data/txs_features.csv"
CLASSES_FILE = "data/txs_classes.csv"
EDGES_FILE = "data/txs_edgelist.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def find_txid_column(df):
    possible = ["txId", "txid", "tx_id", "transaction_id", "id"]
    for col in possible:
        if col in df.columns:
            return col
    return df.columns[0]


def find_class_column(df):
    possible = ["class", "label", "target", "y"]
    for col in possible:
        if col in df.columns:
            return col
    return df.columns[1]


def find_time_column(df):
    possible = ["time_step", "timestep", "time", "timestamp", "step"]
    for col in possible:
        if col in df.columns:
            return col
    return None


def standardize_class_values(series):
    def convert(x):
        if pd.isna(x):
            return -1

        s = str(x).strip().lower()

        if s in ["illicit", "fraud", "fraudulent"]:
            return 1
        if s in ["licit", "legit", "legitimate"]:
            return 0
        if s in ["unknown"]:
            return -1

        try:
            v = int(float(s))
            if v == 1:
                return 1
            elif v == 2:
                return 0
            elif v in [3, -1]:
                return -1
            elif v == 0:
                return 0
            else:
                return -1
        except:
            return -1

    return series.apply(convert)


def find_edge_columns(edges_df):
    possible_pairs = [
        ("txId1", "txId2"),
        ("source", "target"),
        ("src", "dst"),
        ("from", "to")
    ]
    for c1, c2 in possible_pairs:
        if c1 in edges_df.columns and c2 in edges_df.columns:
            return c1, c2
    return edges_df.columns[0], edges_df.columns[1]


def safe_merge(features_df, classes_df):
    tx_col_features = find_txid_column(features_df)
    tx_col_classes = find_txid_column(classes_df)
    class_col = find_class_column(classes_df)

    features_df = features_df.rename(columns={tx_col_features: "txId"})
    classes_df = classes_df.rename(columns={tx_col_classes: "txId", class_col: "class"})
    classes_df["class"] = standardize_class_values(classes_df["class"])

    df = features_df.merge(classes_df[["txId", "class"]], on="txId", how="inner")
    return df


def build_graph(edges_df):
    src_col, dst_col = find_edge_columns(edges_df)
    edges_df = edges_df[[src_col, dst_col]].dropna().copy()
    edges_df.columns = ["source", "target"]
    G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())
    return G


def add_graph_structure_features(df, G):
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G)

    df["degree"] = df["txId"].map(degree_dict).fillna(0)
    df["degree_centrality"] = df["txId"].map(degree_centrality).fillna(0)
    df["clustering_coeff"] = df["txId"].map(clustering).fillna(0)

    return df


def add_neighbor_illicit_ratio(df, G, train_ids):
    train_ids = set(train_ids)
    label_map = dict(zip(df["txId"], df["class"]))

    def calc_ratio(node):
        if node not in G:
            return 0.0

        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            return 0.0

        usable_neighbors = [n for n in neighbors if n in train_ids and n in label_map]
        if len(usable_neighbors) == 0:
            return 0.0

        illicit_count = sum(1 for n in usable_neighbors if label_map[n] == 1)
        return illicit_count / len(usable_neighbors)

    df["neighbor_illicit_ratio"] = df["txId"].apply(calc_ratio)
    return df


def save_basic_plots(df):
    plt.figure(figsize=(6, 4))
    df["class"].value_counts().sort_index().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    df.boxplot(column="degree", by="class")
    plt.title("Degree by Class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Degree")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "degree_by_class.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    df.boxplot(column="neighbor_illicit_ratio", by="class")
    plt.title("Neighbor Illicit Ratio by Class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Neighbor Illicit Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "neighbor_ratio_by_class.png"))
    plt.close()


def prepare_features(df):
    X = df.drop(columns=["txId", "class"], errors="ignore").copy()
    y = df["class"].copy()
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    result = {"name": name, "y_pred": y_pred}

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        result["y_prob"] = y_prob
        result["roc_auc"] = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        result["pr_auc"] = auc(recall, precision)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    if "roc_auc" in result:
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print(f"PR-AUC: {result['pr_auc']:.4f}")

    return result


def plot_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()

    print("\nTop Feature Importances:")
    print(feat_imp.to_string(index=False))


def split_data(df):
    time_col = find_time_column(df)

    if time_col is not None:
        unique_times = sorted(df[time_col].dropna().unique())
        cutoff_index = max(1, int(len(unique_times) * (1 - TEST_SIZE)))
        cutoff_time = unique_times[cutoff_index - 1]

        train_df = df[df[time_col] <= cutoff_time].copy()
        test_df = df[df[time_col] > cutoff_time].copy()

        if train_df["class"].nunique() >= 2 and test_df["class"].nunique() >= 2 and len(test_df) > 0:
            split_type = f"time-based split on {time_col}"
            return train_df, test_df, split_type

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["class"]
    )
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()
    split_type = "stratified random split"
    return train_df, test_df, split_type


def main():
    print("Loading data...")

    features_df = pd.read_csv(FEATURES_FILE)
    classes_df = pd.read_csv(CLASSES_FILE)
    edges_df = pd.read_csv(EDGES_FILE)

    print(f"Features shape: {features_df.shape}")
    print(f"Classes shape : {classes_df.shape}")
    print(f"Edges shape   : {edges_df.shape}")

    raw_class_col = find_class_column(classes_df)
    print("\nRaw class values:")
    print(classes_df[raw_class_col].value_counts(dropna=False).sort_index())

    df = safe_merge(features_df, classes_df)

    print("\nMerged data shape:", df.shape)
    print("Mapped class counts:")
    print(df["class"].value_counts(dropna=False).sort_index())

    df = df[df["class"].isin([0, 1])].copy()
    df["class"] = df["class"].astype(int)

    print("\nAfter removing unknown labels:")
    print(df["class"].value_counts().sort_index())

    if df["class"].nunique() < 2:
        raise ValueError("The processed dataset still has fewer than 2 classes.")

    train_df, test_df, split_type = split_data(df)

    print(f"\nUsing {split_type}")
    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\nTrain class counts:")
    print(train_df["class"].value_counts().sort_index())
    print("\nTest class counts:")
    print(test_df["class"].value_counts().sort_index())

    G = build_graph(edges_df)

    train_ids = set(train_df["txId"])

    df = add_graph_structure_features(df, G)
    df = add_neighbor_illicit_ratio(df, G, train_ids)

    train_df = df[df["txId"].isin(train_df["txId"])].copy()
    test_df = df[df["txId"].isin(test_df["txId"])].copy()

    save_basic_plots(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "processed_transactions.csv"), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_transactions.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_transactions.csv"), index=False)

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    print("\nFinal train feature matrix shape:", X_train.shape)
    print("Final test feature matrix shape :", X_test.shape)

    log_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    log_reg.fit(X_train, y_train)
    log_results = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    rf.fit(X_train, y_train)
    rf_results = evaluate_model("Random Forest", rf, X_test, y_test)

    rf_model = rf.named_steps["model"]
    plot_feature_importance(rf_model, X_train.columns, top_n=15)

    results = []

    if "roc_auc" in log_results:
        results.append({
            "model": log_results["name"],
            "roc_auc": log_results["roc_auc"],
            "pr_auc": log_results["pr_auc"]
        })

    if "roc_auc" in rf_results:
        results.append({
            "model": rf_results["name"],
            "roc_auc": rf_results["roc_auc"],
            "pr_auc": rf_results["pr_auc"]
        })

if __name__ == "__main__":
    main()
