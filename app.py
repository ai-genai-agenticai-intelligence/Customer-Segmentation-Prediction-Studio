import os
import io
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Non-interactive Matplotlib backend for thread-safe server rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Flask Initialization
# ---------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Default dataset path
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
if not os.path.exists(DEFAULT_DATASET_PATH):
    raise FileNotFoundError(f"Default dataset not found at {DEFAULT_DATASET_PATH}")

# In-memory active dataset storage
CURRENT_DATASET = {
    "df": pd.read_csv(DEFAULT_DATASET_PATH),
    "filename": "Mall_Customers.csv",
    "is_custom": False,
}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_numeric_columns(df):
    """Return numeric columns, excluding ID-like columns if possible."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filtered = [col for col in numeric_cols if "id" not in col.lower()]
    return filtered if len(filtered) >= 2 else numeric_cols

def get_gender_column(df):
    for col in ["Genre", "Gender", "gender", "sex", "Sex"]:
        if col in df.columns:
            return col
    return None

def get_cluster_persona(cluster_idx, centroids_df, col_x, col_y):
    """Generate dynamic business personas and strategies."""
    income_val = centroids_df.loc[cluster_idx, col_x]
    spending_val = centroids_df.loc[cluster_idx, col_y]

    med_income = centroids_df[col_x].median()
    med_spending = centroids_df[col_y].median()

    if "income" in col_x.lower() or "salary" in col_x.lower() or "spending" in col_y.lower() or "score" in col_y.lower():
        if income_val >= med_income and spending_val >= med_spending:
            return {
                "name": "💎 VIP / High Earners, High Spenders",
                "tag": "High-Value Elite",
                "badge_color": "#EC4899",
                "desc": "Prime target for luxury items, exclusive designer previews, and personal concierge service.",
                "ltv": "Very High (Tier 1)",
                "channel": "WhatsApp VIP Concierge & Private Invitations",
                "categories": ["Designer Fashion", "Fine Jewelry", "Luxury Watches", "Gourmet Dining"],
                "coupon": "VIP-ELITE-25",
                "discount": "25% Off Luxury Collection + Valet Parking"
            }
        elif income_val < med_income and spending_val >= med_spending:
            return {
                "name": "🛍️ Trendsetters / Low Income, High Spenders",
                "tag": "Aspirational Youth",
                "badge_color": "#F59E0B",
                "desc": "Trendy, young & spontaneous spenders motivated by seasonal drops, fashion pop-ups, and social buzz.",
                "ltv": "High Potential (Tier 2)",
                "channel": "Instagram / TikTok Ads & Mobile Push Notifications",
                "categories": ["Streetwear", "Sneakers", "Cosmetics", "Trendy Cafes"],
                "coupon": "TRENDY-DROP-15",
                "discount": "15% Off New Season + Buy Now Pay Later"
            }
        elif income_val >= med_income and spending_val < med_spending:
            return {
                "name": "🛡️ Savers / High Income, Low Spenders",
                "tag": "Pragmatic Affluent",
                "badge_color": "#06B6D4",
                "desc": "Financially secure shoppers who evaluate purchases carefully and demand maximum quality & durability.",
                "ltv": "Medium-High (Tier 2)",
                "channel": "Quality Product Catalogs & Weekend Newsletters",
                "categories": ["Home Appliances", "Electronics", "High-End Basics", "Automotive"],
                "coupon": "SAVER-PRIME-10",
                "discount": "10% Cashback + Extended 3-Year Free Warranty"
            }
        elif income_val < med_income and spending_val < med_spending:
            return {
                "name": "🏷️ Budget Shoppers / Low Income, Low Spenders",
                "tag": "Frugal Economizer",
                "badge_color": "#64748B",
                "desc": "Cost-conscious customers motivated by bundle discounts, clearance events, and value pricing.",
                "ltv": "Moderate (Tier 3)",
                "channel": "SMS Flash Sales & App Coupon Circulars",
                "categories": ["Clearance Apparel", "Value Packs", "Everyday Basics"],
                "coupon": "BUDGET-BOOST-5",
                "discount": "Buy 1 Get 1 Free on Select Everyday Items"
            }
        else:
            return {
                "name": "⚖️ Standard / Balanced Customers",
                "tag": "Steady Regulars",
                "badge_color": "#3B82F6",
                "desc": "Balanced income and moderate spending habits. Regular mall patrons who appreciate dependable quality.",
                "ltv": "Consistent (Tier 2)",
                "channel": "Seasonal Mall Festivals & Reward Points",
                "categories": ["Department Stores", "Casual Dining", "Family Entertainment"],
                "coupon": "REWARDS-PLUS-12",
                "discount": "12% Off Storewide + Double Loyalty Points"
            }
    else:
        return {
            "name": f"Segment {cluster_idx + 1} Profile",
            "tag": f"Cluster {cluster_idx + 1}",
            "badge_color": "#8B5CF6",
            "desc": f"Customers grouped around centroid ({income_val:.1f}, {spending_val:.1f}).",
            "ltv": "Calculated (Active)",
            "channel": "Omnichannel Digital Campaigns",
            "categories": ["Top Matched Products"],
            "coupon": f"PROMO-SEG-{cluster_idx + 1}",
            "discount": "10% Promotional Welcome Offer"
        }

def compute_centroids(X, labels, k):
    centroids = []
    for i in range(k):
        pts = X[labels == i]
        if len(pts) > 0:
            centroids.append(pts.mean(axis=0))
        else:
            centroids.append(np.zeros(X.shape[1]))
    return np.array(centroids)

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload custom CSV dataset."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file)
        if len(df) < 5:
            return jsonify({"success": False, "error": "Dataset must contain at least 5 rows"}), 400

        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) < 2:
            return jsonify({"success": False, "error": "Dataset must have at least 2 numerical columns for clustering"}), 400

        CURRENT_DATASET["df"] = df
        CURRENT_DATASET["filename"] = file.filename
        CURRENT_DATASET["is_custom"] = True

        return jsonify({
            "success": True,
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "preview": df.head(5).to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/reset_data", methods=["POST"])
def api_reset_data():
    """Reset back to default Mall_Customers.csv dataset."""
    CURRENT_DATASET["df"] = pd.read_csv(DEFAULT_DATASET_PATH)
    CURRENT_DATASET["filename"] = "Mall_Customers.csv"
    CURRENT_DATASET["is_custom"] = False

    numeric_cols = get_numeric_columns(CURRENT_DATASET["df"])
    return jsonify({
        "success": True,
        "filename": "Mall_Customers.csv",
        "rows": len(CURRENT_DATASET["df"]),
        "columns": CURRENT_DATASET["df"].columns.tolist(),
        "numeric_columns": numeric_cols,
    })

@app.route("/api/meta", methods=["GET"])
def api_meta():
    """Return dataset metadata and column options."""
    df = CURRENT_DATASET["df"]
    numeric_cols = get_numeric_columns(df)
    gender_col = get_gender_column(df)

    default_x = "Annual Income (k$)" if "Annual Income (k$)" in numeric_cols else numeric_cols[0]
    default_y = "Spending Score (1-100)" if "Spending Score (1-100)" in numeric_cols else numeric_cols[min(1, len(numeric_cols)-1)]
    default_z = "Age" if "Age" in numeric_cols else (numeric_cols[2] if len(numeric_cols) > 2 else None)

    return jsonify({
        "filename": CURRENT_DATASET["filename"],
        "is_custom": CURRENT_DATASET["is_custom"],
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "default_x": default_x,
        "default_y": default_y,
        "default_z": default_z,
        "gender_column": gender_col,
    })

@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    req = request.get_json() or {}
    algorithm = req.get("algorithm", "kmeans")
    k = int(req.get("k", 5))
    linkage = req.get("linkage", "ward")

    df = CURRENT_DATASET["df"]
    numeric_cols = get_numeric_columns(df)

    col_x = req.get("col_x", "Annual Income (k$)")
    col_y = req.get("col_y", "Spending Score (1-100)")
    col_z = req.get("col_z", "Age")

    if col_x not in df.columns:
        col_x = numeric_cols[0]
    if col_y not in df.columns:
        col_y = numeric_cols[min(1, len(numeric_cols)-1)]
    if col_z not in df.columns:
        col_z = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0]

    X = df[[col_x, col_y]].values

    # K-Means
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_centroids = km.cluster_centers_

    # Hierarchical
    agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage=linkage)
    agg_labels = agg.fit_predict(X)
    agg_centroids = compute_centroids(X, agg_labels, k)

    # Active model
    if algorithm == "hierarchical":
        active_labels = agg_labels
        active_centroids = agg_centroids
    else:
        active_labels = km_labels
        active_centroids = km_centroids

    # Silhouette Score
    sil = float(silhouette_score(X, active_labels)) if len(X) > k else 0.0

    # Personas
    centroids_df = pd.DataFrame(active_centroids, columns=[col_x, col_y])
    personas = {}
    for i in range(k):
        personas[i] = get_cluster_persona(i, centroids_df, col_x, col_y)

    # Segmented Records
    records_df = df.copy()
    records_df["KMeans_Cluster"] = km_labels.tolist()
    records_df["Hierarchical_Cluster"] = agg_labels.tolist()
    records_df["Cluster"] = active_labels.tolist()
    records_df["Persona"] = [personas[c]["name"] for c in active_labels]
    records_df["Persona_Tag"] = [personas[c]["tag"] for c in active_labels]

    # Agreement crosstab
    crosstab_df = pd.crosstab(km_labels, agg_labels)
    crosstab_matrix = crosstab_df.values.tolist()

    # Gender column check
    gender_col = get_gender_column(df)

    # Radar comparison metrics per cluster
    radar_data = []
    for i in range(k):
        c_pts = records_df[records_df["Cluster"] == i]
        females = len(c_pts[c_pts[gender_col].isin(["Female", "F", "female"])]) if gender_col else 0
        radar_data.append({
            "cluster": i,
            "name": personas[i]["name"].split("/")[0].strip(),
            "avg_x": float(c_pts[col_x].mean()),
            "avg_y": float(c_pts[col_y].mean()),
            "avg_z": float(c_pts[col_z].mean()) if col_z in c_pts else 35.0,
            "female_ratio": round((females / len(c_pts) * 100), 1) if len(c_pts) > 0 else 50.0,
            "count": len(c_pts)
        })

    response_data = {
        "k": k,
        "algorithm": algorithm,
        "filename": CURRENT_DATASET["filename"],
        "total_customers": len(df),
        "col_x": col_x,
        "col_y": col_y,
        "col_z": col_z,
        "avg_x": float(df[col_x].mean()),
        "avg_y": float(df[col_y].mean()),
        "avg_z": float(df[col_z].mean()) if col_z in df else 38.8,
        "silhouette": sil,
        "centroids": active_centroids.tolist(),
        "personas": personas,
        "crosstab": crosstab_matrix,
        "radar": radar_data,
        "data": records_df.to_dict(orient="records"),
    }
    return jsonify(response_data)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    req = request.get_json() or {}
    df = CURRENT_DATASET["df"]
    numeric_cols = get_numeric_columns(df)

    col_x = req.get("col_x", "Annual Income (k$)")
    col_y = req.get("col_y", "Spending Score (1-100)")
    col_z = req.get("col_z", "Age")

    if col_x not in df.columns:
        col_x = numeric_cols[0]
    if col_y not in df.columns:
        col_y = numeric_cols[min(1, len(numeric_cols)-1)]
    if col_z not in df.columns:
        col_z = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0]

    val_x = float(req.get("val_x", 60))
    val_y = float(req.get("val_y", 50))
    val_z = float(req.get("val_z", 30))
    algorithm = req.get("algorithm", "kmeans")
    k = int(req.get("k", 5))
    linkage = req.get("linkage", "ward")

    X = df[[col_x, col_y]].values

    # Fit active model
    if algorithm == "hierarchical":
        agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage=linkage)
        labels = agg.fit_predict(X)
        centroids = compute_centroids(X, labels, k)
    else:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

    # Predict cluster via nearest centroid
    new_point = np.array([val_x, val_y])
    distances_to_centroids = [np.linalg.norm(new_point - c) for c in centroids]
    predicted_cluster = int(np.argmin(distances_to_centroids))

    # Persona Profile
    centroids_df = pd.DataFrame(centroids, columns=[col_x, col_y])
    persona_obj = get_cluster_persona(predicted_cluster, centroids_df, col_x, col_y)

    # Cross-check Top 5 Nearest Existing Customers
    distances_to_pts = np.linalg.norm(X - new_point, axis=1)
    nearest_idx = np.argsort(distances_to_pts)[:5]
    nearest_df = df.iloc[nearest_idx].copy()
    nearest_df["Cluster"] = labels[nearest_idx].tolist()
    nearest_df["distance"] = np.round(distances_to_pts[nearest_idx], 2).tolist()

    return jsonify({
        "predicted_cluster": predicted_cluster,
        "persona": persona_obj["name"],
        "tag": persona_obj["tag"],
        "strategy": persona_obj["desc"],
        "ltv": persona_obj["ltv"],
        "channel": persona_obj["channel"],
        "categories": persona_obj["categories"],
        "coupon": persona_obj["coupon"],
        "discount": persona_obj["discount"],
        "nearest_customers": nearest_df.to_dict(orient="records"),
    })

@app.route("/api/elbow")
def api_elbow():
    df = CURRENT_DATASET["df"]
    numeric_cols = get_numeric_columns(df)
    col_x = request.args.get("col_x", numeric_cols[0])
    col_y = request.args.get("col_y", numeric_cols[min(1, len(numeric_cols)-1)])

    if col_x not in df.columns:
        col_x = numeric_cols[0]
    if col_y not in df.columns:
        col_y = numeric_cols[min(1, len(numeric_cols)-1)]

    X = df[[col_x, col_y]].values
    k_range = list(range(1, 11))
    wcss_list = []
    sil_list = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X)
        wcss_list.append(float(km.inertia_))
        if k >= 2:
            sil_list.append(float(silhouette_score(X, km.labels_)))
        else:
            sil_list.append(None)

    valid_sils = [s for s in sil_list if s is not None]
    optimal_sil_k = int(np.argmax(valid_sils) + 2) if len(valid_sils) > 0 else 5

    return jsonify({
        "k_range": k_range,
        "wcss": wcss_list,
        "silhouette": sil_list,
        "optimal_k_elbow": 5,
        "optimal_k_silhouette": optimal_sil_k,
    })

@app.route("/api/dendrogram")
def api_dendrogram():
    k = int(request.args.get("k", 5))
    linkage_type = request.args.get("linkage", "ward")

    df = CURRENT_DATASET["df"]
    numeric_cols = get_numeric_columns(df)
    col_x = request.args.get("col_x", numeric_cols[0])
    col_y = request.args.get("col_y", numeric_cols[min(1, len(numeric_cols)-1)])

    if col_x not in df.columns:
        col_x = numeric_cols[0]
    if col_y not in df.columns:
        col_y = numeric_cols[min(1, len(numeric_cols)-1)]

    X = df[[col_x, col_y]].values
    linkage_matrix = sch.linkage(X, method=linkage_type)

    last_merges = linkage_matrix[-(k - 1):, 2]
    threshold_cut = (last_merges[0] + linkage_matrix[-(k), 2]) / 2 if len(last_merges) > 0 else 0

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=100)
    sch.dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
    )
    ax.axhline(y=threshold_cut, color="#DC2626", linestyle="--", linewidth=2, label=f"Cut-off for k = {k}")
    ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_type.capitalize()} Linkage)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Customer Clusters / Merged Leaves (Sample Counts)", fontsize=11)
    ax.set_ylabel("Euclidean Distance", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

@app.route("/export")
def export_csv():
    df = CURRENT_DATASET["df"]
    numeric_cols = get_numeric_columns(df)
    col_x = request.args.get("col_x", numeric_cols[0])
    col_y = request.args.get("col_y", numeric_cols[min(1, len(numeric_cols)-1)])
    k = int(request.args.get("k", 5))
    cluster_filter = request.args.get("cluster", "all")

    if col_x not in df.columns:
        col_x = numeric_cols[0]
    if col_y not in df.columns:
        col_y = numeric_cols[min(1, len(numeric_cols)-1)]

    X = df[[col_x, col_y]].values
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    agg_labels = agg.fit_predict(X)

    export_df = df.copy()
    export_df["KMeans_Cluster"] = km_labels
    export_df["Hierarchical_Cluster"] = agg_labels

    if cluster_filter != "all":
        try:
            c_int = int(cluster_filter)
            export_df = export_df[export_df["KMeans_Cluster"] == c_int]
            filename = f"Customer_Segmentation_Cluster_{c_int}.csv"
        except ValueError:
            filename = "Customer_Segmentation_Results.csv"
    else:
        filename = "Customer_Segmentation_Results.csv"

    csv_data = export_df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"},
    )

# ---------------------------------------------------------
# Server Main
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Enterprise Customer Segmentation & Prediction Studio on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
