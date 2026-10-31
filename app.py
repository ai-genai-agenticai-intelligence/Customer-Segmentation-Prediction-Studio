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

# Data loading
DATASET_PATH = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df_raw = pd.read_csv(DATASET_PATH)

# Feature Columns
COL_X = "Annual Income (k$)"
COL_Y = "Spending Score (1-100)"
COL_Z = "Age"
GENDER_COL = "Genre" if "Genre" in df_raw.columns else ("Gender" if "Gender" in df_raw.columns else None)

# ---------------------------------------------------------
# Persona Assignment Logic & Actionable Strategy
# ---------------------------------------------------------
def get_cluster_persona(cluster_idx, centroids_df):
    income_val = centroids_df.loc[cluster_idx, COL_X]
    spending_val = centroids_df.loc[cluster_idx, COL_Y]

    med_income = centroids_df[COL_X].median()
    med_spending = centroids_df[COL_Y].median()

    if income_val > med_income and spending_val > med_spending:
        return {
            "name": "💎 VIP / High Earners, High Spenders",
            "tag": "High-Value Elite",
            "badge_color": "danger",
            "desc": "Prime target for luxury goods, personal shopper concierge, and exclusive VIP fashion galas.",
            "ltv": "Very High (Tier 1)",
            "channel": "Personalized WhatsApp Concierge & Private Preview Emails",
            "categories": ["Designer Fashion", "Fine Jewelry", "Luxury Watches", "Gourmet Dining"],
            "coupon": "VIP-ELITE-25",
            "discount": "25% Off Luxury Collection + Free Valet Parking"
        }
    elif income_val < med_income and spending_val > med_spending:
        return {
            "name": "🛍️ Trendsetters / Low Income, High Spenders",
            "tag": "Aspirational Youth",
            "badge_color": "warning",
            "desc": "Youthful & impulsive spenders attracted by trendy fast-fashion, pop-up stores, and social buzz.",
            "ltv": "High Potential (Tier 2)",
            "channel": "Instagram / TikTok Ads, Mobile Push Alerts & Influencer Collabs",
            "categories": ["Streetwear", "Sneakers", "Cosmetics", "Trendy Cafes & Bubble Tea"],
            "coupon": "TRENDY-DROP-15",
            "discount": "15% Off New Season Arrivals + Buy Now Pay Later Perk"
        }
    elif income_val > med_income and spending_val < med_spending:
        return {
            "name": "🛡️ Savers / High Income, Low Spenders",
            "tag": "Pragmatic Affluent",
            "badge_color": "info",
            "desc": "Financially comfortable customers who evaluate purchases carefully and demand long-term durability.",
            "ltv": "Medium-High (Tier 2)",
            "channel": "In-depth Quality Catalogs, Weekend Newsletter & Loyalty Points",
            "categories": ["Home Appliances", "Electronics", "High-End Essentials", "Automotive"],
            "coupon": "SAVER-PRIME-10",
            "discount": "10% Cashback + Extended 3-Year Free Warranty"
        }
    elif income_val < med_income and spending_val < med_spending:
        return {
            "name": "🏷️ Budget Shoppers / Low Income, Low Spenders",
            "tag": "Frugal Economizer",
            "badge_color": "secondary",
            "desc": "Price-sensitive shoppers motivated by extreme value, clearance sales, and volume discount bundles.",
            "ltv": "Moderate (Tier 3)",
            "channel": "SMS Flash Sales, Weekly Circulars & App Coupons",
            "categories": ["Clearance Apparel", "Value Grocery Packs", "Everyday Basics"],
            "coupon": "BUDGET-BOOST-5",
            "discount": "Buy 1 Get 1 Free on Select Everyday Essentials"
        }
    else:
        return {
            "name": "⚖️ Standard / Balanced Shoppers",
            "tag": "Steady Regulars",
            "badge_color": "primary",
            "desc": "Average income and moderate spending habits. Reliable mall visitors who appreciate consistent service.",
            "ltv": "Consistent (Tier 2)",
            "channel": "Email Promotions, Mall Loyalty Cards & Seasonal Festivals",
            "categories": ["Department Stores", "Casual Dining", "Family Entertainment", "Books & Stationery"],
            "coupon": "REWARDS-PLUS-12",
            "discount": "12% Off Storewide + Double Loyalty Reward Points"
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

# Precalculate Elbow & Silhouette curves once for instant loading
@app.route("/api/elbow")
def api_elbow():
    X = df_raw[[COL_X, COL_Y]].values
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

    return jsonify({
        "k_range": k_range,
        "wcss": wcss_list,
        "silhouette": sil_list,
        "optimal_k_elbow": 5,
        "optimal_k_silhouette": int(np.argmax([s for s in sil_list if s is not None]) + 2)
    })

# ---------------------------------------------------------
# Main Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    req = request.get_json() or {}
    algorithm = req.get("algorithm", "kmeans")
    k = int(req.get("k", 5))
    linkage = req.get("linkage", "ward")

    X = df_raw[[COL_X, COL_Y]].values

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
    centroids_df = pd.DataFrame(active_centroids, columns=[COL_X, COL_Y])
    personas = {}
    for i in range(k):
        personas[i] = get_cluster_persona(i, centroids_df)

    # Segmented Records
    records_df = df_raw.copy()
    records_df["KMeans_Cluster"] = km_labels.tolist()
    records_df["Hierarchical_Cluster"] = agg_labels.tolist()
    records_df["Cluster"] = active_labels.tolist()
    records_df["Persona"] = [personas[c]["name"] for c in active_labels]
    records_df["Persona_Tag"] = [personas[c]["tag"] for c in active_labels]

    # Agreement crosstab
    crosstab_df = pd.crosstab(km_labels, agg_labels)
    crosstab_matrix = crosstab_df.values.tolist()

    # Radar comparison metrics per cluster
    radar_data = []
    for i in range(k):
        c_pts = records_df[records_df["Cluster"] == i]
        females = len(c_pts[c_pts[GENDER_COL] == "Female"]) if GENDER_COL else 0
        radar_data.append({
            "cluster": i,
            "name": personas[i]["name"].split("/")[0].strip(),
            "avg_salary": float(c_pts[COL_X].mean()),
            "avg_spending": float(c_pts[COL_Y].mean()),
            "avg_age": float(c_pts[COL_Z].mean()) if COL_Z in c_pts else 35.0,
            "female_ratio": round((females / len(c_pts) * 100), 1) if len(c_pts) > 0 else 50.0,
            "count": len(c_pts)
        })

    response_data = {
        "k": k,
        "algorithm": algorithm,
        "total_customers": len(df_raw),
        "avg_salary": float(df_raw[COL_X].mean()),
        "avg_spending": float(df_raw[COL_Y].mean()),
        "avg_age": float(df_raw[COL_Z].mean()) if COL_Z in df_raw else 38.8,
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
    age = float(req.get("age", 30))
    salary = float(req.get("salary", 60))
    spending = float(req.get("spending", 50))
    algorithm = req.get("algorithm", "kmeans")
    k = int(req.get("k", 5))
    linkage = req.get("linkage", "ward")

    X = df_raw[[COL_X, COL_Y]].values

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
    new_point = np.array([salary, spending])
    distances_to_centroids = [np.linalg.norm(new_point - c) for c in centroids]
    predicted_cluster = int(np.argmin(distances_to_centroids))

    # Persona Profile
    centroids_df = pd.DataFrame(centroids, columns=[COL_X, COL_Y])
    persona_obj = get_cluster_persona(predicted_cluster, centroids_df)

    # Cross-check Top 5 Nearest Existing Customers
    distances_to_pts = np.linalg.norm(X - new_point, axis=1)
    nearest_idx = np.argsort(distances_to_pts)[:5]
    nearest_df = df_raw.iloc[nearest_idx].copy()
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

@app.route("/api/dendrogram")
def api_dendrogram():
    k = int(request.args.get("k", 5))
    linkage_type = request.args.get("linkage", "ward")

    X = df_raw[[COL_X, COL_Y]].values
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
    cluster_filter = request.args.get("cluster", "all")
    k = int(request.args.get("k", 5))

    X = df_raw[[COL_X, COL_Y]].values
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    agg_labels = agg.fit_predict(X)

    export_df = df_raw.copy()
    export_df["KMeans_Cluster"] = km_labels
    export_df["Hierarchical_Cluster"] = agg_labels

    if cluster_filter != "all":
        try:
            c_int = int(cluster_filter)
            export_df = export_df[export_df["KMeans_Cluster"] == c_int]
            filename = f"Mall_Customers_Cluster_{c_int}.csv"
        except ValueError:
            filename = "Mall_Customers_Segmented.csv"
    else:
        filename = "Mall_Customers_Segmented.csv"

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
    print("Starting Enhanced Flask Customer Segmentation App on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
