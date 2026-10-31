# 🛍️ Customer Segmentation & Prediction Studio (Flask Web App)
liveDemo -customer-segmentation-prediction-st.vercel.app

An interactive, data-driven Customer Segmentation and Behavioral Analytics web application built with **Flask**, **Scikit-learn**, **SciPy**, and **Plotly.js**, featuring both **K-Means Clustering** and **Hierarchical Clustering (Agglomerative & Dendrogram)**.

Based on [`K_Means Cluster.ipynb`](K_Means%20Cluster.ipynb) and [`Mall_Customers.csv`](Mall_Customers.csv).

---

## 🚀 Quick Start

To launch the Flask web application in your local environment:

```bash
python app.py
```

The web app will launch at:
👉 **[http://localhost:5000](http://localhost:5000)** (or `http://127.0.0.1:5000`)

---

## 🌟 Key Features

1. **Dual Machine Learning Models**:
   - **K-Means Clustering**: With dynamic $k$ selection (2 to 10), centroids, and inertia optimization.
   - **Hierarchical Clustering (Agglomerative)**: With Ward, Complete, Average, and Single linkage criteria.
   - **Hierarchical Dendrogram**: Full tree linkage visualization with horizontal cut-off threshold line.
   - **Side-by-Side Model Comparison**: Cross-tabulation agreement matrix comparing K-Means and Hierarchical groupings.

2. **🔍 Cross-Check Cluster Data Points**:
   - Filter by any cluster to inspect which exact customer rows and IDs belong to it.
   - View boundary ranges (Min, Max, and Mean) for Salary, Spending Score, and Age.
   - Live searchable customer data table.

3. **🔮 Interactive Prediction for New Customers**:
   - Check the **"🔮 Predict for New Customer"** toggle.
   - Input **Age**, **Annual Salary / Income ($k$)**, and **Spending Score ($1-100$)**.
   - Instantly predicts the customer's cluster and assigned persona.
   - Plots the new customer point on the 2D / 3D cluster scatter plot.
   - **Nearest Neighbors Cross-Check**: Shows the Top 5 closest existing customers in the dataset with Euclidean distance and original clusters.

4. **Zero-Setup Client-Side Visualizations**:
   - Uses **Plotly.js via CDN** directly in the browser for zoom, pan, hover tooltips, and interactive legends.
   - Toggle between **2D Scatter** and **🌐 3D Customer Space (Age × Salary × Spending)**.

5. **Data Export**:
   - Download the full dataset annotated with both `KMeans_Cluster` and `Hierarchical_Cluster` as CSV via `/export`.

---

## 📁 Project Structure

```
CUSTOMER SEGMENTATION/
├── app.py                     # Flask Web Server & Clustering API
├── Mall_Customers.csv         # Customer Dataset (200 records)
├── K_Means Cluster.ipynb      # Initial Jupyter Notebook
├── templates/
│   └── index.html             # Responsive Bootstrap 5 & Plotly.js Frontend
├── requirements.txt           # Dependencies
└── README.md                  # Documentation & Guide
```
