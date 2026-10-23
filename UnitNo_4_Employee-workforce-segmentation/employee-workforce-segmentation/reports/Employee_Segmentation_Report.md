# Employee Segmentation Report

## 1. Project Objective
Group employees with similar characteristics using K-Means and
Hierarchical Clustering. Use PCA to visualize the resulting segments.

## 2. Dataset
- Employee records analyzed: 5000
- K-Means clusters: 2
- Hierarchical clusters: 2

## 3. Methodology
1. Inspected and cleaned the dataset.
2. Imputed missing values where necessary.
3. Encoded categorical features.
4. Standardized numerical features.
5. Applied K-Means and Hierarchical Clustering.
6. Evaluated clustering quality using silhouette, Davies-Bouldin,
   and Calinski-Harabasz metrics.
7. Applied PCA for two-dimensional visualization.

## 4. K-Means Cluster Sizes
KMeans_Cluster
0    2219
1    2781

## 5. Hierarchical Cluster Sizes
Hierarchical_Cluster
0    3164
1    1836

## 6. Model Comparison
       Model  Silhouette Score  Davies-Bouldin Index  Calinski-Harabasz Score
     K-Means          0.056334              3.991612               308.221494
Hierarchical          0.038535              4.800982               200.011701

## 7. Workforce Analysis
Department-level distribution, employee characteristics, and
historical attrition analysis are available in the notebook and
exported results where the relevant columns exist.

## 8. Limitations
Clustering identifies similarity patterns and does not prove that
a particular employee will leave or perform poorly. Historical
attrition is descriptive, not an individual prediction. Cluster
interpretations should be validated before making workforce decisions.

## 9. Conclusion
The project implements two clustering approaches and PCA visualization
to explore workforce segments. Final model selection should consider
evaluation metrics, cluster sizes, and business interpretability.
