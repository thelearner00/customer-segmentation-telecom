"""
Customer Segmentation Models Module

This module contains various clustering algorithms and evaluation methods
specifically designed for customer segmentation tasks with mixed data types.

Key Features:
1. Multiple clustering algorithms (K-Means, Hierarchical, Gaussian Mixture)
2. Automatic optimal cluster number detection
3. Cluster evaluation and interpretation
4. Business-oriented cluster profiling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentationModel:
    """
    A comprehensive customer segmentation model that handles multiple
    clustering algorithms and provides business-oriented insights.
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.n_clusters = None
        self.cluster_labels = None
        self.evaluation_scores = {}
        self.cluster_profiles = None
        
    def find_optimal_clusters(self, X, max_clusters=10, methods=['silhouette', 'elbow']):
        """
        Find optimal number of clusters using multiple methods
        
        Args:
            X: preprocessed feature matrix
            max_clusters: maximum number of clusters to test
            methods: list of methods to use ('silhouette', 'elbow', 'calinski')
            
        Returns:
            dict: results from different methods
        """
        results = {}
        k_range = range(2, max_clusters + 1)
        
        if 'silhouette' in methods:
            silhouette_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            
            results['silhouette'] = {
                'scores': silhouette_scores,
                'optimal_k': k_range[np.argmax(silhouette_scores)],
                'best_score': max(silhouette_scores)
            }
        
        if 'elbow' in methods:
            inertias = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection (can be improved with more sophisticated methods)
            diffs = np.diff(inertias)
            elbow_k = k_range[np.argmin(diffs[1:]) + 2]  # +2 to account for indexing
            
            results['elbow'] = {
                'inertias': inertias,
                'optimal_k': elbow_k
            }
        
        if 'calinski' in methods:
            calinski_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = calinski_harabasz_score(X, labels)
                calinski_scores.append(score)
            
            results['calinski'] = {
                'scores': calinski_scores,
                'optimal_k': k_range[np.argmax(calinski_scores)],
                'best_score': max(calinski_scores)
            }
        
        return results
    
    def plot_cluster_analysis(self, results, save_path=None):
        """
        Plot cluster optimization results
        
        Args:
            results: results from find_optimal_clusters
            save_path: path to save the plot
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, data) in enumerate(results.items()):
            ax = axes[i]
            
            if method == 'silhouette':
                k_range = range(2, len(data['scores']) + 2)
                ax.plot(k_range, data['scores'], 'bo-', linewidth=2, markersize=8)
                ax.axvline(data['optimal_k'], color='red', linestyle='--', 
                          label=f'Optimal k={data["optimal_k"]}')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif method == 'elbow':
                k_range = range(2, len(data['inertias']) + 2)
                ax.plot(k_range, data['inertias'], 'ro-', linewidth=2, markersize=8)
                ax.axvline(data['optimal_k'], color='blue', linestyle='--', 
                          label=f'Elbow k={data["optimal_k"]}')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Inertia (WCSS)')
                ax.set_title('Elbow Method')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif method == 'calinski':
                k_range = range(2, len(data['scores']) + 2)
                ax.plot(k_range, data['scores'], 'go-', linewidth=2, markersize=8)
                ax.axvline(data['optimal_k'], color='orange', linestyle='--', 
                          label=f'Optimal k={data["optimal_k"]}')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Calinski-Harabasz Score')
                ax.set_title('Calinski-Harabasz Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def fit_clustering_model(self, X, n_clusters, algorithm='kmeans', **kwargs):
        """
        Fit a clustering model to the data
        
        Args:
            X: preprocessed feature matrix
            n_clusters: number of clusters
            algorithm: clustering algorithm ('kmeans', 'hierarchical', 'gmm')
            **kwargs: additional parameters for the clustering algorithm
            
        Returns:
            fitted model and cluster labels
        """
        self.n_clusters = n_clusters
        self.model_type = algorithm
        
        if algorithm == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, 
                               n_init=10, **kwargs)
        elif algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        elif algorithm == 'gmm':
            self.model = GaussianMixture(n_components=n_clusters, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Fit the model
        if algorithm == 'gmm':
            self.cluster_labels = self.model.fit_predict(X)
        else:
            self.cluster_labels = self.model.fit_predict(X)
        
        # Calculate evaluation metrics
        self.evaluation_scores = {
            'silhouette': silhouette_score(X, self.cluster_labels),
            'calinski_harabasz': calinski_harabasz_score(X, self.cluster_labels),
            'davies_bouldin': davies_bouldin_score(X, self.cluster_labels)
        }
        
        return self.model, self.cluster_labels
    
    def create_cluster_profiles(self, df_original, cluster_labels, 
                              numerical_features, categorical_features):
        """
        Create comprehensive cluster profiles for business interpretation
        
        Args:
            df_original: original dataframe with meaningful column names
            cluster_labels: cluster assignments
            numerical_features: list of numerical feature names
            categorical_features: list of categorical feature names
            
        Returns:
            dict: cluster profiles with business metrics
        """
        df_with_clusters = df_original.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        profiles = {}
        
        for cluster in sorted(df_with_clusters['Cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                'numerical_stats': {},
                'categorical_stats': {}
            }
            
            # Numerical features analysis
            for feature in numerical_features:
                if feature in cluster_data.columns:
                    profile['numerical_stats'][feature] = {
                        'mean': cluster_data[feature].mean(),
                        'median': cluster_data[feature].median(),
                        'std': cluster_data[feature].std()
                    }
            
            # Categorical features analysis
            for feature in categorical_features:
                if feature in cluster_data.columns:
                    value_counts = cluster_data[feature].value_counts(normalize=True) * 100
                    profile['categorical_stats'][feature] = value_counts.to_dict()
            
            profiles[cluster] = profile
        
        self.cluster_profiles = profiles
        return profiles
    
    def generate_business_insights(self, profiles):
        """
        Generate business insights from cluster profiles
        
        Args:
            profiles: cluster profiles from create_cluster_profiles
            
        Returns:
            dict: business insights for each cluster
        """
        insights = {}
        
        for cluster, profile in profiles.items():
            cluster_insights = {
                'segment_name': self._generate_segment_name(profile),
                'key_characteristics': self._identify_key_characteristics(profile),
                'business_value': self._assess_business_value(profile),
                'recommended_actions': self._generate_recommendations(profile)
            }
            insights[cluster] = cluster_insights
        
        return insights
    
    def _generate_segment_name(self, profile):
        """Generate a descriptive name for the customer segment"""
        # This is a simplified version - can be made more sophisticated
        size_desc = "Large" if profile['percentage'] > 30 else "Medium" if profile['percentage'] > 15 else "Small"
        
        # Look for distinguishing characteristics
        if 'MonthlyCharges' in profile['numerical_stats']:
            avg_charges = profile['numerical_stats']['MonthlyCharges']['mean']
            if avg_charges > 70:
                value_desc = "Premium"
            elif avg_charges < 40:
                value_desc = "Budget"
            else:
                value_desc = "Standard"
        else:
            value_desc = "General"
        
        return f"{value_desc} {size_desc} Segment"
    
    def _identify_key_characteristics(self, profile):
        """Identify key characteristics of the cluster"""
        characteristics = []
        
        # Revenue characteristics
        if 'MonthlyCharges' in profile['numerical_stats']:
            avg_revenue = profile['numerical_stats']['MonthlyCharges']['mean']
            if avg_revenue > 70:
                characteristics.append("High revenue customers")
            elif avg_revenue < 40:
                characteristics.append("Low revenue customers")
        
        # Tenure characteristics
        if 'tenure' in profile['numerical_stats']:
            avg_tenure = profile['numerical_stats']['tenure']['mean']
            if avg_tenure > 36:
                characteristics.append("Long-term customers")
            elif avg_tenure < 12:
                characteristics.append("New customers")
        
        # Churn risk
        if 'Churn' in profile['categorical_stats']:
            churn_rate = profile['categorical_stats']['Churn'].get('Yes', 0)
            if churn_rate > 30:
                characteristics.append("High churn risk")
            elif churn_rate < 15:
                characteristics.append("Low churn risk")
        
        return characteristics
    
    def _assess_business_value(self, profile):
        """Assess the business value of the segment"""
        value_score = 0
        
        # Revenue contribution
        if 'MonthlyCharges' in profile['numerical_stats']:
            avg_revenue = profile['numerical_stats']['MonthlyCharges']['mean']
            value_score += min(avg_revenue / 100, 1) * 0.4  # Max 0.4 points for revenue
        
        # Size contribution
        size_contribution = profile['percentage'] / 100
        value_score += size_contribution * 0.3  # Max 0.3 points for size
        
        # Loyalty (inverse of churn)
        if 'Churn' in profile['categorical_stats']:
            churn_rate = profile['categorical_stats']['Churn'].get('Yes', 0) / 100
            loyalty_score = 1 - churn_rate
            value_score += loyalty_score * 0.3  # Max 0.3 points for loyalty
        
        # Classify business value
        if value_score > 0.7:
            return "High Value"
        elif value_score > 0.4:
            return "Medium Value"
        else:
            return "Low Value"
    
    def _generate_recommendations(self, profile):
        """Generate business recommendations for the segment"""
        recommendations = []
        
        # Revenue-based recommendations
        if 'MonthlyCharges' in profile['numerical_stats']:
            avg_revenue = profile['numerical_stats']['MonthlyCharges']['mean']
            if avg_revenue < 40:
                recommendations.append("Target for upselling and cross-selling")
            elif avg_revenue > 70:
                recommendations.append("Focus on retention with premium support")
        
        # Churn-based recommendations
        if 'Churn' in profile['categorical_stats']:
            churn_rate = profile['categorical_stats']['Churn'].get('Yes', 0)
            if churn_rate > 30:
                recommendations.append("Implement proactive retention campaigns")
            elif churn_rate < 15:
                recommendations.append("Leverage for referral programs")
        
        # Contract-based recommendations
        if 'Contract' in profile['categorical_stats']:
            monthly_rate = profile['categorical_stats']['Contract'].get('Month-to-month', 0)
            if monthly_rate > 60:
                recommendations.append("Offer long-term contract incentives")
        
        return recommendations
    
    def visualize_clusters(self, X, cluster_labels, save_path=None):
        """
        Visualize clusters using PCA dimensionality reduction
        
        Args:
            X: preprocessed feature matrix
            cluster_labels: cluster assignments
            save_path: path to save the plot
        """
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        
        for i in range(self.n_clusters):
            cluster_points = X_pca[cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], label=f'Cluster {i}', 
                       alpha=0.7, s=50)
        
        plt.xlabel(f'First Principal Component (Var: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Var: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('Customer Segments Visualization (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_evaluation_report(self):
        """
        Get a comprehensive evaluation report of the clustering model
        
        Returns:
            dict: evaluation metrics and summary
        """
        report = {
            'model_type': self.model_type,
            'n_clusters': self.n_clusters,
            'evaluation_scores': self.evaluation_scores,
            'cluster_sizes': pd.Series(self.cluster_labels).value_counts().sort_index().to_dict()
        }
        
        # Add interpretation of scores
        report['score_interpretation'] = {
            'silhouette': self._interpret_silhouette_score(self.evaluation_scores['silhouette']),
            'calinski_harabasz': "Higher is better (cluster separation)",
            'davies_bouldin': "Lower is better (cluster compactness)"
        }
        
        return report
    
    def _interpret_silhouette_score(self, score):
        """Interpret silhouette score"""
        if score > 0.7:
            return "Excellent clustering structure"
        elif score > 0.5:
            return "Good clustering structure"
        elif score > 0.25:
            return "Weak clustering structure"
        else:
            return "Poor clustering structure"


def compare_clustering_algorithms(X, n_clusters, algorithms=['kmeans', 'hierarchical', 'gmm']):
    """
    Compare different clustering algorithms on the same dataset
    
    Args:
        X: preprocessed feature matrix
        n_clusters: number of clusters to use
        algorithms: list of algorithms to compare
        
    Returns:
        dict: comparison results
    """
    results = {}
    
    for algorithm in algorithms:
        model = CustomerSegmentationModel()
        model.fit_clustering_model(X, n_clusters, algorithm)
        
        results[algorithm] = {
            'model': model,
            'labels': model.cluster_labels,
            'scores': model.evaluation_scores
        }
    
    return results
