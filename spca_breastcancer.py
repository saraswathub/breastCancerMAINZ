 First install required packages
!pip install rpy2 scikit-learn matplotlib numpy pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

# Set up Bioconductor and load dataset
def load_breastcancer_data():
    biocmanager = importr('BiocManager')
    biocmanager.install('breastCancerMAINZ')
    bcmainz = importr('breastCancerMAINZ')
    
    with localconverter(globalenv.converter):
        data = bcmainz.data('breastCancerMAINZ')
        eset = data.rx2('eset')
        
    # Convert ExpressionSet to pandas DataFrame
    exprs = np.array(eset.do_slot('exprs')).T
    genes = np.array(eset.do_slot('featureData').do_slot('varMetadata')).flatten()
    samples = np.array(eset.do_slot('phenoData').do_slot('data'))
    
    return pd.DataFrame(exprs, columns=genes, index=samples['geo_accession']), samples

# Load dataset
X_df, metadata = load_breastcancer_data()

# Preprocessing
X = X_df.values
X = StandardScaler().fit_transform(X)

# Filter highly variable genes (top 1000)
variances = np.var(X, axis=0)
top_genes_idx = np.argsort(variances)[-1000:]
X_filtered = X[:, top_genes_idx]

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_filtered)

# Apply SPCA
spca = SparsePCA(n_components=2, alpha=0.1, max_iter=1000)
spca_components = spca.fit_transform(X_filtered)

# Visualization
def plot_components(components, title, metadata):
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(components[:, 0], components[:, 1], 
                         c=metadata['recurrence'].astype('category').cat.codes,
                         cmap='viridis', alpha=0.7)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.colorbar(scatter, label='Recurrence Status')
    plt.show()

plot_components(pca_components, 'PCA Components', metadata)
plot_components(spca_components, 'SPCA Components', metadata)

# Analysis comparison
def analyze_components(model, name):
    print(f"\n{name} Analysis:")
    print(f"Explained variance ratio: {model.explained_variance_ratio_}")
    
    if hasattr(model, 'components_'):
        non_zero = np.sum(model.components_ != 0)
        print(f"Non-zero coefficients: {non_zero}")
        print(f"Sparsity: {1 - (non_zero / model.components_.size):.2%}")

analyze_components(pca, 'PCA')
analyze_components(spca, 'SPCA')

# Gene loading analysis
def top_loading_genes(components, gene_names, n=5):
    for i, component in enumerate(components):
        print(f"\nComponent {i+1} Top Genes:")
        idx = np.argsort(np.abs(component))[::-1][:n]
        for j in idx:
            print(f"{gene_names[j]}: {component[j]:.3f}")

gene_names = X_df.columns[top_genes_idx]
top_loading_genes(pca.components_, gene_names)
top_loading_genes(spca.components_, gene_names)
