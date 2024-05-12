**Embedding - Clustering** (**EC**) Community Detection Framework is an extension of popular community detection algorithms - **Louvain** and **Leiden**. **EC** improves the results by creating a stable initial partitioning using the embedded representation of nodes. It is a three-step method of modularity optimization: 

**Step 1:**  Find an embedding function $\mathcal{E} \colon V \to \mathbb{R}^s$ which embeds each node of graph $G$ into a $s$-dimensional vector $\mathcal{E}(v) = \{z_1, z_2, \ldots, z_s\}$, where $s \ll n$.

**Step 2:** Run the clustering algorithm on the obtained latent representation $\mathcal{E}$ to get partition $\mathbf{A}$. 

**Step 3:** Use $\mathbf{A}$ as an initializing partitioning for the **Louvain** or  **Leiden** algorithm. The outcome partition $\mathbf{P}$ is a final result of the **EC** framework, that maximizes the modularity.

This repository provides Jupyter notebooks and codes used to test and validate the **EC** method, as described in papers:

>Bartosz Pankratz, Bogumił Kamiński, and Paweł Prałat. Community detection supported
by node embeddings (searching for a suitable method). In Hocine Cherifi, Rosario Nunzio
Mantegna, Luis M. Rocha, Chantal Cherifi, and Salvatore Micciche, editors, *Complex Networks
and Their Applications XI*, pages 221–232, Cham, 2023. Springer International Publishing. doi:10.1007/978-3-031-21131-7\_17.

and

>Bartosz Pankratz, Bogumił Kamiński, and Paweł Prałat. Performance of Community Detection Algorithms Supported by Node Embeddings, (under review).


### Table of contents

1. [Citing](#cite)<br>
2. [Repository Content](#repo) <br>
2. [Requirements](#req) <br>
2. [Datasets](#data) <br>

<a class="anchor" id="cite"></a>
### Citing 

If you find the **EC** metho useful in your research, please consider citing the following paper:

>Bartosz Pankratz, Bogumił Kamiński, and Paweł Prałat. Community detection supported
by node embeddings (searching for a suitable method). In Hocine Cherifi, Rosario Nunzio
Mantegna, Luis M. Rocha, Chantal Cherifi, and Salvatore Micciche, editors, *Complex Networks
and Their Applications XI*, pages 221–232, Cham, 2023. Springer International Publishing. doi:10.1007/978-3-031-21131-7\_17.

<a class="anchor" id="repo"></a>
### Repository Content 

Content of the repository:
- folder ``ECCD`` contains files used in a main experiment, namely a comparision of different combinations of embedding and clustering in the **EC** framework: 
     - ``EC_main.jl`` is a core of the experiment. For a given sweep of parameters, script generates [ABCD](https://github.com/bkamins/ABCDGraphGenerator.jl) synthetic network. Then, it computes modularity and **AMI** score for the baseline community detection algorithms (**Louvain**, **Leiden** and **ECG**). Finally, script run embedding algorithm on a given graph (**LE**, **LLE**, **GraRep**, **LINE**, **HOPE**, **deepWalk**, **node2vec**, all taken from the [OpenNE](https://github.com/thunlp/OpenNE) package) and call the ``run_clustering.jl`` which calculates the **EC** results for a given embedding.
     - ``EC_main_facebook.jl`` is a modified version of ``EC_main.jl``  for Facebook Datasets (avalaible at [**GEMSEC**] GitHub (https://github.com/benedekrozemberczki/GEMSEC?tab=readme-ov-file) repository or [SNAP](https://snap.stanford.edu/data/gemsec-Facebook.html))
     - ``run_clustering.jl`` is a script that run various clustering algorithms (**k-means**, **Gaussian Mixture Model** and **HDBSCAN**) on top of a given embedding and finds a **EC-Louvain** and **EC-Leiden** partitions.  
- ``ECCD_Experiment.ipynb`` is a notebook with analysis of experiment results.

<a class="anchor" id="req"></a>
### Requirements:

Codes are created in ``Julia 1.7.2`` with following packages:
```
- Clustering v0.14.2
- CGE v2.0.0
- DataFrames v1.3.2
- Formatting v0.4.2
- Graphs v1.6.0
- IJulia v1.23.2
- LaTeXStrings v1.3.0
- Latexify v0.15.14
- Plots v1.27.4
- PyCall v1.93.1
- StatsBase v0.33.13
- StatsPlots v0.14.33
```

Additionaly ``Python 3.7.17`` was used with following packages:
```
- community 0.16
- gensim 3.1.0
- hdbscan 0.8.33
- igraph 0.10.8
- leidenalg 0.10.2
- networkx 2.0
- numpy 1.21.6
- openne 0.0.0 
- partition_igraph 0.0.4
- scikit-learn 1.0.2
- scipy 1.1.0
- tensorflow 1.13.1
```

<a class="anchor" id="data"></a>
### Datasets

Two datasets are provided with this repository:
- [graphs](https://drive.google.com/file/d/1wekpTdvgsKwSuPhNstXl-c9zIM3uLsdK/view?usp=sharing) - dataset contains **ABCD** graphs used in the experiment and separate files with community and degree distribution used to generate **ABCD** networks.
- [results](https://drive.google.com/file/d/1MfPL8JOWObJmlvdErOCGsd65zENlJqwe/view?usp=sharing) - aggregated results of experiments.
Facebook Datasets are avalaible at [**GEMSEC**] GitHub (https://github.com/benedekrozemberczki/GEMSEC?tab=readme-ov-file) repository or [SNAP](https://snap.stanford.edu/data/gemsec-Facebook.html).