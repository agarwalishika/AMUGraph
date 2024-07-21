# AMUGraph - Active Mix up for Graph Anomaly detection

Graph anomaly detection (GAD) aims to learn a function that can detect anomalous entities in a graph. In this tiny paper, we explore node-level anomaly detection. Although graph data is prevalent, ground truth labels are hard to acquire. Hence, active learning can be used to obtain soft labels for efficient supervised learning. Furthermore, we can make better use of the scarce labeled data by applying a data augmentation strategy such as mixup. In this paper, we propose AMUGraph, a method for Active anomaly detection that uses MixUp for data augmentation using soft label on Graphs.


## Files:
- amu_graph.py: contains class code for AMUGraph
- amubandits_stream.py / amubandits_pool.py: code that combines NeurONAL-S and NeurONAL-P respectively with AMUGraph
- classification_layer.py: contains code for the classification layer that takes in latent space embeddings and returns an anomaly score
- graph_dataset.py: contains custom Dataset object for the AMUGraph setting
- neuronal_p_graph.py / neuronal_s_graph.py: uses NeurONAL-P and NeurONAL-S from [1]] for the graph anomaly detection task
- run*.py: python scripts to run the corresponding experiments
- semi_vae.py: SemiVAE from [2]
- soft_labeler_oracle.py.py: labeler based on a GCN for soft labeling
- utils.py: random functions used throughout experiments

## Experiments
First, create Node2Vec embeddings:
1. copy the 'embeddings' folder and prepend a short hand for your dataset (for FraudYelpDataset, I went with yelp_embeddings)
2. modify necessary lines of code in mod_node2vec.py
3. run it - it will generate files of embeddings (to ensure there are minimal CUDA OOM errors)

Next, run an experiment:
1. simply run a 'run_*.py' file