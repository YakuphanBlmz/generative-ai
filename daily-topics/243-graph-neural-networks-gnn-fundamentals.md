# Graph Neural Networks (GNN) Fundamentals

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Graph Theory Fundamentals](#2-graph-theory-fundamentals)
    - [2.1. Graphs, Nodes, and Edges](#21-graphs-nodes-and-edges)
    - [2.2. Node and Edge Features](#22-node-and-edge-features)
    - [2.3. Adjacency Matrix](#23-adjacency-matrix)
- [3. The Need for GNNs: Limitations of Traditional Machine Learning](#3-the-need-for-gnns-limitations-of-traditional-machine-learning)
- [4. Core Concepts of Graph Neural Networks](#4-core-concepts-of-graph-neural-networks)
    - [4.1. Message Passing Paradigm](#41-message-passing-paradigm)
    - [4.2. Neighborhood Aggregation and Update Functions](#42-neighborhood-aggregation-and-update-functions)
    - [4.3. Learning Node Embeddings](#43-learning-node-embeddings)
- [5. Common GNN Architectures (Brief Overview)](#5-common-gnn-architectures-brief-overview)
    - [5.1. Graph Convolutional Networks (GCNs)](#51-graph-convolutional-networks-gcns)
    - [5.2. GraphSAGE](#52-graphsage)
    - [5.3. Graph Attention Networks (GATs)](#53-graph-attention-networks-gats)
- [6. Applications of GNNs](#6-applications-of-gnns)
- [7. Code Example](#7-code-example)
- [8. Conclusion](#8-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
In the rapidly evolving landscape of artificial intelligence, **Graph Neural Networks (GNNs)** have emerged as a powerful paradigm for machine learning on **graph-structured data**. Traditional machine learning models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), are primarily designed for Euclidean data like images (grids) and text (sequences). However, a vast amount of real-world data inherently exists in non-Euclidean formats, characterized by complex relationships and interdependencies, which can be naturally represented as graphs. Examples include social networks, molecular structures, recommender systems, and citation networks. GNNs extend the principles of deep learning to these intricate structures, enabling the extraction of meaningful patterns and facilitating tasks such as node classification, link prediction, and graph classification. This document delves into the fundamental concepts underpinning GNNs, exploring their architecture, operational mechanisms, and diverse applications.

<a name="2-graph-theory-fundamentals"></a>
## 2. Graph Theory Fundamentals
To comprehend GNNs, a foundational understanding of graph theory is essential. A **graph** is a mathematical structure used to model pairwise relations between objects.

<a name="21-graphs-nodes-and-edges"></a>
### 2.1. Graphs, Nodes, and Edges
A graph `G` is formally defined as a pair `(V, E)`, where `V` is a set of **nodes** (or vertices) and `E` is a set of **edges** (or links) connecting pairs of nodes.
-   **Nodes**: Represent individual entities within the graph (e.g., people in a social network, atoms in a molecule).
-   **Edges**: Represent relationships or interactions between nodes (e.g., friendship between people, chemical bond between atoms). Edges can be **undirected** (symmetrical relationship, like friendship) or **directed** (asymmetrical, like following someone on social media). They can also be **weighted**, indicating the strength or cost of a relationship.

<a name="22-node-and-edge-features"></a>
### 2.2. Node and Edge Features
Beyond just connectivity, nodes and edges can possess attributes, known as **features**.
-   **Node Features**: These are vectors describing properties of individual nodes (e.g., age, occupation, location for a person node; atomic number, electronegativity for an atom node). These are typically represented as `X_v` for node `v`.
-   **Edge Features**: These describe properties of the connections themselves (e.g., relationship type, duration of friendship, bond type in a molecule). These are typically represented as `X_uv` for an edge between nodes `u` and `v`.

<a name="23-adjacency-matrix"></a>
### 2.3. Adjacency Matrix
The connectivity of a graph can be mathematically represented using an **adjacency matrix**, `A`. For a graph with `N` nodes, `A` is an `N x N` matrix where `A_ij = 1` if an edge exists between node `i` and node `j`, and `A_ij = 0` otherwise. For weighted graphs, `A_ij` would contain the weight of the edge. For directed graphs, `A_ij` might not be equal to `A_ji`.

<a name="3-the-need-for-gnns-limitations-of-traditional-machine-learning"></a>
## 3. The Need for GNNs: Limitations of Traditional Machine Learning
Traditional deep learning models struggle with graph-structured data due to several inherent challenges:
1.  **Irregular Structure**: Graphs are non-Euclidean; they do not have a fixed spatial grid or sequence. Nodes can have varying numbers of neighbors, making it difficult to apply standard operations like convolutions that rely on local grid-like structures.
2.  **Permutation Invariance**: The order in which nodes are listed or input to a model should not change the outcome for tasks like graph classification. Traditional neural networks are not inherently permutation invariant.
3.  **Variable Size**: Graphs can have varying numbers of nodes and edges, making fixed-size input layers problematic.
4.  **Complex Dependencies**: Relationships between nodes can be indirect and multi-hop, requiring models to capture both local and global dependencies.
GNNs are specifically designed to address these challenges by operating directly on the graph structure and learning representations that incorporate both node features and connectivity information.

<a name="4-core-concepts-of-graph-neural-networks"></a>
## 4. Core Concepts of Graph Neural Networks
The central idea behind GNNs is to learn **node embeddings** (vector representations) that encode information about the node's features and its local neighborhood structure. This is typically achieved through an iterative **message passing** scheme.

<a name="41-message-passing-paradigm"></a>
### 4.1. Message Passing Paradigm
The **message passing paradigm** is the cornerstone of most GNN architectures. In essence, each node iteratively aggregates information (messages) from its direct neighbors and combines this aggregated information with its own current representation to update its state. This process is repeated for a fixed number of layers, allowing information to propagate across the graph and enabling nodes to capture information from increasingly distant neighbors.
For a node `v`, its state at layer `k`, denoted `h_v^(k)`, is updated based on its neighbors' states from layer `k-1`.

<a name="42-neighborhood-aggregation-and-update-functions"></a>
### 4.2. Neighborhood Aggregation and Update Functions
Each layer in a GNN typically involves two main steps for each node `v`:
1.  **Aggregation Function (AGGREGATE)**: This function gathers information from the features of node `v`'s neighbors `u ∈ N(v)`. Common aggregation functions include sum, mean, or max pooling, which are permutation invariant.
    `m_v^(k) = AGGREGATE({h_u^(k-1) | u ∈ N(v)})`
    where `m_v^(k)` is the aggregated message for node `v` at layer `k`.
2.  **Update Function (UPDATE)**: This function combines the aggregated message `m_v^(k)` with the node's own previous representation `h_v^(k-1)` to produce its new representation `h_v^(k)`. This typically involves a neural network layer (e.g., an MLP, ReLU activation).
    `h_v^(k) = UPDATE(h_v^(k-1), m_v^(k))`
By stacking multiple such layers, GNNs can learn complex hierarchical representations of nodes, where each node's embedding reflects its local computational graph.

<a name="43-learning-node-embeddings"></a>
### 4.3. Learning Node Embeddings
The output of a GNN, typically after several message passing layers, is a set of **node embeddings** `H = {h_1^(L), h_2^(L), ..., h_N^(L)}` where `L` is the number of layers. These embeddings are low-dimensional vector representations that capture the structural and feature-based context of each node. These embeddings can then be used for various downstream tasks:
-   **Node Classification**: A classifier (e.g., a softmax layer) can be applied to `h_v^(L)` to predict the label of node `v`.
-   **Link Prediction**: Embeddings of two nodes `h_u^(L)` and `h_v^(L)` can be combined (e.g., via dot product, concatenation) and fed into a classifier to predict the existence or type of an edge between them.
-   **Graph Classification**: All node embeddings in a graph can be aggregated (e.g., by summing or pooling) to form a single graph-level embedding, which can then be used for classifying the entire graph.

<a name="5-common-gnn-architectures-brief-overview"></a>
## 5. Common GNN Architectures (Brief Overview)
Several influential GNN architectures have been proposed, each with variations in their aggregation and update functions.

<a name="51-graph-convolutional-networks-gcns"></a>
### 5.1. Graph Convolutional Networks (GCNs)
**Graph Convolutional Networks (GCNs)** are a foundational GNN model that generalizes the concept of convolution from images to graphs. A GCN layer typically combines information from a node's neighbors by averaging their feature vectors and then applying a linear transformation and a non-linear activation function. The update rule for a GCN often involves the normalized adjacency matrix to prevent issues with nodes having widely varying degrees.

<a name="52-graphsage"></a>
### 5.2. GraphSAGE
**GraphSAGE (Graph SAmpling and aggreGatE)** focuses on learning a generalizable aggregation function that can operate on varying neighborhood sizes. Instead of using the full neighborhood, GraphSAGE samples a fixed number of neighbors for aggregation, which makes it more scalable to large graphs. It employs different aggregation functions like mean, LSTM, or pooling, allowing it to learn inductive representations (i.e., representations for unseen nodes or graphs).

<a name="53-graph-attention-networks-gats"></a>
### 5.3. Graph Attention Networks (GATs)
**Graph Attention Networks (GATs)** introduce an attention mechanism into the message passing framework. Instead of assigning equal weights to all neighbors (as in basic GCNs), GATs learn varying weights for different neighbors based on their features. This allows the model to selectively focus on more important neighbors, potentially improving performance on complex tasks and handling nodes with varying degrees more robustly.

<a name="6-applications-of-gnns"></a>
## 6. Applications of GNNs
GNNs have demonstrated remarkable success across a wide range of domains:
-   **Social Networks**: Friend recommendation, community detection, fake news detection.
-   **Drug Discovery and Chemistry**: Molecular property prediction, drug-target interaction prediction, synthesis planning.
-   **Recommender Systems**: Item recommendation, user-item interaction modeling.
-   **Computer Vision**: Scene graph generation, few-shot learning, point cloud processing.
-   **Natural Language Processing**: Text classification with syntactic graphs, machine translation with semantic graphs.
-   **Traffic Forecasting**: Predicting traffic flow in road networks.
-   **Cybersecurity**: Anomaly detection in network traffic, fraud detection.

<a name="7-code-example"></a>
## 7. Code Example
This short example illustrates a conceptual message passing step for a single node using a simplified mean aggregation.

```python
import torch

def mean_aggregate(node_features, neighbor_indices):
    """
    Conceptual mean aggregation for a node from its neighbors.
    node_features: a dictionary mapping node_id to its feature vector (torch.Tensor)
    neighbor_indices: a list of node_ids representing neighbors
    """
    if not neighbor_indices:
        return torch.zeros_like(node_features[0]) # Return zero vector if no neighbors

    # Stack features of all neighbors
    neighbor_features = [node_features[idx] for idx in neighbor_indices]
    stacked_features = torch.stack(neighbor_features)

    # Compute the mean
    aggregated_message = torch.mean(stacked_features, dim=0)
    return aggregated_message

def gnn_layer_update(current_node_feature, aggregated_message, weight_matrix):
    """
    Conceptual update step for a GNN layer.
    Combines current node feature with aggregated message.
    """
    # Concatenate current feature and aggregated message
    combined_feature = torch.cat([current_node_feature, aggregated_message], dim=0)

    # Apply a linear transformation (e.g., a neural network layer)
    updated_feature = torch.matmul(combined_feature, weight_matrix.T)
    # Apply a non-linear activation (e.g., ReLU, not shown here for brevity)
    return updated_feature

# Example usage:
# Define some initial node features (e.g., 3 nodes, 2 features each)
node_features = {
    0: torch.tensor([1.0, 2.0]),
    1: torch.tensor([3.0, 4.0]),
    2: torch.tensor([5.0, 6.0])
}

# Define adjacency list (who are neighbors of node 0)
node_0_neighbors = [1, 2]

# Step 1: Aggregate messages for node 0
aggregated_msg_for_node_0 = mean_aggregate(node_features, node_0_neighbors)
print(f"Aggregated message for node 0: {aggregated_msg_for_node_0}")

# Step 2: Update node 0's feature
# Assume a simple weight matrix for transformation
# Input dimension for update is (initial_feature_dim + aggregated_msg_dim)
# Output dimension can be anything, let's say 4 for this example.
weight_matrix_dim_in = node_features[0].shape[0] + aggregated_msg_for_node_0.shape[0]
weight_matrix_dim_out = 4
weight_matrix = torch.rand(weight_matrix_dim_out, weight_matrix_dim_in)

updated_feature_for_node_0 = gnn_layer_update(
    node_features[0],
    aggregated_msg_for_node_0,
    weight_matrix
)
print(f"Updated feature for node 0: {updated_feature_for_node_0}")

(End of code example section)
```

<a name="8-conclusion"></a>
## 8. Conclusion
Graph Neural Networks represent a pivotal advancement in machine learning, offering a powerful framework for processing and learning from complex graph-structured data. By extending the principles of deep learning to non-Euclidean domains through iterative message passing, GNNs enable models to capture intricate relationships and extract rich node and graph-level embeddings. From their fundamental graph theory underpinnings to sophisticated architectures like GCNs, GraphSAGE, and GATs, GNNs have opened new avenues for solving challenging problems across diverse fields. As the prevalence of graph-structured data continues to grow, GNNs are poised to play an increasingly critical role in pushing the boundaries of artificial intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## Graf Sinir Ağları (GNN) Temelleri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Graf Teorisi Temelleri](#2-graf-teorisi-temelleri)
    - [2.1. Graflar, Düğümler ve Kenarlar](#21-graflar-düğümler-ve-kenarlar)
    - [2.2. Düğüm ve Kenar Özellikleri](#22-düğüm-ve-kenar-özellikleri)
    - [2.3. Komşuluk Matrisi](#23-komşuluk-matrisi)
- [3. GNN'lere Duyulan İhtiyaç: Geleneksel Makine Öğrenmesinin Sınırlılıkları](#3-gnnlere-duyulan-ihtiyaç-geleneksel-makine-öğrenmesinin-sınırlılıkları)
- [4. Graf Sinir Ağlarının Temel Kavramları](#4-graf-sinir-ağlarının-temel-kavramları)
    - [4.1. Mesajlaşma Paradigması](#41-mesajlaşma-paradigması)
    - [4.2. Komşuluk Toplama ve Güncelleme Fonksiyonları](#42-komşuluk-toplama-ve-güncelleme-fonksiyonları)
    - [4.3. Düğüm Gömülmelerini Öğrenme](#43-düğüm-gömülmelerini-öğrenme)
- [5. Yaygın GNN Mimarileri (Kısa Bir Bakış)](#5-yaygın-gnn-mimarileri-kısa-bir-bakış)
    - [5.1. Graf Evrişimli Ağlar (GCN'ler)](#51-graf-evrişimli-ağlar-gcns)
    - [5.2. GraphSAGE](#52-graphsage)
    - [5.3. Graf Dikkat Ağları (GAT'lar)](#53-graf-dikkat-ağları-gats)
- [6. GNN'lerin Uygulama Alanları](#6-gnnlerin-uygulama-alanları)
- [7. Kod Örneği](#7-kod-örneği)
- [8. Sonuç](#8-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Yapay zeka alanındaki hızlı gelişimde, **Graf Sinir Ağları (GNN'ler)**, **graf yapılı veriler** üzerinde makine öğrenimi için güçlü bir paradigma olarak ortaya çıkmıştır. Evrişimli sinir ağları (CNN'ler) ve tekrarlayan sinir ağları (RNN'ler) gibi geleneksel makine öğrenimi modelleri, öncelikli olarak görüntüler (ızgaralar) ve metin (diziler) gibi Öklid verileri için tasarlanmıştır. Ancak, gerçek dünya verilerinin büyük bir kısmı, karmaşık ilişkiler ve karşılıklı bağımlılıklarla karakterize edilen, doğal olarak graf olarak temsil edilebilen Öklid dışı biçimlerde mevcuttur. Sosyal ağlar, moleküler yapılar, tavsiye sistemleri ve atıf ağları bu duruma örnek teşkil eder. GNN'ler, derin öğrenme prensiplerini bu karmaşık yapılara genişleterek, anlamlı örüntülerin çıkarılmasını ve düğüm sınıflandırması, bağlantı tahmini ve graf sınıflandırması gibi görevleri kolaylaştırmayı mümkün kılar. Bu belge, GNN'lerin altında yatan temel kavramları incelemekte, mimarilerini, çalışma mekanizmalarını ve çeşitli uygulama alanlarını keşfetmektedir.

<a name="2-graf-teorisi-temelleri"></a>
## 2. Graf Teorisi Temelleri
GNN'leri kavramak için, graf teorisinin temel bir anlayışı şarttır. Bir **graf**, nesneler arasındaki ikili ilişkileri modellemek için kullanılan matematiksel bir yapıdır.

<a name="21-graflar-düğümler-ve-kenarlar"></a>
### 2.1. Graflar, Düğümler ve Kenarlar
Bir graf `G` resmi olarak `(V, E)` çifti olarak tanımlanır; burada `V` **düğümler** (veya köşeler) kümesi ve `E` düğüm çiftlerini birbirine bağlayan **kenarlar** (veya bağlantılar) kümesidir.
-   **Düğümler**: Graf içindeki bireysel varlıkları temsil eder (örneğin, bir sosyal ağdaki kişiler, bir moleküldeki atomlar).
-   **Kenarlar**: Düğümler arasındaki ilişkileri veya etkileşimleri temsil eder (örneğin, insanlar arasındaki arkadaşlık, atomlar arasındaki kimyasal bağ). Kenarlar **yönsüz** (arkadaşlık gibi simetrik ilişki) veya **yönlü** (sosyal medyada birini takip etmek gibi asimetrik) olabilir. Ayrıca, bir ilişkinin gücünü veya maliyetini gösteren **ağırlıklı** da olabilirler.

<a name="22-düğüm-ve-kenar-özellikleri"></a>
### 2.2. Düğüm ve Kenar Özellikleri
Sadece bağlantının ötesinde, düğümler ve kenarlar, **özellikler** olarak bilinen niteliklere sahip olabilir.
-   **Düğüm Özellikleri**: Bunlar, bireysel düğümlerin özelliklerini tanımlayan vektörlerdir (örneğin, bir kişi düğümü için yaş, meslek, konum; bir atom düğümü için atom numarası, elektronegatiflik). Bunlar tipik olarak `v` düğümü için `X_v` olarak temsil edilir.
-   **Kenar Özellikleri**: Bunlar, bağlantıların kendilerinin özelliklerini tanımlar (örneğin, ilişki türü, arkadaşlığın süresi, bir moleküldeki bağ türü). Bunlar tipik olarak `u` ve `v` düğümleri arasındaki bir kenar için `X_uv` olarak temsil edilir.

<a name="23-komşuluk-matrisi"></a>
### 2.3. Komşuluk Matrisi
Bir grafın bağlantısı, bir **komşuluk matrisi**, `A` kullanılarak matematiksel olarak temsil edilebilir. `N` düğümlü bir graf için, `A`, `i` düğümü ile `j` düğümü arasında bir kenar varsa `A_ij = 1`, aksi takdirde `A_ij = 0` olan bir `N x N` matristir. Ağırlıklı graflar için `A_ij`, kenarın ağırlığını içerir. Yönlü graflar için `A_ij`, `A_ji`'ye eşit olmayabilir.

<a name="3-gnnlere-duyulan-ihtiyaç-geleneksel-makine-öğrenmesinin-sınırlılıkları"></a>
## 3. GNN'lere Duyulan İhtiyaç: Geleneksel Makine Öğrenmesinin Sınırlılıkları
Geleneksel derin öğrenme modelleri, bazı içsel zorluklar nedeniyle graf yapılı verilerle başa çıkmakta zorlanır:
1.  **Düzensiz Yapı**: Graflar Öklid dışıdır; sabit bir uzamsal ızgaraya veya diziye sahip değillerdir. Düğümlerin değişen sayıda komşusu olabilir, bu da yerel ızgara benzeri yapılara dayanan evrişimler gibi standart operasyonları uygulamayı zorlaştırır.
2.  **Permütasyon Değişmezliği**: Düğümlerin listelenme veya bir modele giriş sırası, graf sınıflandırması gibi görevlerin sonucunu değiştirmemelidir. Geleneksel sinir ağları doğal olarak permütasyon değişmez değildir.
3.  **Değişken Boyut**: Graflar değişen sayıda düğüm ve kenara sahip olabilir, bu da sabit boyutlu girdi katmanlarını sorunlu hale getirir.
4.  **Karmaşık Bağımlılıklar**: Düğümler arasındaki ilişkiler dolaylı ve çok atlamalı olabilir, modellerin hem yerel hem de küresel bağımlılıkları yakalamasını gerektirir.
GNN'ler, bu zorlukları doğrudan graf yapısı üzerinde çalışarak ve hem düğüm özelliklerini hem de bağlantı bilgilerini içeren temsilleri öğrenerek ele almak üzere özel olarak tasarlanmıştır.

<a name="4-graf-sinir-ağlarının-temel-kavramları"></a>
## 4. Graf Sinir Ağlarının Temel Kavramları
GNN'lerin temel fikri, düğümün özelliklerini ve yerel komşuluk yapısını kodlayan **düğüm gömülmelerini** (vektör temsilleri) öğrenmektir. Bu genellikle yinelemeli bir **mesajlaşma** şeması aracılığıyla başarılır.

<a name="41-mesajlaşma-paradigması"></a>
### 4.1. Mesajlaşma Paradigması
**Mesajlaşma paradigması**, çoğu GNN mimarisinin temel taşıdır. Özünde, her düğüm yinelemeli olarak doğrudan komşularından bilgi (mesajlar) toplar ve bu toplu bilgiyi kendi mevcut temsiliyle birleştirerek durumunu günceller. Bu işlem sabit sayıda katman için tekrarlanır, bilginin graf boyunca yayılmasını sağlar ve düğümlerin giderek daha uzak komşulardan bilgi almasını mümkün kılar.
Bir `v` düğümü için, `k` katmanındaki durumu, `h_v^(k)`, `k-1` katmanındaki komşularının durumlarına göre güncellenir.

<a name="42-komşuluk-toplama-ve-güncelleme-fonksiyonları"></a>
### 4.2. Komşuluk Toplama ve Güncelleme Fonksiyonları
Bir GNN'deki her katman, genellikle her `v` düğümü için iki ana adım içerir:
1.  **Toplama Fonksiyonu (AGGREGATE)**: Bu fonksiyon, `v` düğümünün `N(v)` içindeki komşuları `u`'nun özelliklerinden bilgi toplar. Yaygın toplama fonksiyonları arasında toplama, ortalama veya maksimum havuzlama bulunur ve bunlar permütasyon değişmezidir.
    `m_v^(k) = AGGREGATE({h_u^(k-1) | u ∈ N(v)})`
    burada `m_v^(k)`, `k` katmanında `v` düğümü için toplanan mesajdır.
2.  **Güncelleme Fonksiyonu (UPDATE)**: Bu fonksiyon, toplanan mesaj `m_v^(k)` ile düğümün kendi önceki temsili `h_v^(k-1)`'i birleştirerek yeni temsili `h_v^(k)`'yi üretir. Bu genellikle bir sinir ağı katmanı (örneğin, bir MLP, ReLU aktivasyonu) içerir.
    `h_v^(k) = UPDATE(h_v^(k-1), m_v^(k))`
Birden fazla bu tür katmanı istifleyerek, GNN'ler düğümlerin karmaşık hiyerarşik temsillerini öğrenebilir; burada her düğümün gömülmesi, yerel hesaplama grafiğini yansıtır.

<a name="43-düğüm-gömülmelerini-öğrenme"></a>
### 4.3. Düğüm Gömülmelerini Öğrenme
Bir GNN'nin çıktısı, genellikle birkaç mesajlaşma katmanından sonra, bir dizi **düğüm gömülmesidir** `H = {h_1^(L), h_2^(L), ..., h_N^(L)}`, burada `L` katman sayısıdır. Bu gömülmeler, her düğümün yapısal ve özellik tabanlı bağlamını yakalayan düşük boyutlu vektör temsilleridir. Bu gömülmeler daha sonra çeşitli alt görevler için kullanılabilir:
-   **Düğüm Sınıflandırması**: `h_v^(L)`'ye bir sınıflandırıcı (örneğin, bir softmax katmanı) uygulanarak `v` düğümünün etiketi tahmin edilebilir.
-   **Bağlantı Tahmini**: İki düğümün gömülmeleri `h_u^(L)` ve `h_v^(L)` birleştirilebilir (örneğin, nokta çarpımı, birleştirme yoluyla) ve aralarındaki bir kenarın varlığını veya türünü tahmin etmek için bir sınıflandırıcıya beslenebilir.
-   **Graf Sınıflandırması**: Bir grafteki tüm düğüm gömülmeleri toplanarak (örneğin, toplayarak veya havuzlayarak) tek bir graf düzeyinde gömülme oluşturulabilir ve bu daha sonra tüm grafı sınıflandırmak için kullanılabilir.

<a name="5-yaygın-gnn-mimarileri-kısa-bir-bakış"></a>
## 5. Yaygın GNN Mimarileri (Kısa Bir Bakış)
Her biri toplama ve güncelleme fonksiyonlarında farklılıklar gösteren, etkili birçok GNN mimarisi önerilmiştir.

<a name="51-graf-evrişimli-ağlar-gcns"></a>
### 5.1. Graf Evrişimli Ağlar (GCN'ler)
**Graf Evrişimli Ağlar (GCN'ler)**, evrişim kavramını görüntülerden graflara genelleştiren temel bir GNN modelidir. Bir GCN katmanı, tipik olarak bir düğümün komşularının özellik vektörlerini ortalamasını alır ve ardından doğrusal bir dönüşüm ve doğrusal olmayan bir aktivasyon fonksiyonu uygular. Bir GCN için güncelleme kuralı, düğümlerin derecelerinin büyük ölçüde değişmesiyle ortaya çıkan sorunları önlemek için genellikle normalize edilmiş komşuluk matrisini içerir.

<a name="52-graphsage"></a>
### 5.2. GraphSAGE
**GraphSAGE (Graph SAmpling and aggreGatE)**, değişen komşuluk boyutlarında çalışabilen genellenebilir bir toplama fonksiyonu öğrenmeye odaklanır. Tam komşuluğu kullanmak yerine, GraphSAGE toplama için sabit sayıda komşu örnekler, bu da onu büyük graflara daha ölçeklenebilir hale getirir. Ortalama, LSTM veya havuzlama gibi farklı toplama fonksiyonları kullanır ve endüktif temsiller öğrenmesine olanak tanır (yani, görülmemiş düğümler veya graflar için temsiller).

<a name="53-graf-dikkat-ağları-gats"></a>
### 5.3. Graf Dikkat Ağları (GAT'lar)
**Graf Dikkat Ağları (GAT'lar)**, mesajlaşma çerçevesine bir dikkat mekanizması getirir. Tüm komşulara eşit ağırlıklar atamak (temel GCN'lerde olduğu gibi) yerine, GAT'lar farklı komşular için özelliklerine göre değişen ağırlıklar öğrenir. Bu, modelin daha önemli komşulara seçici olarak odaklanmasına olanak tanır, karmaşık görevlerde performansı potansiyel olarak artırır ve değişen derecelere sahip düğümleri daha sağlam bir şekilde işler.

<a name="6-gnnlerin-uygulama-alanları"></a>
## 6. GNN'lerin Uygulama Alanları
GNN'ler, çok çeşitli alanlarda dikkat çekici başarılar göstermiştir:
-   **Sosyal Ağlar**: Arkadaş tavsiyesi, topluluk tespiti, sahte haber tespiti.
-   **İlaç Keşfi ve Kimya**: Moleküler özellik tahmini, ilaç-hedef etkileşimi tahmini, sentez planlaması.
-   **Tavsiye Sistemleri**: Öğe tavsiyesi, kullanıcı-öğe etkileşim modellemesi.
-   **Bilgisayar Görüsü**: Sahne grafı üretimi, az örneklemeli öğrenme, nokta bulutu işleme.
-   **Doğal Dil İşleme**: Sözdizimsel graflarla metin sınıflandırması, anlamsal graflarla makine çevirisi.
-   **Trafik Tahmini**: Yol ağlarındaki trafik akışını tahmin etme.
-   **Siber Güvenlik**: Ağ trafiğinde anomali tespiti, dolandırıcılık tespiti.

<a name="7-kod-örneği"></a>
## 7. Kod Örneği
Bu kısa örnek, basitleştirilmiş bir ortalama toplama kullanarak tek bir düğüm için kavramsal bir mesajlaşma adımını göstermektedir.

```python
import torch

def mean_aggregate(node_features, neighbor_indices):
    """
    Bir düğüm için komşularından kavramsal ortalama toplama.
    node_features: düğüm kimliğinden özellik vektörüne (torch.Tensor) eşleme yapan bir sözlük
    neighbor_indices: komşuları temsil eden düğüm kimliklerinin bir listesi
    """
    if not neighbor_indices:
        # Komşu yoksa sıfır vektörü döndür
        return torch.zeros_like(node_features[0])

    # Tüm komşuların özelliklerini yığınla
    neighbor_features = [node_features[idx] for idx in neighbor_indices]
    stacked_features = torch.stack(neighbor_features)

    # Ortalamayı hesapla
    aggregated_message = torch.mean(stacked_features, dim=0)
    return aggregated_message

def gnn_layer_update(current_node_feature, aggregated_message, weight_matrix):
    """
    Bir GNN katmanı için kavramsal güncelleme adımı.
    Mevcut düğüm özelliğini toplanan mesajla birleştirir.
    """
    # Mevcut özelliği ve toplanan mesajı birleştir
    combined_feature = torch.cat([current_node_feature, aggregated_message], dim=0)

    # Doğrusal bir dönüşüm uygula (örneğin, bir sinir ağı katmanı)
    updated_feature = torch.matmul(combined_feature, weight_matrix.T)
    # Doğrusal olmayan bir aktivasyon uygula (örneğin, ReLU, kısalık için burada gösterilmemiştir)
    return updated_feature

# Örnek kullanım:
# Bazı başlangıç düğüm özelliklerini tanımla (örneğin, 3 düğüm, her biri 2 özellik)
node_features = {
    0: torch.tensor([1.0, 2.0]),
    1: torch.tensor([3.0, 4.0]),
    2: torch.tensor([5.0, 6.0])
}

# Komşuluk listesini tanımla (0 numaralı düğümün komşuları kimler)
node_0_neighbors = [1, 2]

# Adım 1: 0 numaralı düğüm için mesajları topla
aggregated_msg_for_node_0 = mean_aggregate(node_features, node_0_neighbors)
print(f"0 numaralı düğüm için toplanan mesaj: {aggregated_msg_for_node_0}")

# Adım 2: 0 numaralı düğümün özelliğini güncelle
# Dönüşüm için basit bir ağırlık matrisi varsayalım
# Güncelleme için girdi boyutu (başlangıç_özellik_boyutu + toplanan_mesaj_boyutu)
# Çıktı boyutu herhangi bir şey olabilir, bu örnek için 4 diyelim.
weight_matrix_dim_in = node_features[0].shape[0] + aggregated_msg_for_node_0.shape[0]
weight_matrix_dim_out = 4
weight_matrix = torch.rand(weight_matrix_dim_out, weight_matrix_dim_in)

updated_feature_for_node_0 = gnn_layer_update(
    node_features[0],
    aggregated_msg_for_node_0,
    weight_matrix
)
print(f"0 numaralı düğüm için güncellenmiş özellik: {updated_feature_for_node_0}")

(Kod örneği bölümünün sonu)
```

<a name="8-sonuç"></a>
## 8. Sonuç
Graf Sinir Ağları, karmaşık graf yapılı verileri işlemek ve bunlardan öğrenmek için güçlü bir çerçeve sunarak makine öğreniminde önemli bir ilerlemeyi temsil etmektedir. Derin öğrenme prensiplerini yinelemeli mesajlaşma yoluyla Öklid dışı alanlara genişleterek, GNN'ler modellerin karmaşık ilişkileri yakalamasına ve zengin düğüm ve graf düzeyinde gömülmeler çıkarmasına olanak tanır. Temel graf teorisi dayanaklarından GCN'ler, GraphSAGE ve GAT'lar gibi gelişmiş mimarilere kadar, GNN'ler çeşitli alanlarda zorlu sorunları çözmek için yeni yollar açmıştır. Graf yapılı verilerin yaygınlığı artmaya devam ettikçe, GNN'ler yapay zekanın sınırlarını zorlamada giderek daha kritik bir rol oynamaya hazırlanmaktadır.
