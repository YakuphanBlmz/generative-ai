# Graph Convolutional Networks (GCN)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. What are Graph Convolutional Networks (GCNs)?](#2-what-are-graph-convolutional-networks-gcns)
- [3. How GCNs Work: The Graph Convolution Operation](#3-how-gcns-work-the-graph-convolution-operation)
  - [3.1. Graph Representation](#31-graph-representation)
  - [3.2. The Message Passing Paradigm](#32-the-message-passing-paradigm)
  - [3.3. Layer-wise Propagation Rule](#33-layer-wise-propagation-rule)
- [4. Applications of GCNs](#4-applications-of-gcns)
- [5. Advantages and Limitations](#5-advantages-and-limitations)
  - [5.1. Advantages](#51-advantages)
  - [5.2. Limitations](#52-limitations)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

---

### 1. Introduction
In the realm of machine learning, most prominent deep learning architectures, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), are designed to process data with Euclidean structures like images (grids) or text (sequences). However, a vast amount of real-world data inherently exists in **non-Euclidean structures**, often represented as **graphs**. Examples include social networks, molecular structures, citation networks, and knowledge graphs. Traditional deep learning models struggle to effectively process such irregular and complex data structures due to their fixed input size and lack of spatial invariance.

**Graph Convolutional Networks (GCNs)** emerged as a powerful paradigm to extend the success of deep learning to graph-structured data. Introduced by Kipf and Welling in 2017, GCNs provide an elegant and efficient framework for learning **node embeddings** and **graph-level representations** by performing convolution-like operations directly on graphs. They facilitate the propagation and aggregation of feature information across nodes and their neighbors, enabling the model to capture both local and global structural properties of the graph. This document delves into the fundamental principles, mechanisms, applications, and considerations of GCNs, elucidating their significance in the rapidly evolving field of **Generative AI** and beyond.

### 2. What are Graph Convolutional Networks (GCNs)?
**Graph Convolutional Networks (GCNs)** are a class of neural networks specifically designed to operate on **graph-structured data**. Conceptually, they adapt the notion of convolution, which is central to CNNs for image processing, to the non-Euclidean domain of graphs. In a traditional CNN, a filter slides over a grid of pixels, performing a weighted sum of neighboring pixel values to extract features. GCNs extend this idea by defining a similar "convolution" operation that aggregates information from a node's immediate neighbors.

The primary goal of a GCN is to learn a function that maps nodes in a graph to a low-dimensional feature space (i.e., **node embeddings**) such that nodes with similar structural roles or features are mapped close to each other. These embeddings can then be used for various downstream tasks like **node classification**, **link prediction**, or **graph classification**.

Unlike traditional neural networks that require fixed-size input vectors, GCNs can handle graphs of varying sizes and structures. They achieve this by leveraging the **adjacency matrix** (which defines node connections) and the **node feature matrix** (which describes features associated with each node). The core idea revolves around **message passing**, where each node iteratively aggregates information from its neighbors and its own previous state to update its representation. This iterative aggregation allows information to propagate across the graph, enabling nodes to learn representations that incorporate multi-hop neighborhood information.

### 3. How GCNs Work: The Graph Convolution Operation
The operational mechanism of a GCN revolves around a layer-wise propagation rule that updates node representations based on their neighbors' representations. This process can be understood through the lens of **message passing** and **spectral graph theory**.

#### 3.1. Graph Representation
Before delving into the convolution operation, it's crucial to understand how a graph is represented in a machine learning context:
*   **Adjacency Matrix (A):** For a graph with `N` nodes, `A` is an `N x N` matrix where `A_ij = 1` if there is an edge between node `i` and node `j`, and `0` otherwise. For undirected graphs, `A` is symmetric.
*   **Node Feature Matrix (X):** `X` is an `N x F` matrix, where `N` is the number of nodes and `F` is the number of features per node. `X_i` represents the feature vector for node `i`.

#### 3.2. The Message Passing Paradigm
At its core, a GCN layer operates on the principle of **message passing**. In each layer, every node:
1.  **Receives messages** (feature vectors) from its immediate neighbors.
2.  **Aggregates** these messages, often along with its own previous state, using a permutation-invariant function (e.g., sum, mean, max).
3.  **Transforms** the aggregated information using a neural network layer (e.g., a linear transformation followed by an activation function) to produce its new representation for the next layer.

This process is repeated across multiple GCN layers, allowing nodes to gather information from increasingly distant neighbors, thereby capturing global structural patterns.

#### 3.3. Layer-wise Propagation Rule
One of the most widely recognized and simplified GCN models was proposed by Kipf and Welling (2017), deriving from a first-order approximation of spectral graph convolutions. The layer-wise propagation rule for a GCN layer can be formulated as:

`H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))`

Where:
*   `H^(l)`: The input feature matrix for the `l`-th layer. For the first layer (`l=0`), `H^(0) = X` (the initial node feature matrix).
*   `H^(l+1)`: The output feature matrix for the `l`-th layer, which becomes the input for the `(l+1)`-th layer. Each row `h_i^(l+1)` is the updated feature representation (embedding) for node `i`.
*   `Ã = A + I`: The **adjacency matrix** `A` with added self-loops (`I` is the identity matrix). Adding self-loops ensures that a node includes its own features when aggregating information from its neighborhood.
*   `D̃`: The **degree matrix** of `Ã`. `D̃_ii = Σ_j Ã_ij`. The normalization `D̃^(-1/2) Ã D̃^(-1/2)` is a symmetric normalization which helps in preventing exploding/vanishing gradients and can be interpreted as normalizing the aggregated features by the degree of the nodes.
*   `W^(l)`: A layer-specific **learnable weight matrix**. This matrix transforms the aggregated features. It's the primary source of parameters in the GCN layer.
*   `σ`: An **activation function**, such as ReLU (`max(0, x)`), applied element-wise.

This formulation effectively smooths node features over the graph structure. Each node's new feature vector is a linear transformation of the average of its neighbors' feature vectors (including itself), where the averaging is weighted by the node degrees. By stacking multiple such layers, GCNs can learn increasingly complex and abstract representations of nodes, incorporating information from wider neighborhoods.

### 4. Applications of GCNs
GCNs have demonstrated remarkable success across a diverse range of applications, particularly in domains where data naturally exhibits graph structures:

*   **Node Classification:** This is a semi-supervised learning task where the goal is to predict the label of unlabeled nodes given a partially labeled graph. GCNs excel by propagating label information and learning expressive node embeddings that capture both node features and network topology. Examples include classifying papers in a citation network (e.g., Cora, Citeseer datasets) or users in a social network.
*   **Link Prediction:** Predicting the existence of a link between two nodes, or recommending new connections. This is crucial in social networks (friend recommendations), knowledge graphs (relationship inference), and drug discovery (predicting molecular interactions). GCNs can learn representations for pairs of nodes that can then be used by a simple classifier to predict link probability.
*   **Graph Classification:** Assigning a label to an entire graph. This is common in cheminformatics (classifying molecules based on their properties), material science, and bioinformatics. GCNs can be combined with **graph pooling** layers to create a single, fixed-size representation for the entire graph, which can then be fed into a standard classifier.
*   **Recommendation Systems:** Modeling user-item interaction graphs to provide personalized recommendations. GCNs can capture complex collaborative filtering patterns by jointly learning user and item embeddings.
*   **Drug Discovery and Material Science:** Representing molecules as graphs where atoms are nodes and bonds are edges. GCNs can predict molecular properties, identify potential drug candidates, or design new materials with desired characteristics.
*   **Traffic Prediction:** Modeling road networks as graphs and predicting traffic flow by considering spatial and temporal dependencies.
*   **Social Network Analysis:** Detecting communities, identifying influential users, or understanding information diffusion patterns.

These applications highlight the versatility of GCNs in extracting meaningful insights from complex relational data, pushing the boundaries of what's possible with machine learning on non-Euclidean structures.

### 5. Advantages and Limitations

Like any powerful machine learning model, GCNs come with their own set of advantages and limitations that define their适用性和挑战。

#### 5.1. Advantages
*   **Handles Non-Euclidean Data:** The most significant advantage is their ability to directly process and learn from graph-structured data, which traditional deep learning models cannot.
*   **Captures Relational Information:** GCNs naturally leverage the connectivity patterns (edges) between nodes, allowing them to learn representations that encode both node features and the underlying graph topology.
*   **Effective for Semi-Supervised Learning:** Due to the message passing mechanism, information from labeled nodes can propagate to unlabeled nodes, making GCNs particularly effective in semi-supervised learning scenarios where only a small fraction of nodes have labels.
*   **Parameter Efficiency:** The weight matrices `W^(l)` are shared across all nodes within a layer, leading to a relatively small number of parameters compared to approaches that might require separate embeddings for each node, especially for large graphs.
*   **Inductive Capabilities (with variations):** While the original GCN formulation is transductive (requires the full graph during training), variations like GraphSAGE have introduced inductive capabilities, allowing them to generalize to unseen nodes or even entirely new graphs.

#### 5.2. Limitations
*   **Scalability Challenges:** Training GCNs on very large graphs (millions or billions of nodes) can be computationally intensive, especially for full-batch gradient descent. Sampling-based methods (e.g., GraphSAGE, PinSAGE) have been developed to address this.
*   **Over-smoothing:** As more GCN layers are stacked, node representations tend to become increasingly similar, leading to a loss of discriminative power. This phenomenon, known as **over-smoothing**, limits the depth of GCNs, typically to 2-4 layers.
*   **Transductivity:** The original GCN model, being dependent on the full adjacency matrix, is inherently transductive. It cannot directly handle new, unseen nodes without retraining or specific adaptations.
*   **Capturing Long-Range Dependencies:** Due to the local nature of the convolution operation (aggregating from direct neighbors), capturing dependencies between distant nodes in a deep graph can be challenging and often leads to over-smoothing.
*   **Feature Engineering:** The quality of initial node features (X) significantly impacts GCN performance. In many real-world scenarios, rich node features may not be readily available, requiring creative approaches for feature engineering.
*   **Interpretability:** Like many deep learning models, understanding exactly *why* a GCN makes a certain prediction can be difficult, posing challenges for interpretability and trustworthiness.

Despite these limitations, ongoing research continues to develop more advanced GCN architectures and training strategies to mitigate these issues, further expanding their applicability.

### 6. Code Example
Here's a simplified conceptual Python code snippet demonstrating a single GCN layer using NumPy for clarity, without deep learning frameworks. This illustrates the core propagation rule.

```python
import numpy as np

def gcn_layer(adj_matrix, node_features, weights):
    """
    Implements a single simplified GCN layer based on Kipf & Welling (2017).

    Args:
        adj_matrix (np.array): The adjacency matrix A of the graph (N x N).
        node_features (np.array): The input node feature matrix H_l (N x F_in).
        weights (np.array): The learnable weight matrix W_l (F_in x F_out).

    Returns:
        np.array: The output node feature matrix H_{l+1} (N x F_out) after ReLU activation.
    """
    N = adj_matrix.shape[0]
    
    # 1. Add self-loops to the adjacency matrix (Ã = A + I)
    adj_tilde = adj_matrix + np.eye(N)
    
    # 2. Compute the degree matrix D̃
    # D̃_ii = sum_j(Ã_ij)
    deg_tilde = np.sum(adj_tilde, axis=1)
    
    # 3. Compute D̃^(-1/2)
    deg_tilde_inv_sqrt = np.power(deg_tilde, -0.5)
    deg_tilde_inv_sqrt[np.isinf(deg_tilde_inv_sqrt)] = 0 # Handle nodes with degree 0
    D_tilde_inv_sqrt = np.diag(deg_tilde_inv_sqrt)
    
    # 4. Normalize Ã: D̃^(-1/2) Ã D̃^(-1/2)
    normalized_adj = D_tilde_inv_sqrt @ adj_tilde @ D_tilde_inv_sqrt
    
    # 5. Apply the GCN propagation rule: H^(l+1) = σ(normalized_adj @ H^(l) @ W^(l))
    # Linear transformation
    transformed_features = normalized_adj @ node_features @ weights
    
    # Activation function (ReLU)
    output_features = np.maximum(0, transformed_features)
    
    return output_features

# Example Usage:
# Define a small graph (e.g., 4 nodes)
# Node 0 -- Node 1
# |        /
# Node 2 -- Node 3

adj = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
])

# Initial node features (e.g., 4 nodes, 2 features each)
features = np.array([
    [1.0, 0.5],
    [0.1, 1.2],
    [0.8, 0.3],
    [1.5, 0.7]
])

# Learnable weights (e.g., transforming 2 input features to 3 output features)
# In a real scenario, these would be initialized randomly and learned via backpropagation.
weights_layer1 = np.array([
    [0.2, -0.1, 0.5],
    [0.3, 0.6, -0.2]
])

# Apply one GCN layer
output_features_layer1 = gcn_layer(adj, features, weights_layer1)
print("Output features after 1st GCN layer:\n", output_features_layer1)

# To stack layers, the output_features_layer1 would become the input node_features for the next layer.
# And you would define new weights for the second layer.
weights_layer2 = np.array([
    [-0.1, 0.3],
    [0.4, 0.1],
    [0.2, -0.4]
])
output_features_layer2 = gcn_layer(adj, output_features_layer1, weights_layer2)
print("\nOutput features after 2nd GCN layer:\n", output_features_layer2)

(End of code example section)
```

### 7. Conclusion
**Graph Convolutional Networks (GCNs)** represent a transformative advancement in machine learning, extending the power of deep learning to the rich and complex domain of graph-structured data. By adapting the concept of convolution to irregular topologies, GCNs enable the effective learning of **node embeddings** that encapsulate both local features and global graph topology through an elegant **message passing** paradigm.

Their ability to process non-Euclidean data has unlocked new possibilities across diverse fields, from **node classification** in social and citation networks to **drug discovery** and **recommendation systems**. While challenges such as **scalability** for massive graphs and the **over-smoothing** phenomenon persist, continuous research and development are actively addressing these limitations, leading to more robust and powerful GCN variants.

As Generative AI continues to evolve, GCNs are poised to play an increasingly crucial role, particularly in tasks involving graph generation, conditional graph generation, and learning representations for complex relational structures. Their foundational contribution has paved the way for a new era of graph neural networks, demonstrating the immense potential of integrating deep learning with graph theory for understanding and generating intricate network data.
---
<br>

<a name="türkçe-içerik"></a>
## Grafik Evrişimsel Ağlar (GCN)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Grafik Evrişimsel Ağlar (GCN) Nedir?](#2-grafik-evrişimsel-ağlar-gcn-nedir)
- [3. GCN'ler Nasıl Çalışır: Grafik Evrişim Operasyonu](#3-gcns-nasıl-çalışır-grafik-evrişim-operasyonu)
  - [3.1. Grafik Temsili](#31-grafik-temsili)
  - [3.2. Mesaj İletme Paradigması](#32-mesaj-iletme-paradigması)
  - [3.3. Katman Bazında Yayılım Kuralı](#33-katman-bazında-yayılım-kuralı)
- [4. GCN Uygulamaları](#4-gcn-uygulamaları)
- [5. Avantajları ve Sınırlamaları](#5-avantajları-ve-sınırlamaları)
  - [5.1. Avantajları](#51-avantajları)
  - [5.2. Sınırlamaları](#52-sınırlamaları)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

---

### 1. Giriş
Makine öğrenimi alanında, Geleneksel Evrişimsel Ağlar (CNN'ler) ve Tekrarlayan Sinir Ağları (RNN'ler) gibi öne çıkan derin öğrenme mimarilerinin çoğu, görüntüler (ızgaralar) veya metinler (diziler) gibi **Öklid yapılarına** sahip verileri işlemek üzere tasarlanmıştır. Ancak, gerçek dünyadaki verilerin büyük bir kısmı doğası gereği **Öklid dışı yapılarda** bulunur ve genellikle **grafikler** olarak temsil edilir. Sosyal ağlar, moleküler yapılar, alıntı ağları ve bilgi grafikleri bu duruma örnek teşkil eder. Geleneksel derin öğrenme modelleri, sabit girdi boyutları ve uzamsal değişmezlik eksikliği nedeniyle bu tür düzensiz ve karmaşık veri yapılarını etkili bir şekilde işleme konusunda zorlanır.

**Grafik Evrişimsel Ağlar (GCN'ler)**, derin öğrenmenin başarısını grafik yapılı verilere genişletmek için güçlü bir paradigma olarak ortaya çıkmıştır. Kipf ve Welling tarafından 2017'de tanıtılan GCN'ler, evrişim benzeri işlemleri doğrudan grafikler üzerinde gerçekleştirerek **düğüm gömüleri (node embeddings)** ve **grafik düzeyinde temsiller** öğrenmek için zarif ve verimli bir çerçeve sunar. Düğümler ve komşuları arasında öznitelik bilgisinin yayılmasını ve toplanmasını kolaylaştırarak modelin grafiğin hem yerel hem de küresel yapısal özelliklerini yakalamasına olanak tanırlar. Bu belge, GCN'lerin temel prensiplerini, mekanizmalarını, uygulamalarını ve hususlarını detaylandırarak, **Üretken Yapay Zeka** alanındaki önemlerini açıklamaktadır.

### 2. Grafik Evrişimsel Ağlar (GCN) Nedir?
**Grafik Evrişimsel Ağlar (GCN'ler)**, özellikle **grafik yapılı veriler** üzerinde çalışmak üzere tasarlanmış bir sinir ağı sınıfıdır. Kavramsal olarak, görüntü işleme için CNN'lerin merkezi olan evrişim kavramını, grafiklerin Öklid dışı alanına uyarlarlar. Geleneksel bir CNN'de, bir filtre piksel ızgarası üzerinde kayarak komşu piksel değerlerinin ağırlıklı toplamını gerçekleştirerek özellikler çıkarır. GCN'ler, bu fikri bir düğümün en yakın komşularından bilgi toplayan benzer bir "evrişim" işlemi tanımlayarak genişletirler.

Bir GCN'nin temel amacı, grafikteki düğümleri düşük boyutlu bir özellik uzayına (yani **düğüm gömüleri**) eşleyen bir fonksiyon öğrenmektir, öyle ki benzer yapısal rollere veya özelliklere sahip düğümler birbirine yakın eşlenir. Bu gömüler daha sonra **düğüm sınıflandırması**, **bağlantı tahmini** veya **grafik sınıflandırması** gibi çeşitli alt görevler için kullanılabilir.

Sabit boyutlu girdi vektörleri gerektiren geleneksel sinir ağlarının aksine, GCN'ler farklı boyut ve yapıda grafikleri işleyebilir. Bunu, **komşuluk matrisi (adjacency matrix)** (düğüm bağlantılarını tanımlayan) ve **düğüm özellik matrisi (node feature matrix)** (her düğümle ilişkili özellikleri açıklayan) kullanarak başarırlar. Temel fikir, her düğümün önceki durumunun yanı sıra komşularından gelen bilgiyi yinelemeli olarak toplayarak temsilini güncellediği **mesaj iletimi** etrafında döner. Bu yinelemeli toplama, bilginin grafik boyunca yayılmasına olanak tanır ve düğümlerin çoklu atlamalı komşuluk bilgilerini içeren temsiller öğrenmesini sağlar.

### 3. GCN'ler Nasıl Çalışır: Grafik Evrişim Operasyonu
Bir GCN'nin operasyonel mekanizması, düğüm temsillerini komşularının temsillerine göre güncelleyen katman bazında bir yayılım kuralı etrafında döner. Bu süreç, **mesaj iletimi** ve **spektral grafik teorisi** perspektifinden anlaşılabilir.

#### 3.1. Grafik Temsili
Evrişim işlemine geçmeden önce, bir grafiğin makine öğrenimi bağlamında nasıl temsil edildiğini anlamak çok önemlidir:
*   **Komşuluk Matrisi (A):** `N` düğümlü bir grafik için, `A_ij = 1` eğer `i` düğümü ile `j` düğümü arasında bir kenar varsa ve `0` aksi halde olan `N x N` boyutunda bir matristir. Yönsüz grafikler için `A` simetriktir.
*   **Düğüm Özellik Matrisi (X):** `X`, `N` düğüm sayısı ve `F` her düğüm başına düşen özellik sayısı olmak üzere `N x F` boyutunda bir matristir. `X_i`, `i` düğümünün özellik vektörünü temsil eder.

#### 3.2. Mesaj İletme Paradigması
Özünde, bir GCN katmanı **mesaj iletme** prensibiyle çalışır. Her katmanda, her düğüm:
1.  En yakın komşularından **mesajlar** (özellik vektörleri) alır.
2.  Bu mesajları, genellikle kendi önceki durumuyla birlikte, permütasyon-değişmez bir fonksiyon (örn. toplam, ortalama, maksimum) kullanarak **toplar (aggregate)**.
3.  Toplanan bilgiyi, bir sinir ağı katmanı (örn. doğrusal dönüşüm ve ardından bir aktivasyon fonksiyonu) kullanarak **dönüştürerek** bir sonraki katman için yeni temsilini üretir.

Bu süreç birden fazla GCN katmanı boyunca tekrar edilir, bu da düğümlerin giderek daha uzak komşulardan bilgi toplamasını ve böylece küresel yapısal örüntüleri yakalamasını sağlar.

#### 3.3. Katman Bazında Yayılım Kuralı
Kipf ve Welling (2017) tarafından önerilen ve spektral grafik evrişimlerinin birinci dereceden bir yaklaşımından türetilen en yaygın ve basitleştirilmiş GCN modellerinden biri, katman bazında yayılım kuralını şu şekilde formüle eder:

`H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))`

Burada:
*   `H^(l)`: `l`. katman için girdi özellik matrisi. İlk katman (`l=0`) için `H^(0) = X` (başlangıç düğüm özellik matrisi).
*   `H^(l+1)`: `l`. katman için çıktı özellik matrisi, `(l+1)`. katman için girdi olur. Her satır `h_i^(l+1)`, `i` düğümü için güncellenmiş özellik temsilidir (gömüsüdür).
*   `Ã = A + I`: Kendi kendine döngüler eklenmiş **komşuluk matrisi** `A` (`I` birim matristir). Kendi kendine döngü eklemek, bir düğümün komşuluk bilgilerini toplarken kendi özelliklerini de içermesini sağlar.
*   `D̃`: `Ã`'nin **derece matrisi**. `D̃_ii = Σ_j Ã_ij`. Normalizasyon `D̃^(-1/2) Ã D̃^(-1/2)`, patlayan/kaybolan gradyanları önlemeye yardımcı olan simetrik bir normalizasyondur ve toplanan özelliklerin düğümlerin derecesine göre normalleştirilmesi olarak yorumlanabilir.
*   `W^(l)`: Katmana özgü **öğrenilebilir ağırlık matrisi**. Bu matris, toplanan özellikleri dönüştürür. GCN katmanındaki ana parametre kaynağıdır.
*   `σ`: Öğeleri bağımsız olarak uygulanan ReLU (`max(0, x)`) gibi bir **aktivasyon fonksiyonu**.

Bu formülasyon, düğüm özelliklerini grafik yapısı üzerinde etkili bir şekilde düzeltir. Her düğümün yeni özellik vektörü, komşularının (kendi de dahil olmak üzere) özellik vektörlerinin, düğüm dereceleriyle ağırlıklandırılmış ortalamasının doğrusal bir dönüşümüdür. Birden fazla katman biriktirerek GCN'ler, daha geniş komşuluklardaki bilgiyi birleştirerek giderek daha karmaşık ve soyut düğüm temsilleri öğrenebilir.

### 4. GCN Uygulamaları
GCN'ler, özellikle verilerin doğal olarak grafik yapıları sergilediği çeşitli uygulamalarda dikkat çekici bir başarı göstermiştir:

*   **Düğüm Sınıflandırması:** Bu, kısmen etiketlenmiş bir grafik verildiğinde etiketsiz düğümlerin etiketini tahmin etmeyi amaçlayan yarı denetimli bir öğrenme görevidir. GCN'ler, etiket bilgilerini yayarak ve hem düğüm özelliklerini hem de ağ topolojisini yakalayan etkileyici düğüm gömüleri öğrenerek üstün başarı gösterir. Örnekler arasında bir alıntı ağındaki (örn. Cora, Citeseer veri kümeleri) makaleleri veya bir sosyal ağdaki kullanıcıları sınıflandırmak yer alır.
*   **Bağlantı Tahmini:** İki düğüm arasındaki bir bağlantının varlığını tahmin etmek veya yeni bağlantıları önermek. Bu, sosyal ağlarda (arkadaş önerileri), bilgi grafiklerinde (ilişki çıkarımı) ve ilaç keşfinde (moleküler etkileşimleri tahmin etme) çok önemlidir. GCN'ler, daha sonra bağlantı olasılığını tahmin etmek için basit bir sınıflandırıcı tarafından kullanılabilecek düğüm çiftleri için temsiller öğrenebilir.
*   **Grafik Sınıflandırması:** Tüm bir grafiğe bir etiket atamak. Bu, kemoinformatikte (molekülleri özelliklerine göre sınıflandırma), malzeme biliminde ve biyoinformatikte yaygındır. GCN'ler, tüm grafik için tek, sabit boyutlu bir temsil oluşturmak için **grafik havuzlama (graph pooling)** katmanlarıyla birleştirilebilir ve bu daha sonra standart bir sınıflandırıcıya beslenebilir.
*   **Öneri Sistemleri:** Kişiselleştirilmiş öneriler sunmak için kullanıcı-öğe etkileşim grafiklerini modellemek. GCN'ler, kullanıcı ve öğe gömülerini birlikte öğrenerek karmaşık işbirlikçi filtreleme örüntülerini yakalayabilir.
*   **İlaç Keşfi ve Malzeme Bilimi:** Molekülleri, atomların düğüm ve bağların kenar olduğu grafikler olarak temsil etmek. GCN'ler moleküler özellikleri tahmin edebilir, potansiyel ilaç adaylarını tanımlayabilir veya istenen özelliklere sahip yeni malzemeler tasarlayabilir.
*   **Trafik Tahmini:** Yol ağlarını grafikler olarak modellemek ve uzamsal ve zamansal bağımlılıkları dikkate alarak trafik akışını tahmin etmek.
*   **Sosyal Ağ Analizi:** Toplulukları tespit etmek, etkili kullanıcıları belirlemek veya bilgi yayılım modellerini anlamak.

Bu uygulamalar, GCN'lerin karmaşık ilişkisel verilerden anlamlı içgörüler elde etmedeki çok yönlülüğünü vurgulamakta ve Öklid dışı yapılar üzerinde makine öğrenimi ile mümkün olanın sınırlarını zorlamaktadır.

### 5. Avantajları ve Sınırlamaları

Her güçlü makine öğrenimi modelinde olduğu gibi, GCN'lerin de uygulanabilirliklerini ve zorluklarını tanımlayan kendine özgü avantajları ve sınırlamaları vardır.

#### 5.1. Avantajları
*   **Öklid Dışı Verileri İşler:** En önemli avantajı, geleneksel derin öğrenme modellerinin yapamadığı grafik yapılı verileri doğrudan işleme ve öğrenme yetenekleridir.
*   **İlişkisel Bilgileri Yakalar:** GCN'ler, düğümler arasındaki bağlantı örüntülerini (kenarları) doğal olarak kullanır, bu da onların hem düğüm özelliklerini hem de temel grafik topolojisini kodlayan temsiller öğrenmelerini sağlar.
*   **Yarı Denetimli Öğrenme için Etkili:** Mesaj iletim mekanizması sayesinde, etiketli düğümlerden gelen bilgiler etiketsiz düğümlere yayılabilir, bu da GCN'leri yalnızca küçük bir düğüm grubunun etiketi olduğu yarı denetimli öğrenme senaryolarında özellikle etkili kılar.
*   **Parametre Verimliliği:** `W^(l)` ağırlık matrisleri bir katmandaki tüm düğümler arasında paylaşılır, bu da özellikle büyük grafikler için her düğüm için ayrı gömüler gerektirebilecek yaklaşımlara kıyasla nispeten az sayıda parametreye yol açar.
*   **Tümevarımsal Yetenekler (çeşitliliklerle):** Orijinal GCN formülasyonu transdüktif (eğitim sırasında tüm grafiği gerektirir) olsa da, GraphSAGE gibi varyasyonlar, görülmemiş düğümlere ve hatta tamamen yeni grafiklere genelleşmelerine olanak tanıyan tümevarımsal yetenekler sunmuştur.

#### 5.2. Sınırlamaları
*   **Ölçeklenebilirlik Zorlukları:** Çok büyük grafikler (milyonlarca veya milyarlarca düğüm) üzerinde GCN'leri eğitmek, özellikle tam-parti gradyan inişi için hesaplama açısından yoğun olabilir. Örnekleme tabanlı yöntemler (örn. GraphSAGE, PinSAGE) bu sorunu çözmek için geliştirilmiştir.
*   **Aşırı Düzgünleştirme (Over-smoothing):** Daha fazla GCN katmanı istiflendikçe, düğüm temsilleri giderek daha benzer hale gelme eğilimindedir ve bu da ayırt edici gücün kaybına yol açar. **Aşırı düzgünleştirme** olarak bilinen bu fenomen, GCN'lerin derinliğini genellikle 2-4 katmanla sınırlar.
*   **Transdüktivite:** Tüm komşuluk matrisine bağımlı olan orijinal GCN modeli, doğal olarak transdüktiftir. Yeniden eğitim veya belirli adaptasyonlar olmaksızın yeni, görülmemiş düğümleri doğrudan işleyemez.
*   **Uzun Menzilli Bağımlılıkları Yakalama:** Evrişim işleminin yerel doğası (doğrudan komşulardan toplama) nedeniyle, derin bir grafikteki uzak düğümler arasındaki bağımlılıkları yakalamak zor olabilir ve genellikle aşırı düzgünleştirmeye yol açar.
*   **Öznitelik Mühendisliği:** Başlangıç düğüm özelliklerinin (X) kalitesi GCN performansını önemli ölçüde etkiler. Birçok gerçek dünya senaryosunda, zengin düğüm özellikleri kolayca bulunmayabilir, bu da öznitelik mühendisliği için yaratıcı yaklaşımlar gerektirir.
*   **Yorumlanabilirlik:** Birçok derin öğrenme modelinde olduğu gibi, bir GCN'nin belirli bir tahmini *neden* yaptığını tam olarak anlamak zor olabilir, bu da yorumlanabilirlik ve güvenilirlik açısından zorluklar yaratır.

Bu sınırlamalara rağmen, devam eden araştırmalar bu sorunları azaltmak için daha gelişmiş GCN mimarileri ve eğitim stratejileri geliştirmeye devam etmekte ve uygulanabilirliklerini daha da genişletmektedir.

### 6. Kod Örneği
Burada, derin öğrenme framework'leri olmadan, yalnızca NumPy kullanarak tek bir GCN katmanını gösteren basitleştirilmiş bir kavramsal Python kod parçacığı bulunmaktadır. Bu, çekirdek yayılım kuralını göstermektedir.

```python
import numpy as np

def gcn_layer(adj_matrix, node_features, weights):
    """
    Kipf & Welling (2017) tabanlı basitleştirilmiş tek bir GCN katmanını uygular.

    Argümanlar:
        adj_matrix (np.array): Grafiğin komşuluk matrisi A (N x N).
        node_features (np.array): Giriş düğüm özellik matrisi H_l (N x F_in).
        weights (np.array): Öğrenilebilir ağırlık matrisi W_l (F_in x F_out).

    Dönüş:
        np.array: ReLU aktivasyonundan sonraki çıktı düğüm özellik matrisi H_{l+1} (N x F_out).
    """
    N = adj_matrix.shape[0]
    
    # 1. Komşuluk matrisine kendi kendine döngüler ekle (Ã = A + I)
    adj_tilde = adj_matrix + np.eye(N)
    
    # 2. Derece matrisi D̃'yi hesapla
    # D̃_ii = sum_j(Ã_ij)
    deg_tilde = np.sum(adj_tilde, axis=1)
    
    # 3. D̃^(-1/2)'yi hesapla
    deg_tilde_inv_sqrt = np.power(deg_tilde, -0.5)
    deg_tilde_inv_sqrt[np.isinf(deg_tilde_inv_sqrt)] = 0 # Derecesi 0 olan düğümleri ele al
    D_tilde_inv_sqrt = np.diag(deg_tilde_inv_sqrt)
    
    # 4. Ã'yi normalize et: D̃^(-1/2) Ã D̃^(-1/2)
    normalized_adj = D_tilde_inv_sqrt @ adj_tilde @ D_tilde_inv_sqrt
    
    # 5. GCN yayılım kuralını uygula: H^(l+1) = σ(normalized_adj @ H^(l) @ W^(l))
    # Doğrusal dönüşüm
    transformed_features = normalized_adj @ node_features @ weights
    
    # Aktivasyon fonksiyonu (ReLU)
    output_features = np.maximum(0, transformed_features)
    
    return output_features

# Örnek Kullanım:
# Küçük bir grafik tanımla (örn. 4 düğüm)
# Düğüm 0 -- Düğüm 1
# |        /
# Düğüm 2 -- Düğüm 3

adj = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
])

# Başlangıç düğüm özellikleri (örn. 4 düğüm, her biri 2 özellik)
features = np.array([
    [1.0, 0.5],
    [0.1, 1.2],
    [0.8, 0.3],
    [1.5, 0.7]
])

# Öğrenilebilir ağırlıklar (örn. 2 giriş özelliğini 3 çıkış özelliğine dönüştürme)
# Gerçek bir senaryoda, bunlar rastgele başlatılır ve geri yayılım yoluyla öğrenilir.
weights_layer1 = np.array([
    [0.2, -0.1, 0.5],
    [0.3, 0.6, -0.2]
])

# Bir GCN katmanı uygula
output_features_layer1 = gcn_layer(adj, features, weights_layer1)
print("1. GCN katmanından sonraki çıktı özellikleri:\n", output_features_layer1)

# Katmanları üst üste bindirmek için, output_features_layer1 bir sonraki katman için giriş node_features olur.
# Ve ikinci katman için yeni ağırlıklar tanımlamanız gerekir.
weights_layer2 = np.array([
    [-0.1, 0.3],
    [0.4, 0.1],
    [0.2, -0.4]
])
output_features_layer2 = gcn_layer(adj, output_features_layer1, weights_layer2)
print("\n2. GCN katmanından sonraki çıktı özellikleri:\n", output_features_layer2)

(Kod örneği bölümünün sonu)
```

### 7. Sonuç
**Grafik Evrişimsel Ağlar (GCN'ler)**, derin öğrenmenin gücünü zengin ve karmaşık grafik yapılı veri alanına genişleterek makine öğreniminde dönüştürücü bir ilerlemeyi temsil etmektedir. Evrişim kavramını düzensiz topolojilere uyarlayarak, GCN'ler zarif bir **mesaj iletimi** paradigması aracılığıyla hem yerel özellikleri hem de küresel grafik topolojisini kapsayan **düğüm gömülerini** etkili bir şekilde öğrenmeyi mümkün kılar.

Öklid dışı verileri işleme yetenekleri, sosyal ve alıntı ağlarındaki **düğüm sınıflandırmasından** **ilaç keşfi** ve **öneri sistemlerine** kadar çeşitli alanlarda yeni olanaklar yaratmıştır. Büyük grafikler için **ölçeklenebilirlik** ve **aşırı düzgünleştirme** fenomeni gibi zorluklar devam etse de, sürekli araştırma ve geliştirme bu sınırlamaları aktif olarak ele almakta ve daha sağlam ve güçlü GCN varyantlarına yol açmaktadır.

Üretken Yapay Zeka gelişmeye devam ettikçe, GCN'ler özellikle grafik üretimi, koşullu grafik üretimi ve karmaşık ilişkisel yapılar için temsiller öğrenme gibi görevlerde giderek daha önemli bir rol oynamaya hazırlanmaktadır. Temel katkıları, derin öğrenmeyi grafik teorisi ile entegre ederek karmaşık ağ verilerini anlama ve üretme konusundaki muazzam potansiyeli gösteren yeni bir grafik sinir ağları çağının önünü açmıştır.

