# Graph Convolutional Networks (GCN)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Theoretical Foundations of GCNs](#2-theoretical-foundations-of-gcns)
  - [2.1. Graphs and Adjacency Matrices](#21-graphs-and-adjacency-matrices)
  - [2.2. Limitations of Traditional Neural Networks for Graph Data](#22-limitations-of-traditional-neural-networks-for-graph-data)
  - [2.3. Spectral Graph Convolutions and the Graph Laplacian](#23-spectral-graph-convolutions-and-the-graph-laplacian)
  - [2.4. The Simplified GCN Layer](#24-the-simplified-gcn-layer)
- [3. Applications of GCNs](#3-applications-of-gcns)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
Graph Convolutional Networks (GCNs) represent a pivotal advancement in the field of deep learning, extending the powerful concepts of Convolutional Neural Networks (CNNs) from regular grid-like data (such as images) to arbitrarily structured graph data. Graphs, composed of **nodes** (or vertices) and **edges** (or links), are ubiquitous in various domains, modeling relationships between entities. Examples include social networks, molecular structures, citation networks, and knowledge graphs. Traditional neural network architectures, including standard CNNs, are ill-suited for processing graph-structured data due to its irregular topology, varying neighborhood sizes, and the absence of a fixed spatial ordering for nodes.

GCNs overcome these challenges by defining convolution operations directly on graphs. The core idea is to aggregate information from a node's local neighborhood, transforming node features based on the features of its neighbors and the structure of the graph itself. This aggregation process allows GCNs to learn representations that capture both the attributes of individual nodes and their relational context within the larger graph. By iteratively stacking GCN layers, information can propagate across multiple hops, enabling nodes to learn from increasingly distant neighbors. This capability makes GCNs particularly effective for tasks like **node classification**, **link prediction**, and **graph classification**, where understanding the interplay between entities and their connections is crucial. The rise of GCNs has opened new avenues for applying deep learning techniques to complex, non-Euclidean data, significantly impacting fields from bioinformatics to recommendation systems and natural language processing.

<a name="2-theoretical-foundations-of-gcns"></a>
## 2. Theoretical Foundations of GCNs
Understanding GCNs requires a grasp of basic graph theory concepts and the motivation behind adapting convolutional operations to graphs.

<a name="21-graphs-and-adjacency-matrices"></a>
### 2.1. Graphs and Adjacency Matrices
A graph `G = (V, E)` consists of a set of **nodes** `V` (where `|V| = N` is the number of nodes) and a set of **edges** `E`. Edges represent connections between nodes. Graphs can be directed or undirected, weighted or unweighted. For an unweighted graph, its structure is typically represented by an **adjacency matrix** `A` of size `N x N`, where `A_ij = 1` if an edge exists between node `i` and node `j`, and `A_ij = 0` otherwise. For undirected graphs, `A` is symmetric. Each node `i` also possesses a set of **features** `x_i`, which can be aggregated into a feature matrix `X` of size `N x F`, where `F` is the number of features per node.

<a name="22-limitations-of-traditional-neural-networks-for-graph-data"></a>
### 2.2. Limitations of Traditional Neural Networks for Graph Data
Traditional neural networks, including Multi-Layer Perceptrons (MLPs) and CNNs, are designed for data with a fixed, ordered structure.
*   **MLPs** treat inputs as independent features, ignoring structural relationships between nodes.
*   **CNNs** rely on local connectivity patterns (filters) that operate on grid-like data (e.g., pixels in an image) with a fixed number of neighbors and spatial order. Graphs lack this regularity: nodes can have varying numbers of neighbors, and there's no inherent "up," "down," "left," or "right." Furthermore, the order of nodes in the adjacency matrix is arbitrary and can change without altering the graph's fundamental structure, yet it would drastically change the input to a traditional neural network, leading to inconsistent outputs (lack of **permutation invariance**).

<a name="23-spectral-graph-convolutions-and-the-graph-laplacian"></a>
### 2.3. Spectral Graph Convolutions and the Graph Laplacian
The concept of graph convolution initially emerged from the **spectral domain** of graphs. In signal processing, convolution is often performed by transforming signals into the frequency domain using the Fourier Transform, applying an element-wise product with a filter, and then inverse transforming. Analogously, graph signals (node features) can be transformed into the graph spectral domain using the eigenvectors of the **graph Laplacian matrix** `L = D - A`, where `D` is the **degree matrix** (a diagonal matrix where `D_ii` is the degree of node `i`). The normalized Laplacian `L_norm = I - D^(-1/2) A D^(-1/2)` is often preferred.

A convolution operation `x * g` on a graph signal `x` with a filter `g` can be defined in the spectral domain as:
`x * g = U ( (U^T x) ⊙ (U^T g) )`
where `U` is the matrix of eigenvectors of `L` (forming the graph Fourier basis), `U^T x` is the graph Fourier transform of `x`, and `⊙` denotes the element-wise product. This approach, while theoretically sound, is computationally expensive due to the eigenvalue decomposition of `L` and applying the filter `g` in the spectral domain.

<a name="24-the-simplified-gcn-layer"></a>
### 2.4. The Simplified GCN Layer
To address the computational issues of spectral GCNs, Kipf and Welling introduced a simplified, first-order approximation of the spectral convolution. Their key insight was to limit the filter to a first-order neighborhood (i.e., immediate neighbors) and avoid explicit eigendecomposition.

The core propagation rule for a single GCN layer is given by:
`H^(l+1) = σ(Ã H^(l) W^(l))`

Let's break down the components:
*   `H^(l)`: The input feature matrix at layer `l`, where `H^(0) = X` (the initial node features). `H^(l)` has dimensions `N x F^(l)`, where `F^(l)` is the number of features at layer `l`.
*   `H^(l+1)`: The output feature matrix for the next layer `l+1`, with dimensions `N x F^(l+1)`.
*   `W^(l)`: A learnable weight matrix for layer `l`, with dimensions `F^(l) x F^(l+1)`. This matrix acts similarly to weights in a fully connected layer, transforming feature dimensions.
*   `σ(.)`: An element-wise non-linear activation function, such as ReLU.
*   `Ã`: The **renormalized adjacency matrix**. It is calculated as `Ã = D̃^(-1/2) Ã D̃^(-1/2)`, where:
    *   `Ã = A + I_N`: The adjacency matrix `A` with self-loops added (`I_N` is the identity matrix). Adding self-loops ensures that a node's own features are included in its neighborhood aggregation, preventing information loss from the node itself.
    *   `D̃`: The degree matrix of `Ã`. `D̃_ii = Σ_j Ã_ij`.
    *   The term `D̃^(-1/2) Ã D̃^(-1/2)` serves as a normalization factor. It ensures that the feature scales remain stable across layers and prevents issues like exploding or vanishing gradients. Specifically, `D̃^(-1/2)` scales the features of connected nodes inversely proportional to their degree, effectively averaging neighbors' features.

In essence, the GCN layer performs the following operations:
1.  **Add self-loops to the adjacency matrix**: `Ã = A + I`.
2.  **Normalize the adjacency matrix**: `Ã = D̃^(-1/2) Ã D̃^(-1/2)`. This step performs a form of averaging over the features of a node's neighbors, including itself.
3.  **Multiply by feature matrix**: `Ã H^(l)` aggregates the features from a node's neighborhood (including itself).
4.  **Apply linear transformation**: `(Ã H^(l)) W^(l)` transforms the aggregated features using learnable weights.
5.  **Apply non-linearity**: `σ(...)` introduces non-linearity, enabling the network to learn complex patterns.

By stacking multiple GCN layers, a node can aggregate information from its `k`-hop neighborhood after `k` layers, allowing it to capture increasingly global structural information.

<a name="3-applications-of-gcns"></a>
## 3. Applications of GCNs
GCNs have demonstrated state-of-the-art performance across a wide range of tasks involving graph-structured data:

*   **Node Classification:** Predicting the category or label of a node within a graph. For example, classifying users in a social network (e.g., identifying spammers) or documents in a citation network (e.g., categorizing research papers).
*   **Link Prediction:** Predicting the existence of missing or future links between nodes. This is crucial in recommendation systems (e.g., suggesting friends on social media or items to buy), knowledge graph completion, and drug-target interaction prediction.
*   **Graph Classification:** Classifying entire graphs based on their structure and node features. Applications include classifying molecules based on their properties (e.g., toxicity, drug efficacy) or identifying types of social networks.
*   **Recommendation Systems:** Leveraging user-item interaction graphs to provide personalized recommendations by understanding user preferences and item relationships.
*   **Drug Discovery and Bioinformatics:** Analyzing molecular structures, protein-protein interaction networks, and gene regulatory networks to predict drug properties, identify disease pathways, and design new compounds.
*   **Traffic Prediction:** Modeling road networks as graphs to predict traffic flow, congestion, and optimal routing.
*   **Social Network Analysis:** Understanding community structures, influence propagation, and detecting anomalies or misinformation.
*   **Natural Language Processing:** Constructing graphs from text (e.g., dependency trees, co-occurrence graphs) to enhance tasks like sentiment analysis, relation extraction, and text classification.

<a name="4-code-example"></a>
## 4. Code Example
Here's a simple Python code snippet demonstrating a single GCN layer using PyTorch. This example focuses on the forward pass of the GCN layer, illustrating the matrix multiplications and normalization steps.

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # Define a learnable weight matrix for linear transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize weights (e.g., using Xavier uniform initialization)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj_matrix):
        """
        Forward pass for a single GCN layer.

        Args:
            features (torch.Tensor): Node features matrix (N x F_in).
            adj_matrix (torch.Tensor): Adjacency matrix (N x N), usually pre-normalized.

        Returns:
            torch.Tensor: Output features matrix (N x F_out).
        """
        # Step 1: Linear transformation of features
        # (N x F_in) @ (F_in x F_out) -> (N x F_out)
        h = torch.mm(features, self.weight)

        # Step 2: Graph convolution (aggregation and normalization)
        # This assumes adj_matrix is already the renormalized adjacency matrix (Ã)
        # (N x N) @ (N x F_out) -> (N x F_out)
        output = torch.mm(adj_matrix, h)

        # Step 3: Apply non-linearity (e.g., ReLU, typically outside the layer definition for modularity)
        # For simplicity, we can include it here, but often it's applied after the layer.
        # output = torch.relu(output) # Example of where to apply activation

        return output

# Example usage:
# Assuming 4 nodes, 16 input features, 32 output features
num_nodes = 4
in_feats = 16
out_feats = 32

# Create dummy node features (N x F_in)
node_features = torch.randn(num_nodes, in_feats)

# Create a dummy renormalized adjacency matrix (Ã)
# In a real scenario, you'd calculate D̃^(-1/2) (A + I) D̃^(-1/2)
# For simplicity, let's just make a symmetric matrix that sums to 1 for each row for average-like behavior
adj_norm = torch.tensor([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.5, 0.5]
], dtype=torch.float32)

# Instantiate the GCN layer
gcn_layer = GCNLayer(in_feats, out_feats)

# Perform forward pass
output_features = gcn_layer(node_features, adj_norm)

print("Input Features Shape:", node_features.shape)
print("Output Features Shape:", output_features.shape)
# print("Output Features:\n", output_features) # Uncomment to see the actual features

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion
Graph Convolutional Networks have revolutionized the way deep learning models interact with graph-structured data. By elegantly adapting the concept of convolution to irregular topologies, GCNs enable the powerful aggregation of neighborhood information, leading to rich, context-aware node representations. Their theoretical foundations, stemming from spectral graph theory and simplified through practical approximations, provide a robust framework for learning directly on graphs. The versatility of GCNs is evident in their widespread applications, from fundamental tasks like node and graph classification to complex real-world problems in drug discovery, recommendation systems, and social network analysis. As research in graph neural networks continues to evolve, GCNs remain a foundational and influential architecture, paving the way for further innovations in understanding and leveraging the interconnectedness of data.

---
<br>

<a name="türkçe-içerik"></a>
## Graf Evrişim Ağları (GCN)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. GCN'lerin Teorik Temelleri](#2-gcnlerin-teorik-temelleri)
  - [2.1. Graf Yapıları ve Komşuluk Matrisleri](#21-graf-yapıları-ve-komşuluk-matrisleri)
  - [2.2. Geleneksel Sinir Ağlarının Graf Verileri İçin Sınırlamaları](#22-geleneksel-sinir-ağlarının-graf-verileri-için-sınırlamaları)
  - [2.3. Spektral Graf Evrişimleri ve Graf Laplasyeni](#23-spektral-graf-evrişimleri-ve-graf-laplasyeni)
  - [2.4. Basitleştirilmiş GCN Katmanı](#24-basitleştirilmiş-gcn-katmanı)
- [3. GCN'lerin Uygulama Alanları](#3-gcnlerin-uygulama-alanları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Graf Evrişim Ağları (GCN'ler), derin öğrenme alanında önemli bir ilerlemeyi temsil ederek, Evrişimsel Sinir Ağlarının (CNN'ler) güçlü kavramlarını düzenli ızgara benzeri verilerden (görseller gibi) keyfi yapılandırılmış graf verilerine genişletir. **Düğümler** (veya köşeler) ve **kenarlardan** (veya bağlantılardan) oluşan graflar, çeşitli alanlarda varlıklar arasındaki ilişkileri modelleyerek yaygın olarak bulunur. Örnekler arasında sosyal ağlar, moleküler yapılar, alıntı ağları ve bilgi grafları yer alır. Geleneksel sinir ağı mimarileri, standart CNN'ler dahil olmak üzere, düzensiz topolojileri, değişen komşuluk boyutları ve düğümler için sabit bir uzamsal sıralamanın olmaması nedeniyle graf yapılı verileri işlemek için uygun değildir.

GCN'ler, evrişim işlemlerini doğrudan graflar üzerinde tanımlayarak bu zorlukların üstesinden gelir. Temel fikir, bir düğümün yerel komşuluğundaki bilgiyi bir araya getirmek, düğüm özelliklerini komşularının özelliklerine ve grafın yapısına göre dönüştürmektir. Bu toplama süreci, GCN'lerin hem tek tek düğümlerin niteliklerini hem de daha büyük graf içindeki ilişkisel bağlamlarını yakalayan temsiller öğrenmesini sağlar. GCN katmanları ardışık olarak istiflenerek, bilgi birden fazla adıma yayılarak düğümlerin giderek daha uzak komşulardan öğrenmesine olanak tanır. Bu yetenek, **düğüm sınıflandırması**, **bağlantı tahmini** ve **graf sınıflandırması** gibi görevler için GCN'leri özellikle etkili kılar; bu görevlerde varlıklar ve bağlantıları arasındaki etkileşimi anlamak çok önemlidir. GCN'lerin yükselişi, derin öğrenme tekniklerini karmaşık, Öklid dışı verilere uygulamak için yeni yollar açarak biyoinformatikten tavsiye sistemlerine ve doğal dil işlemeye kadar birçok alanı önemli ölçüde etkilemiştir.

<a name="2-gcnlerin-teorik-temelleri"></a>
## 2. GCN'lerin Teorik Temelleri
GCN'leri anlamak, temel graf teorisi kavramlarını ve evrişimsel işlemleri graflara uyarlamanın motivasyonunu kavramayı gerektirir.

<a name="21-graf-yapıları-ve-komşuluk-matrisleri"></a>
### 2.1. Graf Yapıları ve Komşuluk Matrisleri
Bir graf `G = (V, E)`, bir **düğümler** kümesi `V` (burada `|V| = N` düğüm sayısıdır) ve bir **kenarlar** kümesi `E`'den oluşur. Kenarlar, düğümler arasındaki bağlantıları temsil eder. Graflar yönlü veya yönsüz, ağırlıklı veya ağırlıksız olabilir. Ağırlıksız bir graf için yapısı tipik olarak `N x N` boyutunda bir **komşuluk matrisi** `A` ile temsil edilir; burada `A_ij = 1` eğer `i` düğümü ile `j` düğümü arasında bir kenar varsa ve `A_ij = 0` aksi takdirde. Yönsüz graflar için `A` simetriktir. Her `i` düğümü ayrıca bir dizi **özelliğe** `x_i` sahiptir, bunlar `N x F` boyutunda bir özellik matrisi `X` olarak birleştirilebilir; burada `F` düğüm başına özellik sayısıdır.

<a name="22-geleneksel-sinir-ağlarının-graf-verileri-için-sınırlamaları"></a>
### 2.2. Geleneksel Sinir Ağlarının Graf Verileri İçin Sınırlamaları
Çok Katmanlı Algılayıcılar (MLP'ler) ve CNN'ler dahil olmak üzere geleneksel sinir ağları, sabit, sıralı bir yapıya sahip veriler için tasarlanmıştır.
*   **MLP'ler**, girdileri bağımsız özellikler olarak ele alır ve düğümler arasındaki yapısal ilişkileri göz ardı eder.
*   **CNN'ler**, sabit sayıda komşu ve uzamsal sıraya sahip ızgara benzeri veriler (örn. bir görüntüdeki pikseller) üzerinde çalışan yerel bağlantı desenlerine (filtreler) dayanır. Graflar bu düzenliliğe sahip değildir: düğümlerin değişen sayıda komşusu olabilir ve doğal bir "yukarı", "aşağı", "sol" veya "sağ" yoktur. Dahası, komşuluk matrisindeki düğümlerin sırası keyfidir ve grafın temel yapısını değiştirmeden değişebilir, ancak bu durum geleneksel bir sinir ağının girdisini büyük ölçüde değiştirerek tutarsız çıktılara yol açar (**permütasyon değişmezliği** eksikliği).

<a name="23-spektral-graf-evrişimleri-ve-graf-laplasyeni"></a>
### 2.3. Spektral Graf Evrişimleri ve Graf Laplasyeni
Graf evrişimi kavramı başlangıçta grafların **spektral alanından** ortaya çıktı. Sinyal işlemede, evrişim genellikle sinyallerin Fourier Dönüşümü kullanılarak frekans alanına dönüştürülmesi, bir filtre ile eleman bazında çarpılması ve ardından ters dönüşüm ile gerçekleştirilir. Benzer şekilde, graf sinyalleri (düğüm özellikleri), **graf Laplasyen matrisi** `L = D - A`'nın özvektörleri kullanılarak graf spektral alanına dönüştürülebilir; burada `D` **derece matrisi**dir (köşegen elemanı `D_ii` olan bir köşegen matris, `i` düğümünün derecesidir). Normalize edilmiş Laplasyen `L_norm = I - D^(-1/2) A D^(-1/2)` genellikle tercih edilir.

Bir `x` graf sinyali üzerinde `g` filtresi ile bir evrişim işlemi `x * g` spektral alanda şu şekilde tanımlanabilir:
`x * g = U ( (U^T x) ⊙ (U^T g) )`
burada `U`, `L`'nin özvektörlerinden oluşan matris (graf Fourier tabanını oluşturur), `U^T x`, `x`'in graf Fourier dönüşümüdür ve `⊙` eleman bazında çarpımı gösterir. Bu yaklaşım, teorik olarak sağlam olmasına rağmen, `L`'nin özdeğer ayrıştırması ve `g` filtresinin spektral alanda uygulanması nedeniyle hesaplama açısından pahalıdır.

<a name="24-basitleştirilmiş-gcn-katmanı"></a>
### 2.4. Basitleştirilmiş GCN Katmanı
Spektral GCN'lerin hesaplama sorunlarını gidermek için Kipf ve Welling, spektral evrişimin basitleştirilmiş, birinci dereceden bir yaklaşımını tanıttılar. Temel fikirleri, filtreyi birinci dereceden bir komşulukla (yani, doğrudan komşularla) sınırlamak ve açık bir özayrışmadan kaçınmaktı.

Tek bir GCN katmanı için temel yayılım kuralı şöyle verilir:
`H^(l+1) = σ(Ã H^(l) W^(l))`

Bileşenleri inceleyelim:
*   `H^(l)`: `l` katmanındaki giriş özellik matrisi, burada `H^(0) = X` (başlangıç düğüm özellikleri). `H^(l)`'nin boyutları `N x F^(l)`'dir, burada `F^(l)`, `l` katmanındaki özellik sayısıdır.
*   `H^(l+1)`: Bir sonraki `l+1` katmanı için çıkış özellik matrisi, `N x F^(l+1)` boyutlarında.
*   `W^(l)`: `l` katmanı için öğrenilebilir bir ağırlık matrisi, `F^(l) x F^(l+1)` boyutlarında. Bu matris, tam bağlantılı bir katmandaki ağırlıklara benzer şekilde hareket eder ve özellik boyutlarını dönüştürür.
*   `σ(.)`: ReLU gibi, eleman bazında bir doğrusal olmayan aktivasyon fonksiyonu.
*   `Ã`: **Yeniden normalize edilmiş komşuluk matrisi**. `Ã = D̃^(-1/2) Ã D̃^(-1/2)` olarak hesaplanır, burada:
    *   `Ã = A + I_N`: Kendine döngüler eklenmiş komşuluk matrisi `A` (`I_N` birim matristir). Kendine döngüler eklemek, bir düğümün kendi özelliklerinin komşuluk toplamasına dahil edilmesini sağlar ve düğümün kendisinden gelen bilgi kaybını önler.
    *   `D̃`: `Ã`'nin derece matrisi. `D̃_ii = Σ_j Ã_ij`.
    *   `D̃^(-1/2) Ã D̃^(-1/2)` terimi bir normalizasyon faktörü olarak hizmet eder. Katmanlar arasında özellik ölçeklerinin sabit kalmasını sağlar ve patlayan veya kaybolan gradyanlar gibi sorunları önler. Özellikle, `D̃^(-1/2)`, bağlı düğümlerin özelliklerini dereceleriyle ters orantılı olarak ölçeklendirir, etkili bir şekilde komşuların özelliklerini ortalamasını sağlar.

Özetle, GCN katmanı aşağıdaki işlemleri gerçekleştirir:
1.  **Komşuluk matrisine kendine döngüler ekler**: `Ã = A + I`.
2.  **Komşuluk matrisini normalize eder**: `Ã = D̃^(-1/2) Ã D̃^(-1/2)`. Bu adım, bir düğümün komşuluğundaki (kendisi dahil) özellikler üzerinde bir tür ortalama alma işlemi yapar.
3.  **Özellik matrisi ile çarpar**: `Ã H^(l)` bir düğümün komşuluğundan (kendisi dahil) özellikleri bir araya getirir.
4.  **Doğrusal dönüşüm uygular**: `(Ã H^(l)) W^(l)` birleştirilmiş özellikleri öğrenilebilir ağırlıklar kullanarak dönüştürür.
5.  **Doğrusal olmayanlık uygular**: `σ(...)` doğrusal olmayanlık ekleyerek ağın karmaşık desenleri öğrenmesini sağlar.

Birden çok GCN katmanı üst üste istiflenerek, bir düğüm `k` katmanından sonra `k`-adım komşuluğundan bilgi toplayabilir, bu da giderek daha küresel yapısal bilgileri yakalamasını sağlar.

<a name="3-applications-of-gcns"></a>
## 3. Applications of GCNs
GCNs have demonstrated state-of-the-art performance across a wide range of tasks involving graph-structured data:

*   **Node Classification:** Predicting the category or label of a node within a graph. For example, classifying users in a social network (e.g., identifying spammers) or documents in a citation network (e.g., categorizing research papers).
*   **Link Prediction:** Predicting the existence of missing or future links between nodes. This is crucial in recommendation systems (e.g., suggesting friends on social media or items to buy), knowledge graph completion, and drug-target interaction prediction.
*   **Graph Classification:** Classifying entire graphs based on their structure and node features. Applications include classifying molecules based on their properties (e.g., toxicity, drug efficacy) or identifying types of social networks.
*   **Recommendation Systems:** Leveraging user-item interaction graphs to provide personalized recommendations by understanding user preferences and item relationships.
*   **Drug Discovery and Bioinformatics:** Analyzing molecular structures, protein-protein interaction networks, and gene regulatory networks to predict drug properties, identify disease pathways, and design new compounds.
*   **Traffic Prediction:** Modeling road networks as graphs to predict traffic flow, congestion, and optimal routing.
*   **Social Network Analysis:** Understanding community structures, influence propagation, and detecting anomalies or misinformation.
*   **Natural Language Processing:** Constructing graphs from text (e.g., dependency trees, co-occurrence graphs) to enhance tasks like sentiment analysis, relation extraction, and text classification.

<a name="4-code-example"></a>
## 4. Code Example
Here's a simple Python code snippet demonstrating a single GCN layer using PyTorch. This example focuses on the forward pass of the GCN layer, illustrating the matrix multiplications and normalization steps.

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # Define a learnable weight matrix for linear transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize weights (e.g., using Xavier uniform initialization)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj_matrix):
        """
        Forward pass for a single GCN layer.

        Args:
            features (torch.Tensor): Node features matrix (N x F_in).
            adj_matrix (torch.Tensor): Adjacency matrix (N x N), usually pre-normalized.

        Returns:
            torch.Tensor: Output features matrix (N x F_out).
        """
        # Step 1: Linear transformation of features
        # (N x F_in) @ (F_in x F_out) -> (N x F_out)
        h = torch.mm(features, self.weight)

        # Step 2: Graph convolution (aggregation and normalization)
        # This assumes adj_matrix is already the renormalized adjacency matrix (Ã)
        # (N x N) @ (N x F_out) -> (N x F_out)
        output = torch.mm(adj_matrix, h)

        # Step 3: Apply non-linearity (e.g., ReLU, typically outside the layer definition for modularity)
        # For simplicity, we can include it here, but often it's applied after the layer.
        # output = torch.relu(output) # Example of where to apply activation

        return output

# Example usage:
# Assuming 4 nodes, 16 input features, 32 output features
num_nodes = 4
in_feats = 16
out_feats = 32

# Create dummy node features (N x F_in)
node_features = torch.randn(num_nodes, in_feats)

# Create a dummy renormalized adjacency matrix (Ã)
# In a real scenario, you'd calculate D̃^(-1/2) (A + I) D̃^(-1/2)
# For simplicity, let's just make a symmetric matrix that sums to 1 for each row for average-like behavior
adj_norm = torch.tensor([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.5, 0.5]
], dtype=torch.float32)

# Instantiate the GCN layer
gcn_layer = GCNLayer(in_feats, out_feats)

# Perform forward pass
output_features = gcn_layer(node_features, adj_norm)

print("Input Features Shape:", node_features.shape)
print("Output Features Shape:", output_features.shape)
# print("Output Features:\n", output_features) # Uncomment to see the actual features

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion
Graph Convolutional Networks have revolutionized the way deep learning models interact with graph-structured data. By elegantly adapting the concept of convolution to irregular topologies, GCNs enable the powerful aggregation of neighborhood information, leading to rich, context-aware node representations. Their theoretical foundations, stemming from spectral graph theory and simplified through practical approximations, provide a robust framework for learning directly on graphs. The versatility of GCNs is evident in their widespread applications, from fundamental tasks like node and graph classification to complex real-world problems in drug discovery, recommendation systems, and social network analysis. As research in graph neural networks continues to evolve, GCNs remain a foundational and influential architecture, paving the way for further innovations in understanding and leveraging the interconnectedness of data.

---
<br>

<a name="türkçe-içerik"></a>
## Graf Evrişim Ağları (GCN)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. GCN'lerin Teorik Temelleri](#2-gcnlerin-teorik-temelleri)
  - [2.1. Graf Yapıları ve Komşuluk Matrisleri](#21-graf-yapıları-ve-komşuluk-matrisleri)
  - [2.2. Geleneksel Sinir Ağlarının Graf Verileri İçin Sınırlamaları](#22-geleneksel-sinir-ağlarının-graf-verileri-için-sınırlamaları)
  - [2.3. Spektral Graf Evrişimleri ve Graf Laplasyeni](#23-spektral-graf-evrişimleri-ve-graf-laplasyeni)
  - [2.4. Basitleştirilmiş GCN Katmanı](#24-basitleştirilmiş-gcn-katmanı)
- [3. GCN'lerin Uygulama Alanları](#3-gcnlerin-uygulama-alanları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Graf Evrişim Ağları (GCN'ler), derin öğrenme alanında önemli bir ilerlemeyi temsil ederek, Evrişimsel Sinir Ağlarının (CNN'ler) güçlü kavramlarını düzenli ızgara benzeri verilerden (görseller gibi) keyfi yapılandırılmış graf verilerine genişletir. **Düğümler** (veya köşeler) ve **kenarlardan** (veya bağlantılardan) oluşan graflar, çeşitli alanlarda varlıklar arasındaki ilişkileri modelleyerek yaygın olarak bulunur. Örnekler arasında sosyal ağlar, moleküler yapılar, alıntı ağları ve bilgi grafları yer alır. Geleneksel sinir ağı mimarileri, standart CNN'ler dahil olmak üzere, düzensiz topolojileri, değişen komşuluk boyutları ve düğümler için sabit bir uzamsal sıralamanın olmaması nedeniyle graf yapılı verileri işlemek için uygun değildir.

GCN'ler, evrişim işlemlerini doğrudan graflar üzerinde tanımlayarak bu zorlukların üstesinden gelir. Temel fikir, bir düğümün yerel komşuluğundaki bilgiyi bir araya getirmek, düğüm özelliklerini komşularının özelliklerine ve grafın yapısına göre dönüştürmektir. Bu toplama süreci, GCN'lerin hem tek tek düğümlerin niteliklerini hem de daha büyük graf içindeki ilişkisel bağlamlarını yakalayan temsiller öğrenmesini sağlar. GCN katmanları ardışık olarak istiflenerek, bilgi birden fazla adıma yayılarak düğümlerin giderek daha uzak komşulardan öğrenmesine olanak tanır. Bu yetenek, **düğüm sınıflandırması**, **bağlantı tahmini** ve **graf sınıflandırması** gibi görevler için GCN'leri özellikle etkili kılar; bu görevlerde varlıklar ve bağlantıları arasındaki etkileşimi anlamak çok önemlidir. GCN'lerin yükselişi, derin öğrenme tekniklerini karmaşık, Öklid dışı verilere uygulamak için yeni yollar açarak biyoinformatikten tavsiye sistemlerine ve doğal dil işlemeye kadar birçok alanı önemli ölçüde etkilemiştir.

<a name="2-gcnlerin-teorik-temelleri"></a>
## 2. GCN'lerin Teorik Temelleri
GCN'leri anlamak, temel graf teorisi kavramlarını ve evrişimsel işlemleri graflara uyarlamanın motivasyonunu kavramayı gerektirir.

<a name="21-graf-yapıları-ve-komşuluk-matrisleri"></a>
### 2.1. Graf Yapıları ve Komşuluk Matrisleri
Bir graf `G = (V, E)`, bir **düğümler** kümesi `V` (burada `|V| = N` düğüm sayısıdır) ve bir **kenarlar** kümesi `E`'den oluşur. Kenarlar, düğümler arasındaki bağlantıları temsil eder. Graflar yönlü veya yönsüz, ağırlıklı veya ağırlıksız olabilir. Ağırlıksız bir graf için yapısı tipik olarak `N x N` boyutunda bir **komşuluk matrisi** `A` ile temsil edilir; burada `A_ij = 1` eğer `i` düğümü ile `j` düğümü arasında bir kenar varsa ve `A_ij = 0` aksi takdirde. Yönsüz graflar için `A` simetriktir. Her `i` düğümü ayrıca bir dizi **özelliğe** `x_i` sahiptir, bunlar `N x F` boyutunda bir özellik matrisi `X` olarak birleştirilebilir; burada `F` düğüm başına özellik sayısıdır.

<a name="22-geleneksel-sinir-ağlarının-graf-verileri-için-sınırlamaları"></a>
### 2.2. Geleneksel Sinir Ağlarının Graf Verileri İçin Sınırlamaları
Çok Katmanlı Algılayıcılar (MLP'ler) ve CNN'ler dahil olmak üzere geleneksel sinir ağları, sabit, sıralı bir yapıya sahip veriler için tasarlanmıştır.
*   **MLP'ler**, girdileri bağımsız özellikler olarak ele alır ve düğümler arasındaki yapısal ilişkileri göz ardı eder.
*   **CNN'ler**, sabit sayıda komşu ve uzamsal sıraya sahip ızgara benzeri veriler (örn. bir görüntüdeki pikseller) üzerinde çalışan yerel bağlantı desenlerine (filtreler) dayanır. Graflar bu düzenliliğe sahip değildir: düğümlerin değişen sayıda komşusu olabilir ve doğal bir "yukarı", "aşağı", "sol" veya "sağ" yoktur. Dahası, komşuluk matrisindeki düğümlerin sırası keyfidir ve grafın temel yapısını değiştirmeden değişebilir, ancak bu durum geleneksel bir sinir ağının girdisini büyük ölçüde değiştirerek tutarsız çıktılara yol açar (**permütasyon değişmezliği** eksikliği).

<a name="23-spektral-graf-evrişimleri-ve-graf-laplasyeni"></a>
### 2.3. Spektral Graf Evrişimleri ve Graf Laplasyeni
Graf evrişimi kavramı başlangıçta grafların **spektral alanından** ortaya çıktı. Sinyal işlemede, evrişim genellikle sinyallerin Fourier Dönüşümü kullanılarak frekans alanına dönüştürülmesi, bir filtre ile eleman bazında çarpılması ve ardından ters dönüşüm ile gerçekleştirilir. Benzer şekilde, graf sinyalleri (düğüm özellikleri), **graf Laplasyen matrisi** `L = D - A`'nın özvektörleri kullanılarak graf spektral alanına dönüştürülebilir; burada `D` **derece matrisi**dir (köşegen elemanı `D_ii` olan bir köşegen matris, `i` düğümünün derecesidir). Normalize edilmiş Laplasyen `L_norm = I - D^(-1/2) A D^(-1/2)` genellikle tercih edilir.

Bir `x` graf sinyali üzerinde `g` filtresi ile bir evrişim işlemi `x * g` spektral alanda şu şekilde tanımlanabilir:
`x * g = U ( (U^T x) ⊙ (U^T g) )`
burada `U`, `L`'nin özvektörlerinden oluşan matris (graf Fourier tabanını oluşturur), `U^T x`, `x`'in graf Fourier dönüşümüdür ve `⊙` eleman bazında çarpımı gösterir. Bu yaklaşım, teorik olarak sağlam olmasına rağmen, `L`'nin özdeğer ayrıştırması ve `g` filtresinin spektral alanda uygulanması nedeniyle hesaplama açısından pahalıdır.

<a name="24-basitleştirilmiş-gcn-katmanı"></a>
### 2.4. Basitleştirilmiş GCN Katmanı
Spektral GCN'lerin hesaplama sorunlarını gidermek için Kipf ve Welling, spektral evrişimin basitleştirilmiş, birinci dereceden bir yaklaşımını tanıttılar. Temel fikirleri, filtreyi birinci dereceden bir komşulukla (yani, doğrudan komşularla) sınırlamak ve açık bir özayrışmadan kaçınmaktı.

Tek bir GCN katmanı için temel yayılım kuralı şöyle verilir:
`H^(l+1) = σ(Ã H^(l) W^(l))`

Bileşenleri inceleyelim:
*   `H^(l)`: `l` katmanındaki giriş özellik matrisi, burada `H^(0) = X` (başlangıç düğüm özellikleri). `H^(l)`'nin boyutları `N x F^(l)`'dir, burada `F^(l)`, `l` katmanındaki özellik sayısıdır.
*   `H^(l+1)`: Bir sonraki `l+1` katmanı için çıkış özellik matrisi, `N x F^(l+1)` boyutlarında.
*   `W^(l)`: `l` katmanı için öğrenilebilir bir ağırlık matrisi, `F^(l) x F^(l+1)` boyutlarında. Bu matris, tam bağlantılı bir katmandaki ağırlıklara benzer şekilde hareket eder ve özellik boyutlarını dönüştürür.
*   `σ(.)`: ReLU gibi, eleman bazında bir doğrusal olmayan aktivasyon fonksiyonu.
*   `Ã`: **Yeniden normalize edilmiş komşuluk matrisi**. `Ã = D̃^(-1/2) Ã D̃^(-1/2)` olarak hesaplanır, burada:
    *   `Ã = A + I_N`: Kendine döngüler eklenmiş komşuluk matrisi `A` (`I_N` birim matristir). Kendine döngüler eklemek, bir düğümün kendi özelliklerinin komşuluk toplamasına dahil edilmesini sağlar ve düğümün kendisinden gelen bilgi kaybını önler.
    *   `D̃`: `Ã`'nin derece matrisi. `D̃_ii = Σ_j Ã_ij`.
    *   `D̃^(-1/2) Ã D̃^(-1/2)` terimi bir normalizasyon faktörü olarak hizmet eder. Katmanlar arasında özellik ölçeklerinin sabit kalmasını sağlar ve patlayan veya kaybolan gradyanlar gibi sorunları önler. Özellikle, `D̃^(-1/2)`, bağlı düğümlerin özelliklerini dereceleriyle ters orantılı olarak ölçeklendirir, etkili bir şekilde komşuların özelliklerini ortalamasını sağlar.

Özetle, GCN katmanı aşağıdaki işlemleri gerçekleştirir:
1.  **Komşuluk matrisine kendine döngüler ekler**: `Ã = A + I`.
2.  **Komşuluk matrisini normalize eder**: `Ã = D̃^(-1/2) Ã D̃^(-1/2)`. Bu adım, bir düğümün komşuluğundaki (kendisi dahil) özellikler üzerinde bir tür ortalama alma işlemi yapar.
3.  **Özellik matrisi ile çarpar**: `Ã H^(l)` bir düğümün komşuluğundan (kendisi dahil) özellikleri bir araya getirir.
4.  **Doğrusal dönüşüm uygular**: `(Ã H^(l)) W^(l)` birleştirilmiş özellikleri öğrenilebilir ağırlıklar kullanarak dönüştürür.
5.  **Doğrusal olmayanlık uygular**: `σ(...)` doğrusal olmayanlık ekleyerek ağın karmaşık desenleri öğrenmesini sağlar.

Birden çok GCN katmanı üst üste istiflenerek, bir düğüm `k` katmanından sonra `k`-adım komşuluğundan bilgi toplayabilir, bu da giderek daha küresel yapısal bilgileri yakalamasını sağlar.

<a name="3-gcnlerin-uygulama-alanları"></a>
## 3. GCN'lerin Uygulama Alanları
GCN'ler, graf yapılı verileri içeren çok çeşitli görevlerde son teknoloji performans göstermiştir:

*   **Düğüm Sınıflandırması:** Bir graf içindeki bir düğümün kategorisini veya etiketini tahmin etme. Örneğin, bir sosyal ağdaki kullanıcıları sınıflandırmak (örn. spam gönderenleri belirlemek) veya bir alıntı ağındaki belgeleri sınıflandırmak (örn. araştırma makalelerini kategorize etmek).
*   **Bağlantı Tahmini:** Düğümler arasındaki eksik veya gelecekteki bağlantıların varlığını tahmin etme. Bu, tavsiye sistemlerinde (örn. sosyal medyada arkadaş veya satın alınacak öğe önerme), bilgi grafı tamamlama ve ilaç-hedef etkileşimi tahmininde çok önemlidir.
*   **Graf Sınıflandırması:** Tüm grafları yapılarına ve düğüm özelliklerine göre sınıflandırma. Uygulamalar arasında molekülleri özelliklerine göre sınıflandırma (örn. toksisite, ilaç etkinliği) veya sosyal ağ türlerini tanımlama yer alır.
*   **Tavsiye Sistemleri:** Kullanıcı tercihlerini ve öğe ilişkilerini anlayarak kişiselleştirilmiş öneriler sunmak için kullanıcı-öğe etkileşim graflarını kullanma.
*   **İlaç Keşfi ve Biyoinformatik:** İlaç özelliklerini tahmin etmek, hastalık yollarını tanımlamak ve yeni bileşikler tasarlamak için moleküler yapıları, protein-protein etkileşim ağlarını ve gen düzenleyici ağları analiz etme.
*   **Trafik Tahmini:** Trafik akışını, sıkışıklığı ve en uygun rotayı tahmin etmek için yol ağlarını graf olarak modelleme.
*   **Sosyal Ağ Analizi:** Topluluk yapılarını, etki yayılımını anlama ve anormallikleri veya yanlış bilgiyi tespit etme.
*   **Doğal Dil İşleme:** Duygu analizi, ilişki çıkarımı ve metin sınıflandırması gibi görevleri geliştirmek için metinden graflar oluşturma (örn. bağımlılık ağaçları, birlikte oluşma grafları).

<a name="4-kod-örneği"></a>
## 4. Kod Örneği
İşte PyTorch kullanarak tek bir GCN katmanını gösteren basit bir Python kod parçacığı. Bu örnek, GCN katmanının ileri geçişine odaklanarak matris çarpımlarını ve normalizasyon adımlarını göstermektedir.

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # Doğrusal dönüşüm için öğrenilebilir bir ağırlık matrisi tanımla
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Ağırlıkları başlat (örn. Xavier tekdüze başlatma kullanarak)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj_matrix):
        """
        Tek bir GCN katmanı için ileri geçiş.

        Argümanlar:
            features (torch.Tensor): Düğüm özellikleri matrisi (N x F_in).
            adj_matrix (torch.Tensor): Komşuluk matrisi (N x N), genellikle önceden normalize edilmiş.

        Döndürür:
            torch.Tensor: Çıkış özellikleri matrisi (N x F_out).
        """
        # Adım 1: Özelliklerin doğrusal dönüşümü
        # (N x F_in) @ (F_in x F_out) -> (N x F_out)
        h = torch.mm(features, self.weight)

        # Adım 2: Graf evrişimi (birleştirme ve normalizasyon)
        # Bu, adj_matrix'in zaten yeniden normalize edilmiş komşuluk matrisi (Ã) olduğunu varsayar.
        # (N x N) @ (N x F_out) -> (N x F_out)
        output = torch.mm(adj_matrix, h)

        # Adım 3: Doğrusal olmayanlık uygula (örn. ReLU, genellikle katman tanımının dışında modülerlik için)
        # Basitlik için buraya dahil edilebilir, ancak genellikle katmandan sonra uygulanır.
        # output = torch.relu(output) # Aktivasyonu uygulayacak örnek yer

        return output

# Örnek kullanım:
# 4 düğüm, 16 giriş özelliği, 32 çıkış özelliği varsayalım
num_nodes = 4
in_feats = 16
out_feats = 32

# Sahte düğüm özellikleri oluştur (N x F_in)
node_features = torch.randn(num_nodes, in_feats)

# Sahte yeniden normalize edilmiş komşuluk matrisi (Ã) oluştur
# Gerçek bir senaryoda, D̃^(-1/2) (A + I) D̃^(-1/2) hesaplardınız.
# Basitlik için, her satırın toplamı 1 olan simetrik bir matris yapalım (ortalama benzeri davranış için)
adj_norm = torch.tensor([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.5, 0.5]
], dtype=torch.float32)

# GCN katmanını örnekle
gcn_layer = GCNLayer(in_feats, out_feats)

# İleri geçişi gerçekleştir
output_features = gcn_layer(node_features, adj_norm)

print("Giriş Özellikleri Şekli:", node_features.shape)
print("Çıkış Özellikleri Şekli:", output_features.shape)
# print("Çıkış Özellikleri:\n", output_features) # Gerçek özellikleri görmek için yorumu kaldırın

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç
Graf Evrişim Ağları, derin öğrenme modellerinin graf yapılı verilerle etkileşim kurma biçimini devrim niteliğinde değiştirmiştir. Evrişim kavramını düzensiz topolojilere zarif bir şekilde uyarlayarak, GCN'ler zengin, bağlama duyarlı düğüm temsillerine yol açan güçlü komşuluk bilgilerinin birleştirilmesini sağlar. Spektral graf teorisinden kaynaklanan ve pratik yaklaşımlarla basitleştirilen teorik temelleri, graflar üzerinde doğrudan öğrenmek için sağlam bir çerçeve sunar. GCN'lerin çok yönlülüğü, düğüm ve graf sınıflandırması gibi temel görevlerden ilaç keşfi, tavsiye sistemleri ve sosyal ağ analizi gibi karmaşık gerçek dünya problemlerine kadar geniş uygulama alanlarında açıkça görülmektedir. Graf sinir ağları araştırmaları gelişmeye devam ettikçe, GCN'ler verilerin birbirine bağlılığını anlama ve bundan yararlanma konusunda daha fazla yeniliğe yol açan temel ve etkili bir mimari olmaya devam etmektedir.