# Graph Attention Networks (GAT)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Mechanism of Graph Attention Networks (GAT)](#2-core-mechanism-of-graph-attention-networks-gat)
  - [2.1. Feature Transformation](#21-feature-transformation)
  - [2.2. Computing Attention Coefficients](#22-computing-attention-coefficients)
  - [2.3. Softmax Normalization](#23-softmax-normalization)
  - [2.4. Aggregation and Non-Linearity](#24-aggregation-and-non-linearity)
  - [2.5. Multi-Head Attention](#25-multi-head-attention)
- [3. Advantages and Applications](#3-advantages-and-applications)
  - [3.1. Advantages over GCNs](#31-advantages-over-gcns)
  - [3.2. Applications](#32-applications)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<br>

### 1. Introduction
Graph Neural Networks (GNNs) have emerged as powerful tools for processing data represented in graph structures, demonstrating remarkable success in various domains from social network analysis to molecular chemistry. Among the innovations within the GNN paradigm, **Graph Attention Networks (GATs)** stand out by introducing an attention mechanism into the message-passing framework. Proposed by Veličković et al. in 2017, GATs address some of the limitations of earlier GNN models, particularly Graph Convolutional Networks (GCNs), by allowing nodes to dynamically learn the importance of their neighbors when aggregating features.

Traditional GCNs typically assign fixed, pre-defined weights (e.g., based on degree normalization) to neighbors during aggregation. This approach makes GCNs inherently **transductive**, meaning they struggle to generalize to unseen nodes or graph structures without retraining. Furthermore, the fixed weighting schema prevents the model from discerning varying levels of importance among different neighbors, potentially leading to suboptimal representations, especially in graphs with heterogeneous connectivity patterns.

GATs overcome these challenges by employing a **self-attention mechanism**. Each node computes attention scores for its neighbors, indicating how much each neighbor's features should contribute to the node's new representation. This process is shareable across all nodes in the graph, making GATs inherently **inductive**. This means GATs can generalize to unseen graphs and nodes, a critical capability for real-world applications where graph structures are constantly evolving or vary significantly. The attention mechanism also offers a degree of **interpretability**, as the learned attention weights can reveal which neighbors are most influential for a given node.

This document will delve into the core mechanism of GATs, explore their advantages over previous models, discuss their practical applications, and provide a illustrative code example.

<a name="2-core-mechanism-of-graph-attention-networks-gat"></a>
### 2. Core Mechanism of Graph Attention Networks (GAT)
The central idea behind GAT is to compute the hidden representations of each node by attending over its neighbors, essentially allowing the network to specify different weights to different neighbors within a neighborhood. This process can be broken down into several key steps for a single GAT layer.

Let $N$ be the number of nodes in the graph, and for each node $i$, let its input features be $\mathbf{h}_i \in \mathbb{R}^F$, where $F$ is the number of features. The goal of a GAT layer is to produce new output features $\mathbf{h}'_i \in \mathbb{R}^{F'}$ for each node.

#### 2.1. Feature Transformation
Before computing attention coefficients, a shared linear transformation is applied to every node's features. This transformation projects the input features into a higher-level feature space, which is crucial for learning rich representations. A learnable weight matrix $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is used for this purpose:
$\mathbf{z}_i = \mathbf{W}\mathbf{h}_i$

Where $\mathbf{z}_i$ are the transformed features of node $i$.

#### 2.2. Computing Attention Coefficients
The core of GAT lies in computing **unnormalized attention coefficients** $e_{ij}$ between a node $i$ and its neighbor $j \in \mathcal{N}(i)$. This coefficient indicates the importance of node $j$'s features to node $i$. It is computed using a shared attention mechanism $a: \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$, parameterized by a single-layer feedforward neural network, followed by a LeakyReLU non-linearity.
$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \,||\, \mathbf{W}\mathbf{h}_j])$

Here, $\mathbf{a} \in \mathbb{R}^{2F'}$ is a learnable weight vector, and $||$ denotes the **concatenation** operation. The attention mechanism $a$ is applied to the concatenated transformed features of the two nodes. The LeakyReLU activation function introduces non-linearity, allowing the model to learn complex relationships. Only neighbors (and often the node itself) participate in this calculation due to the **masked attention** property.

#### 2.3. Softmax Normalization
To make the attention coefficients comparable across different neighborhoods and to sum to 1, a **softmax function** is applied over all neighbors $j$ of node $i$:
$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$

These $\alpha_{ij}$ are the **normalized attention coefficients**. They represent the final weights assigned to each neighbor $j$ when aggregating features for node $i$.

#### 2.4. Aggregation and Non-Linearity
Once the normalized attention coefficients are obtained, they are used to compute a weighted sum of the transformed neighbor features. This aggregated sum is then passed through an activation function $\sigma$ (e.g., ELU) to produce the final output features for node $i$ for this layer:
$\mathbf{h}'_i = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}\mathbf{h}_j \right)$

If the node itself is included in its neighborhood ($\mathcal{N}(i)$ includes $i$), then $\mathbf{h}_i$ also contributes to its own representation, similar to a self-loop in GCNs.

#### 2.5. Multi-Head Attention
To stabilize the learning process and enable the model to learn different aspects of relationships, GATs employ **multi-head attention**, similar to the Transformer architecture. Instead of a single attention mechanism, $K$ independent attention mechanisms are run in parallel. Each "head" computes its own set of attention coefficients and aggregates features:
$\mathbf{h}'_i^{(k)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\mathbf{h}_j \right)$

The outputs from these $K$ heads are then combined. For intermediate layers, they are typically **concatenated**:
$\mathbf{h}'_i = \mathbf{h}'_i^{(1)} \,||\, \mathbf{h}'_i^{(2)} \,||\, \dots \,||\, \mathbf{h}'_i^{(K)}$

For the final output layer, concatenation is often not desirable as it increases the feature dimensionality too much. Instead, the features from the $K$ heads are usually **averaged**, optionally followed by a final linear transformation:
$\mathbf{h}'_i = \frac{1}{K} \sum_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\mathbf{h}_j \right)$

Multi-head attention allows the model to capture diverse neighborhood dependencies and often leads to more robust and higher-quality node representations.

<a name="3-advantages-and-applications"></a>
### 3. Advantages and Applications

#### 3.1. Advantages over GCNs
Graph Attention Networks offer several significant advantages over earlier models like Graph Convolutional Networks (GCNs):

*   **Inductive Capability:** GATs are inherently **inductive**. The attention mechanism, being a local operation, calculates attention coefficients for each node independently of the global graph structure. This allows GATs to generalize to unseen nodes or even entirely new graphs without requiring a full re-training, a major improvement over the transductive nature of many GCN variants.
*   **Variable Neighbor Importance:** Unlike GCNs that typically assign pre-defined or fixed weights to neighbors (e.g., inverse of degree), GATs learn the relative importance of each neighbor to a central node. This dynamic weighting allows the model to focus on the most relevant parts of the neighborhood for a given task, leading to richer and more adaptive representations, especially in graphs with diverse connectivity.
*   **Handling Varying Node Degrees:** GATs naturally handle nodes with varying numbers of neighbors (degrees). The attention mechanism normalizes over the neighbors, ensuring that the aggregation process is not biased by the number of neighbors but rather by their learned importance.
*   **Interpretability:** The attention coefficients $\alpha_{ij}$ learned by GATs can provide insights into which neighbors are deemed most important for a node's representation. This offers a degree of **interpretability**, helping researchers understand the model's decision-making process.
*   **Computational Efficiency:** While GATs involve computing attention coefficients, the operation is parallelizable across all nodes and edges. The complexity is proportional to the number of edges, making it computationally efficient, especially for sparse graphs.

#### 3.2. Applications
GATs have been successfully applied to a wide range of tasks involving graph-structured data:

*   **Node Classification:** This is a primary application, where the goal is to predict the label of a node based on its features and its connections within the graph. Benchmarks like Cora, CiteSeer, and PubMed datasets are commonly used for this.
*   **Link Prediction:** Predicting the existence of a link between two nodes, which is crucial in social networks (friend recommendations), knowledge graphs (entity relationships), and biological networks.
*   **Graph Classification:** Classifying entire graphs based on their structure and node features, relevant in areas like drug discovery (molecular graphs) or material science.
*   **Recommendation Systems:** Leveraging user-item interaction graphs to suggest items to users or predict user preferences.
*   **Computer Vision:** Graph-based representations of images (e.g., superpixels) can use GATs for tasks like semantic segmentation.
*   **Natural Language Processing (NLP):** Processing text data structured as graphs (e.g., dependency trees, co-occurrence graphs) for tasks like sentiment analysis or text classification.
*   **Traffic Prediction:** Modeling road networks as graphs to predict traffic flow and congestion.

<a name="4-code-example"></a>
## 4. Code Example
This example demonstrates a basic implementation of a GAT layer using PyTorch Geometric (`torch_geometric`), a widely used library for GNNs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Define a simple Graph Attention Network (GAT) model
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        # First GAT layer: input features to hidden features
        # Use a concatenation for multi-head output in intermediate layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
        # Second GAT layer: hidden features to output features
        # For the final layer, usually average multi-head outputs
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # Apply first GAT layer, followed by ELU activation and dropout
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        # Apply second GAT layer
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1) # Apply log_softmax for node classification

# Example usage:
# Assume we have node features (x) and graph connectivity (edge_index)
# Let's create dummy data for demonstration
num_nodes = 100
num_features = 10
num_classes = 3
hidden_dim = 8
heads = 4 # Number of attention heads

# Dummy node features: num_nodes x num_features
x = torch.randn(num_nodes, num_features)
# Dummy edge index: 2 x num_edges (representing connections)
# Example: 0-1, 1-2, 2-0, 3-4, ...
edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)

# Initialize the GAT model
model = SimpleGAT(in_channels=num_features,
                  hidden_channels=hidden_dim,
                  out_channels=num_classes,
                  num_heads=heads)

# Perform a forward pass
output = model(x, edge_index)
print("Output shape:", output.shape) # Should be [num_nodes, num_classes]
print("First 5 node predictions (log_softmax scores):\n", output[:5])


(End of code example section)
```

<a name="5-conclusion"></a>
### 5. Conclusion
Graph Attention Networks (GATs) represent a significant advancement in the field of Graph Neural Networks by integrating the powerful concept of attention directly into the graph message-passing mechanism. By allowing each node to dynamically assign varying importance weights to its neighbors, GATs overcome key limitations of earlier GNN models, particularly their transductive nature and inability to adaptively capture diverse neighborhood information.

The inductive capabilities, improved handling of varying node degrees, and the inherent interpretability offered by attention coefficients make GATs a versatile and robust model for a wide array of graph-structured data tasks. From fundamental problems like node classification and link prediction to complex applications in drug discovery, recommendation systems, and computer vision, GATs have demonstrated superior performance and flexibility. As research in GNNs continues to evolve, the attention mechanism pioneered by GATs remains a foundational concept, influencing the design of more advanced and powerful graph learning architectures. The ability of GATs to learn localized, weighted aggregations based on feature similarities has cemented their place as a cornerstone of modern graph representation learning.

---
<br>

<a name="türkçe-içerik"></a>
## Graf Dikkat Ağları (GAT)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Graf Dikkat Ağlarının (GAT) Çekirdek Mekanizması](#2-gat-çekirdek-mekanizması)
  - [2.1. Öznitelik Dönüşümü](#21-öznitelik-dönüşümü)
  - [2.2. Dikkat Katsayılarının Hesaplanması](#22-dikkat-katsayılarının-hesaplanması)
  - [2.3. Softmax Normalizasyonu](#23-softmax-normalizasyonu)
  - [2.4. Birleştirme ve Doğrusal Olmayanlık](#24-birleştirme-ve-doğrusal-olmayanlık)
  - [2.5. Çok Kafalı Dikkat (Multi-Head Attention)](#25-çok-kafalı-dikkat-multi-head-attention)
- [3. Avantajlar ve Uygulamalar](#3-avantajlar-ve-uygulamalar)
  - [3.1. GCN'lere Göre Avantajları](#31-gcnlere-göre-avantajları)
  - [3.2. Uygulamalar](#32-uygulamalar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<br>

<a name="1-giriş"></a>
### 1. Giriş
Graf Sinir Ağları (GNN'ler), graf yapılarında temsil edilen verileri işlemek için güçlü araçlar olarak ortaya çıkmış, sosyal ağ analizinden moleküler kimyaya kadar çeşitli alanlarda dikkat çekici başarılar göstermiştir. GNN paradigması içindeki yenilikler arasında, mesaj iletimi çerçevesine bir dikkat mekanizması ekleyerek öne çıkan **Graf Dikkat Ağları (GAT'ler)** yer almaktadır. Veličković ve ark. tarafından 2017'de önerilen GAT'ler, özellikle Graf Evrişim Ağlarının (GCN'ler) bazı sınırlamalarını gidererek, düğümlerin öznitelikleri birleştirirken komşularının önemini dinamik olarak öğrenmelerine olanak tanır.

Geleneksel GCN'ler, birleştirme sırasında komşulara genellikle sabit, önceden tanımlanmış ağırlıklar (örneğin, derece normalizasyonuna dayalı) atar. Bu yaklaşım, GCN'leri doğası gereği **transdüktif** yapar, yani yeni düğümlere veya graf yapılarına, yeniden eğitim olmaksızın genelleme yapmakta zorlanırlar. Ayrıca, sabit ağırlıklandırma şeması, modelin farklı komşular arasındaki değişen önem düzeylerini ayırt etmesini engeller, bu da özellikle heterojen bağlantı desenlerine sahip graflarda suboptimal temsillerle sonuçlanabilir.

GAT'ler bu zorlukların üstesinden bir **kendi kendine dikkat (self-attention) mekanizması** kullanarak gelir. Her düğüm, komşuları için dikkat skorları hesaplayarak, her komşunun özniteliklerinin düğümün yeni temsiline ne kadar katkıda bulunması gerektiğini belirtir. Bu süreç, grafdaki tüm düğümler arasında paylaşılabilir, bu da GAT'leri doğası gereği **endüktif** kılar. Bu, GAT'lerin görülmemiş graflara ve düğümlere genelleme yapabileceği anlamına gelir ki, graf yapılarının sürekli geliştiği veya önemli ölçüde değiştiği gerçek dünya uygulamaları için kritik bir yetenektir. Dikkat mekanizması aynı zamanda bir miktar **yorumlanabilirlik** sunar, çünkü öğrenilen dikkat ağırlıkları, belirli bir düğüm için hangi komşuların en etkili olduğunu ortaya çıkarabilir.

Bu belge, GAT'lerin çekirdek mekanizmasını ayrıntılı olarak inceleyecek, önceki modellere göre avantajlarını keşfedecek, pratik uygulamalarını tartışacak ve açıklayıcı bir kod örneği sunacaktır.

<a name="2-gat-çekirdek-mekanizması"></a>
### 2. Graf Dikkat Ağlarının (GAT) Çekirdek Mekanizması
GAT'nin temel fikri, her düğümün gizli temsillerini, komşularına dikkat ederek hesaplamaktır; bu da ağın, bir mahalle içindeki farklı komşulara farklı ağırlıklar atamasını sağlar. Bu süreç, tek bir GAT katmanı için birkaç temel adıma ayrılabilir.

Grafdaki düğüm sayısı $N$ olsun ve her $i$ düğümü için girdi öznitelikleri $\mathbf{h}_i \in \mathbb{R}^F$ olsun, burada $F$ öznitelik sayısıdır. Bir GAT katmanının amacı, her düğüm için yeni çıktı öznitelikleri $\mathbf{h}'_i \in \mathbb{R}^{F'}$ üretmektir.

#### 2.1. Öznitelik Dönüşümü
Dikkat katsayılarını hesaplamadan önce, her düğümün özniteliklerine paylaşılan bir doğrusal dönüşüm uygulanır. Bu dönüşüm, zengin temsiller öğrenmek için kritik olan, girdi özniteliklerini daha yüksek seviyeli bir öznitelik uzayına yansıtır. Bu amaçla, öğrenilebilir bir ağırlık matrisi $\mathbf{W} \in \mathbb{R}^{F' \times F}$ kullanılır:
$\mathbf{z}_i = \mathbf{W}\mathbf{h}_i$

Burada $\mathbf{z}_i$, $i$ düğümünün dönüştürülmüş öznitelikleridir.

#### 2.2. Dikkat Katsayılarının Hesaplanması
GAT'nin çekirdeği, bir $i$ düğümü ile komşusu $j \in \mathcal{N}(i)$ arasındaki **normalize edilmemiş dikkat katsayıları** $e_{ij}$'yi hesaplamakta yatar. Bu katsayı, $j$ düğümünün özniteliklerinin $i$ düğümüne olan önemini gösterir. Tek katmanlı bir ileri beslemeli sinir ağıyla parametrelendirilen, ardından bir LeakyReLU doğrusal olmayanlık ile takip edilen, paylaşılan bir dikkat mekanizması $a: \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$ kullanılarak hesaplanır.
$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \,||\, \mathbf{W}\mathbf{h}_j])$

Burada, $\mathbf{a} \in \mathbb{R}^{2F'}$ öğrenilebilir bir ağırlık vektörüdür ve $||$ **birleştirme (concatenation)** işlemini belirtir. Dikkat mekanizması $a$, iki düğümün birleştirilmiş dönüştürülmüş özniteliklerine uygulanır. LeakyReLU aktivasyon fonksiyonu, modelin karmaşık ilişkileri öğrenmesini sağlayan doğrusal olmayanlık sağlar. **Maskelenmiş dikkat (masked attention)** özelliği nedeniyle bu hesaplamaya yalnızca komşular (ve genellikle düğümün kendisi) katılır.

#### 2.3. Softmax Normalizasyonu
Dikkat katsayılarını farklı mahalleler arasında karşılaştırılabilir hale getirmek ve toplamlarının 1 olmasını sağlamak için, $i$ düğümünün tüm $j$ komşuları üzerinde bir **softmax fonksiyonu** uygulanır:
$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$

Bu $\alpha_{ij}$ değerleri, **normalize edilmiş dikkat katsayılarıdır**. Bunlar, $i$ düğümü için öznitelikleri birleştirirken her komşu $j$'ye atanan nihai ağırlıkları temsil eder.

#### 2.4. Birleştirme ve Doğrusal Olmayanlık
Normalize edilmiş dikkat katsayıları elde edildikten sonra, dönüştürülmüş komşu özniteliklerinin ağırlıklı toplamını hesaplamak için kullanılırlar. Bu birleştirilmiş toplam daha sonra bir aktivasyon fonksiyonu $\sigma$ (örneğin, ELU) aracılığıyla geçirilerek bu katman için $i$ düğümünün nihai çıktı özniteliklerini üretir:
$\mathbf{h}'_i = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}\mathbf{h}_j \right)$

Eğer düğümün kendisi de komşuluğuna dahil edilirse ($\mathcal{N}(i)$ içinde $i$ de varsa), o zaman $\mathbf{h}_i$ de kendi temsiline katkıda bulunur, bu GCN'lerdeki self-loop'a benzerdir.

#### 2.5. Çok Kafalı Dikkat (Multi-Head Attention)
Öğrenme sürecini stabilize etmek ve modelin ilişkilerin farklı yönlerini öğrenmesini sağlamak için GAT'ler, Transformer mimarisine benzer şekilde **çok kafalı dikkat (multi-head attention)** kullanır. Tek bir dikkat mekanizması yerine, $K$ bağımsız dikkat mekanizması paralel olarak çalıştırılır. Her "kafa" kendi dikkat katsayılarını hesaplar ve öznitelikleri birleştirir:
$\mathbf{h}'_i^{(k)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\mathbf{h}_j \right)$

Bu $K$ kafadan gelen çıktılar daha sonra birleştirilir. Ara katmanlar için genellikle **birleştirme (concatenation)** yapılır:
$\mathbf{h}'_i = \mathbf{h}'_i^{(1)} \,||\, \mathbf{h}'_i^{(2)} \,||\, \dots \,||\, \mathbf{h}'_i^{(K)}$

Nihai çıktı katmanı için birleştirme genellikle istenmez, çünkü öznitelik boyutunu çok fazla artırır. Bunun yerine, $K$ kafadan gelen öznitelikler genellikle **ortalaması alınır**, isteğe bağlı olarak nihai bir doğrusal dönüşümle takip edilir:
$\mathbf{h}'_i = \frac{1}{K} \sum_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\mathbf{h}_j \right)$

Çok kafalı dikkat, modelin çeşitli komşuluk bağımlılıklarını yakalamasına olanak tanır ve genellikle daha sağlam ve daha yüksek kaliteli düğüm temsilleri sağlar.

<a name="3-avantajlar-ve-uygulamalar"></a>
### 3. Avantajlar ve Uygulamalar

#### 3.1. GCN'lere Göre Avantajları
Graf Dikkat Ağları, Graf Evrişim Ağları (GCN'ler) gibi önceki modellere göre önemli avantajlar sunar:

*   **Endüktif Yetenek:** GAT'ler doğası gereği **endüktiftir**. Dikkat mekanizması, yerel bir işlem olduğundan, her düğüm için dikkat katsayılarını küresel graf yapısından bağımsız olarak hesaplar. Bu, GAT'lerin görülmemiş düğümlere veya hatta tamamen yeni graflara tam bir yeniden eğitim gerektirmeden genelleme yapmasına olanak tanır, bu da birçok GCN varyantının transdüktif doğasına göre önemli bir gelişmedir.
*   **Değişken Komşu Önemi:** GCN'lerin genellikle komşulara önceden tanımlanmış veya sabit ağırlıklar (örneğin, derecenin tersi) atamasının aksine, GAT'ler her bir komşunun merkezi bir düğüme göreceli önemini öğrenir. Bu dinamik ağırlıklandırma, modelin belirli bir görev için mahallenin en ilgili kısımlarına odaklanmasını sağlar, bu da özellikle çeşitli bağlantıya sahip graflarda daha zengin ve daha uyarlanabilir temsillerle sonuçlanır.
*   **Değişen Düğüm Derecelerini Yönetme:** GAT'ler, değişen sayıda komşuya (dereceye) sahip düğümleri doğal olarak yönetir. Dikkat mekanizması, komşular üzerinde normalizasyon yapar, böylece birleştirme sürecinin komşu sayısından değil, öğrenilen önemlerinden etkilenmesini sağlar.
*   **Yorumlanabilirlik:** GAT'ler tarafından öğrenilen dikkat katsayıları $\alpha_{ij}$, hangi komşuların bir düğümün temsili için en önemli kabul edildiği konusunda bilgi sağlayabilir. Bu, bir dereceye kadar **yorumlanabilirlik** sunarak araştırmacıların modelin karar verme sürecini anlamalarına yardımcı olur.
*   **Hesaplama Verimliliği:** GAT'ler dikkat katsayılarının hesaplanmasını içerse de, işlem tüm düğümler ve kenarlar arasında paralelleştirilebilir. Karmaşıklık, kenar sayısıyla orantılıdır, bu da onu özellikle seyrek graflar için hesaplama açısından verimli kılar.

#### 3.2. Uygulamalar
GAT'ler, graf yapılı verileri içeren geniş bir görev yelpazesine başarıyla uygulanmıştır:

*   **Düğüm Sınıflandırması:** Bu, birincil bir uygulamadır ve amaç, bir düğümün özniteliklerine ve graf içindeki bağlantılarına dayanarak etiketini tahmin etmektir. Cora, CiteSeer ve PubMed veri kümeleri bu amaçla yaygın olarak kullanılmaktadır.
*   **Kenar Tahmini:** İki düğüm arasındaki bir bağlantının varlığını tahmin etmek, sosyal ağlarda (arkadaş tavsiyeleri), bilgi graflarında (varlık ilişkileri) ve biyolojik ağlarda kritik öneme sahiptir.
*   **Graf Sınıflandırması:** Yapılarına ve düğüm özniteliklerine dayanarak tüm grafları sınıflandırmak, ilaç keşfi (moleküler graflar) veya malzeme bilimi gibi alanlarda önemlidir.
*   **Tavsiye Sistemleri:** Kullanıcı-öğe etkileşim graflarını kullanarak kullanıcılara öğe önermek veya kullanıcı tercihlerini tahmin etmek.
*   **Bilgisayar Görüsü:** Görüntülerin graf tabanlı temsilleri (örneğin, superpixels) semantik segmentasyon gibi görevler için GAT'leri kullanabilir.
*   **Doğal Dil İşleme (NLP):** Duygu analizi veya metin sınıflandırması gibi görevler için graf olarak yapılandırılmış metin verilerini (örneğin, bağımlılık ağaçları, birlikte oluşma grafları) işlemek.
*   **Trafik Tahmini:** Karayolu ağlarını graf olarak modelleyerek trafik akışını ve tıkanıklığını tahmin etmek.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği
Bu örnek, GNN'ler için yaygın olarak kullanılan bir kütüphane olan PyTorch Geometric (`torch_geometric`) kullanarak temel bir GAT katmanının uygulamasını göstermektedir.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Basit bir Graf Dikkat Ağı (GAT) modeli tanımlama
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        # İlk GAT katmanı: girdi özniteliklerinden gizli özniteliklere
        # Ara katmanlarda çok kafalı çıktı için birleştirme (concatenation) kullanın
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
        # İkinci GAT katmanı: gizli özniteliklerden çıktı özniteliklerine
        # Son katman için genellikle çok kafalı çıktıların ortalaması alınır
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # İlk GAT katmanını uygulayın, ardından ELU aktivasyonu ve dropout
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        # İkinci GAT katmanını uygulayın
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1) # Düğüm sınıflandırması için log_softmax uygulayın

# Örnek kullanım:
# Düğüm özniteliklerimiz (x) ve graf bağlantımız (edge_index) olduğunu varsayalım
# Gösterim için sahte veri oluşturalım
num_nodes = 100 # Düğüm sayısı
num_features = 10 # Her düğümün öznitelik sayısı
num_classes = 3 # Sınıf sayısı
hidden_dim = 8 # Gizli katman boyutu
heads = 4 # Dikkat kafası sayısı

# Sahte düğüm öznitelikleri: num_nodes x num_features
x = torch.randn(num_nodes, num_features)
# Sahte kenar indeksi: 2 x num_edges (bağlantıları temsil eder)
# Örnek: 0-1, 1-2, 2-0, 3-4, ...
edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)

# GAT modelini başlatın
model = SimpleGAT(in_channels=num_features,
                  hidden_channels=hidden_dim,
                  out_channels=num_classes,
                  num_heads=heads)

# Bir ileri besleme işlemi gerçekleştirin
output = model(x, edge_index)
print("Çıktı şekli:", output.shape) # [num_nodes, num_classes] olmalı
print("İlk 5 düğüm tahmini (log_softmax skorları):\n", output[:5])

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
### 5. Sonuç
Graf Dikkat Ağları (GAT'ler), dikkat kavramının graf mesaj iletim mekanizmasına doğrudan entegrasyonuyla Graf Sinir Ağları alanında önemli bir ilerlemeyi temsil etmektedir. Her düğümün komşularına değişen önem ağırlıklarını dinamik olarak atamasına izin vererek, GAT'ler önceki GNN modellerinin temel sınırlamalarının, özellikle transdüktif doğalarının ve çeşitli komşuluk bilgilerini adaptif olarak yakalayamamalarının üstesinden gelir.

Endüktif yetenekleri, değişen düğüm derecelerini daha iyi ele alması ve dikkat katsayılarının sunduğu doğal yorumlanabilirlik, GAT'leri çok çeşitli graf yapılı veri görevleri için çok yönlü ve sağlam bir model haline getirir. Düğüm sınıflandırması ve kenar tahmini gibi temel problemlerden ilaç keşfi, tavsiye sistemleri ve bilgisayar görüşü gibi karmaşık uygulamalara kadar GAT'ler üstün performans ve esneklik göstermiştir. GNN'lerdeki araştırmalar gelişmeye devam ederken, GAT'ler tarafından öncülük edilen dikkat mekanizması, daha gelişmiş ve güçlü graf öğrenme mimarilerinin tasarımını etkileyen temel bir kavram olmaya devam etmektedir. GAT'lerin öznitelik benzerliklerine dayalı yerelleştirilmiş, ağırlıklı birleştirmeleri öğrenme yeteneği, modern graf temsil öğreniminin temel taşlarından biri olarak yerini sağlamlaştırmıştır.




