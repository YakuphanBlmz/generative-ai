# Quantum Machine Learning (QML) Basics

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts of Quantum Machine Learning](#2-core-concepts-of-quantum-machine-learning)
  - [2.1. Quantum Computing Fundamentals](#21-quantum-computing-fundamentals)
  - [2.2. Quantum Data Encoding](#22-quantum-data-encoding)
  - [2.3. Variational Quantum Algorithms (VQAs)](#23-variational-quantum-algorithms-vqas)
- [3. Key QML Algorithms and Applications](#3-key-qml-algorithms-and-applications)
  - [3.1. Quantum Support Vector Machines (QSVM)](#31-quantum-support-vector-machines-qsvm)
  - [3.2. Quantum Neural Networks (QNNs)](#32-quantum-neural-networks-qnns)
  - [3.3. Quantum Clustering Algorithms](#33-quantum-clustering-algorithms)
  - [3.4. Applications](#34-applications)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<br>

### 1. Introduction
**Quantum Machine Learning (QML)** stands at the exciting intersection of **quantum computing** and **machine learning**, exploring how quantum phenomena like **superposition**, **entanglement**, and **quantum interference** can enhance traditional machine learning algorithms or enable entirely new computational paradigms. The field seeks to leverage the immense computational power of quantum systems to process and analyze data in ways intractable for classical computers, especially for problems involving high-dimensional feature spaces or complex optimizations.

The primary motivation for QML arises from the limitations encountered by classical machine learning in handling exponentially growing data sizes and complex models. While classical algorithms have achieved remarkable successes, their performance can be bottlenecked by computational resources when dealing with certain types of data (e.g., quantum chemical simulations, high-dimensional sensor data) or computationally intensive tasks. QML promises a potential "quantum advantage" by representing and manipulating information in a quantum state, potentially leading to exponential speedups or improved model expressivity for specific tasks like classification, regression, and optimization. This interdisciplinary field is still in its nascent stages, largely driven by theoretical advancements and the development of **Noisy Intermediate-Scale Quantum (NISQ)** devices.

### 2. Core Concepts of Quantum Machine Learning

To understand QML, a foundational grasp of both quantum computing and how classical data can be processed on quantum hardware is essential.

#### 2.1. Quantum Computing Fundamentals
At the heart of quantum computing are **qubits**, the quantum analogue of classical bits. Unlike classical bits that can only be 0 or 1, a qubit can exist in a **superposition** of both states simultaneously, represented as $\alpha|0\rangle + \beta|1\rangle$, where $\alpha$ and $\beta$ are complex probability amplitudes such that $|\alpha|^2 + |\beta|^2 = 1$. This property allows a single qubit to store more information than a classical bit.

Furthermore, multiple qubits can become **entangled**, forming a unified quantum state where the state of one qubit instantaneously influences the state of another, regardless of spatial separation. Entanglement is a key resource for many quantum algorithms, enabling non-classical correlations. **Quantum gates** are operations applied to qubits to manipulate their superposition and entanglement, analogous to logic gates in classical computing. Common gates include Hadamard (H), Pauli-X (NOT), CNOT, and rotation gates (Rx, Ry, Rz). Finally, **measurement** collapses a qubit's superposition into a definite classical state (0 or 1) with probabilities determined by its amplitudes.

#### 2.2. Quantum Data Encoding
A crucial step in QML is **quantum data encoding**, which involves mapping classical data points onto quantum states. This process transforms classical feature vectors into quantum states that can be processed by quantum circuits. Several encoding schemes exist:
*   **Basis Encoding:** Each bit of classical data is mapped to a qubit's basis state (0 or 1). Simple but inefficient for large datasets.
*   **Amplitude Encoding:** The amplitudes of a quantum state are used to store classical data. This can encode $2^n$ classical values into $n$ qubits, offering an exponential compression of data. However, preparing such states can be challenging.
*   **Angle Encoding (or Feature Map Encoding):** Classical features are mapped to rotation angles applied to qubits. This is a common technique in variational quantum algorithms, allowing a classical feature vector $x$ to be embedded into a quantum state $| \phi(x) \rangle$ by applying a parametrized quantum circuit.

The choice of encoding significantly impacts the potential for quantum advantage and the complexity of the quantum circuit.

#### 2.3. Variational Quantum Algorithms (VQAs)
**Variational Quantum Algorithms (VQAs)** are a prominent class of algorithms for **NISQ** devices. They employ a hybrid quantum-classical approach:
1.  A **parametrized quantum circuit (PQC)**, often called an **ansatz**, processes data and generates an output quantum state. The PQC contains a set of tunable parameters $\theta$.
2.  The quantum state is measured, and its expectation values are used to compute a **cost function** (or objective function) on a classical computer.
3.  A classical optimizer (e.g., gradient descent) iteratively updates the parameters $\theta$ to minimize the cost function, effectively "training" the quantum circuit.

VQAs are central to QML because they allow quantum circuits to learn patterns from data in a similar fashion to classical neural networks. Examples include **Variational Quantum Eigensolver (VQE)** for chemistry and **Quantum Approximate Optimization Algorithm (QAOA)** for combinatorial optimization, which can be adapted for ML tasks.

### 3. Key QML Algorithms and Applications

QML research has led to the development of quantum analogues for many classical machine learning algorithms, alongside novel quantum-native approaches.

#### 3.1. Quantum Support Vector Machines (QSVM)
**Quantum Support Vector Machines (QSVMs)** are quantum extensions of classical Support Vector Machines. The core idea is to replace the classical kernel function, which implicitly maps data into a higher-dimensional feature space, with a **quantum kernel**. This quantum kernel is computed by evaluating the overlap (fidelity) between quantum states prepared by a **quantum feature map** $U_{\phi}(x)$ for different data points $x$. By potentially accessing exponentially larger Hilbert spaces, QSVMs might achieve better classification boundaries or faster training for certain datasets compared to classical SVMs.

#### 3.2. Quantum Neural Networks (QNNs)
**Quantum Neural Networks (QNNs)** are quantum versions of artificial neural networks, typically implemented as VQAs. In a QNN, the layers of a neural network are replaced by parametrized quantum circuits. Qubits act as neurons, and quantum gates with adjustable parameters serve as weighted connections. The input data is encoded into quantum states, processed by the PQC, and then measured. The results are fed into a classical optimizer to update the quantum circuit's parameters. QNNs are being explored for tasks like classification, pattern recognition, and even generative modeling (e.g., **Quantum Generative Adversarial Networks - QGANs**).

#### 3.3. Quantum Clustering Algorithms
Clustering algorithms like **K-Means** can also be adapted for quantum computers. **Quantum K-Means** aims to find clusters in quantum data or by using quantum distance metrics. The distance between data points can be computed using quantum state overlap, potentially accelerating the distance calculation phase. Other quantum approaches to clustering might involve using quantum walks or amplitude amplification for faster exploration of the data space.

#### 3.4. Applications
The potential applications of QML span various domains:
*   **Drug Discovery and Material Science:** Simulating molecular interactions and discovering new materials, leveraging quantum mechanics inherent in these systems.
*   **Financial Modeling:** Portfolio optimization, risk assessment, and fraud detection, where complex data correlations can be challenging for classical methods.
*   **Image and Pattern Recognition:** Enhanced feature extraction and classification, especially for high-dimensional image data.
*   **Optimization Problems:** Solving complex combinatorial optimization problems faster than classical algorithms.
*   **Artificial Intelligence:** Advancing core AI capabilities through more powerful learning models.

### 4. Code Example

This short Qiskit example demonstrates how to create a simple parametrized quantum circuit, which is a fundamental building block for many QML algorithms. It creates a circuit with two qubits and applies rotation gates (Ry, Rz) with placeholder parameters.

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Create a Quantum Circuit with 2 qubits
qc = QuantumCircuit(2)

# Define two parameters for rotation gates
param_theta = Parameter('θ')
param_phi = Parameter('φ')

# Apply a Hadamard gate to the first qubit to create superposition
qc.h(0)

# Apply a parametrized Ry rotation gate to the first qubit
qc.ry(param_theta, 0)

# Apply a parametrized Rz rotation gate to the second qubit
qc.rz(param_phi, 1)

# Apply a CNOT gate (entanglement) between the two qubits
qc.cx(0, 1)

# Draw the circuit (for visualization in Jupyter notebooks or print to string)
print("Parameterized Quantum Circuit:")
print(qc.draw(output='text'))

# Example of binding parameters (for later execution)
# You would typically do this iteratively with an optimizer in a VQA loop
# bound_qc = qc.assign_parameters({param_theta: 1.57, param_phi: 3.14})
# print("\nCircuit with bound parameters:")
# print(bound_qc.draw(output='text'))

(End of code example section)
```

### 5. Conclusion
**Quantum Machine Learning (QML)** represents a transformative frontier in artificial intelligence, offering a tantalizing promise of overcoming limitations inherent in classical computing paradigms. By harnessing the unique principles of quantum mechanics—**superposition**, **entanglement**, and **interference**—QML seeks to unlock unprecedented computational power for tasks ranging from complex data analysis to advanced optimization. While the field is still in its formative years, with significant challenges such as **decoherence**, **error correction**, and the limitations of current **NISQ devices**, the rapid advancements in both quantum hardware and algorithmic research are paving the way for a future where quantum computers could augment or even revolutionize machine learning. The continued interdisciplinary collaboration between quantum physicists, computer scientists, and machine learning experts will be crucial in realizing the full potential of QML, ushering in an era of truly powerful and innovative artificial intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## Kuantum Makine Öğrenimi (QML) Temelleri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Kuantum Makine Öğreniminin Temel Kavramları](#2-kuantum-makine-ogreniminin-temel-kavramlari)
  - [2.1. Kuantum Bilişim Temelleri](#21-kuantum-bilisim-temelleri)
  - [2.2. Kuantum Veri Kodlama](#22-kuantum-veri-kodlama)
  - [2.3. Varyasyonel Kuantum Algoritmaları (VQA'lar)](#23-varyasyonel-kuantum-algoritmalari-vqalar)
- [3. Önemli QML Algoritmaları ve Uygulamaları](#3-onemli-qml-algoritmalari-ve-uygulamalari)
  - [3.1. Kuantum Destek Vektör Makineleri (QSVM)](#31-kuantum-destek-vektor-makineleri-qsvm)
  - [3.2. Kuantum Yapay Sinir Ağları (QNN'ler)](#32-kuantum-yapay-sinir-aglari-qnns)
  - [3.3. Kuantum Kümeleme Algoritmaları](#33-kuantum-kumeleme-algoritmalari)
  - [3.4. Uygulamalar](#34-uygulamalar)
- [4. Kod Örneği](#4-kod-ornegi)
- [5. Sonuç](#5-sonuc)

<br>

### 1. Giriş
**Kuantum Makine Öğrenimi (QML)**, **kuantum bilişim** ve **makine öğreniminin** heyecan verici kesişim noktasında yer alır ve **süperpozisyon**, **dolanıklık** ve **kuantum girişimi** gibi kuantum olgularının geleneksel makine öğrenimi algoritmalarını nasıl geliştirebileceğini veya tamamen yeni hesaplama paradigmalarını nasıl mümkün kılabileceğini araştırır. Alan, yüksek boyutlu özellik uzayları veya karmaşık optimizasyonlar içeren sorunlar için klasik bilgisayarlar için imkansız olan yollarla verileri işlemek ve analiz etmek için kuantum sistemlerinin muazzam hesaplama gücünden yararlanmayı amaçlamaktadır.

QML'nin ana motivasyonu, klasik makine öğreniminin üstel olarak büyüyen veri boyutları ve karmaşık modellerle başa çıkmada karşılaştığı sınırlamalardan kaynaklanmaktadır. Klasik algoritmalar dikkate değer başarılar elde etmiş olsa da, belirli veri türleriyle (örn. kuantum kimyasal simülasyonlar, yüksek boyutlu sensör verileri) veya hesaplama açısından yoğun görevlerle uğraşırken performansları hesaplama kaynakları tarafından tıkanabilir. QML, bilgiyi kuantum durumunda temsil ederek ve manipüle ederek potansiyel bir "kuantum avantajı" vaat etmektedir; bu da sınıflandırma, regresyon ve optimizasyon gibi belirli görevler için üstel hızlanmalar veya iyileştirilmiş model ifade gücü sağlayabilir. Bu disiplinlerarası alan, büyük ölçüde teorik ilerlemeler ve **Gürültülü Ara Ölçekli Kuantum (NISQ)** cihazlarının geliştirilmesiyle desteklenen, hala başlangıç aşamasındadır.

### 2. Kuantum Makine Öğreniminin Temel Kavramları

QML'yi anlamak için hem kuantum bilişime hem de klasik verilerin kuantum donanımında nasıl işlenebileceğine dair temel bir kavrayış şarttır.

#### 2.1. Kuantum Bilişim Temelleri
Kuantum bilişimin kalbinde, klasik bitlerin kuantum analoğu olan **kübitler** bulunur. Yalnızca 0 veya 1 olabilen klasik bitlerin aksine, bir kübit aynı anda her iki durumun bir **süperpozisyonunda** bulunabilir ve $\alpha|0\rangle + \beta|1\rangle$ olarak temsil edilir; burada $\alpha$ ve $\beta$ karmaşık olasılık genlikleri olup $|\alpha|^2 + |\beta|^2 = 1$ koşulunu sağlarlar. Bu özellik, tek bir kübitin klasik bir bitten daha fazla bilgi depolamasına olanak tanır.

Ayrıca, birden fazla kübit **dolanık** hale gelebilir ve uzaysal ayrıma bakılmaksızın bir kübitin durumunun anında diğerinin durumunu etkilediği birleşik bir kuantum durumu oluşturabilir. Dolanıklık, birçok kuantum algoritması için temel bir kaynaktır ve klasik olmayan korelasyonları mümkün kılar. **Kuantum kapıları**, kübitlere süperpozisyonlarını ve dolanıklıklarını manipüle etmek için uygulanan işlemlerdir ve klasik bilişimdeki mantık kapılarına benzerdir. Yaygın kapılar arasında Hadamard (H), Pauli-X (NOT), CNOT ve dönüş kapıları (Rx, Ry, Rz) bulunur. Son olarak, **ölçüm**, bir kübitin süperpozisyonunu, genlikleri tarafından belirlenen olasılıklarla belirli bir klasik duruma (0 veya 1) çökertir.

#### 2.2. Kuantum Veri Kodlama
QML'deki önemli bir adım, klasik veri noktalarını kuantum durumlarına eşlemeyi içeren **kuantum veri kodlamadır**. Bu süreç, klasik özellik vektörlerini kuantum devreleri tarafından işlenebilecek kuantum durumlarına dönüştürür. Çeşitli kodlama şemaları mevcuttur:
*   **Temel Kodlama:** Klasik verinin her biti, bir kübitin temel durumuna (0 veya 1) eşlenir. Basit ancak büyük veri kümeleri için verimsizdir.
*   **Genlik Kodlama:** Bir kuantum durumunun genlikleri, klasik verileri depolamak için kullanılır. Bu, $2^n$ klasik değeri $n$ kübite kodlayabilir ve verilerin üstel bir sıkıştırmasını sunar. Ancak, bu tür durumları hazırlamak zor olabilir.
*   **Açı Kodlama (veya Özellik Haritası Kodlama):** Klasik özellikler, kübitlere uygulanan dönüş açılarına eşlenir. Bu, varyasyonel kuantum algoritmalarında yaygın bir tekniktir ve klasik bir özellik vektörü $x$'in parametreli bir kuantum devresi uygulayarak bir kuantum durumu $| \phi(x) \rangle$'a gömülmesini sağlar.

Kodlama seçimi, kuantum avantajı potansiyelini ve kuantum devresinin karmaşıklığını önemli ölçüde etkiler.

#### 2.3. Varyasyonel Kuantum Algoritmaları (VQA'lar)
**Varyasyonel Kuantum Algoritmaları (VQA'lar)**, **NISQ** cihazları için öne çıkan bir algoritma sınıfıdır. Hibrit bir kuantum-klasik yaklaşım kullanırlar:
1.  Genellikle bir **ansatz** olarak adlandırılan **parametreli kuantum devresi (PQC)**, verileri işler ve bir çıkış kuantum durumu oluşturur. PQC, bir dizi ayarlanabilir parametre $\theta$ içerir.
2.  Kuantum durumu ölçülür ve beklenen değerleri klasik bir bilgisayarda bir **maliyet fonksiyonu** (veya amaç fonksiyonu) hesaplamak için kullanılır.
3.  Klasik bir optimize edici (örn. gradyan inişi), maliyet fonksiyonunu minimize etmek için parametreler $\theta$'yı yinelemeli olarak günceller ve kuantum devresini etkin bir şekilde "eğitir".

VQA'lar, kuantum devrelerinin klasik sinir ağlarına benzer şekilde verilerden öğrenme modellerini öğrenmesine izin verdikleri için QML'nin merkezindedir. Örnekler arasında kimya için **Varyasyonel Kuantum Özdeğer Çözücü (VQE)** ve kombinatoryal optimizasyon için **Kuantum Yaklaşık Optimizasyon Algoritması (QAOA)** bulunur, bunlar ML görevlerine uyarlanabilir.

### 3. Önemli QML Algoritmaları ve Uygulamaları

QML araştırması, birçok klasik makine öğrenimi algoritmasının kuantum analoglarının yanı sıra yeni kuantum-yerel yaklaşımların geliştirilmesine yol açmıştır.

#### 3.1. Kuantum Destek Vektör Makineleri (QSVM)
**Kuantum Destek Vektör Makineleri (QSVM'ler)**, klasik Destek Vektör Makinelerinin kuantum uzantılarıdır. Temel fikir, verileri üst boyutlu bir özellik uzayına örtük olarak eşleyen klasik çekirdek fonksiyonunu, bir **kuantum çekirdekle** değiştirmektir. Bu kuantum çekirdek, farklı veri noktaları $x$ için bir **kuantum özellik haritası** $U_{\phi}(x)$ tarafından hazırlanan kuantum durumları arasındaki örtüşmeyi (sadakati) değerlendirerek hesaplanır. Potansiyel olarak üstel olarak daha büyük Hilbert uzaylarına erişerek, QSVM'ler belirli veri kümeleri için klasik SVM'lere kıyasla daha iyi sınıflandırma sınırları veya daha hızlı eğitim sağlayabilir.

#### 3.2. Kuantum Yapay Sinir Ağları (QNN'ler)
**Kuantum Yapay Sinir Ağları (QNN'ler)**, genellikle VQA'lar olarak uygulanan yapay sinir ağlarının kuantum versiyonlarıdır. Bir QNN'de, bir sinir ağının katmanları parametreli kuantum devreleri ile değiştirilir. Kübitler nöronlar gibi davranır ve ayarlanabilir parametrelere sahip kuantum kapıları ağırlıklı bağlantılar olarak görev yapar. Giriş verileri kuantum durumlarına kodlanır, PQC tarafından işlenir ve ardından ölçülür. Sonuçlar, kuantum devresinin parametrelerini güncellemek için klasik bir optimize ediciye beslenir. QNN'ler, sınıflandırma, örüntü tanıma ve hatta üretken modelleme (örn. **Kuantum Üretken Çekişmeli Ağlar - QGAN'lar**) gibi görevler için araştırılmaktadır.

#### 3.3. Kuantum Kümeleme Algoritmaları
**K-Ortalamalar** gibi kümeleme algoritmaları da kuantum bilgisayarlar için uyarlanabilir. **Kuantum K-Ortalamalar**, kuantum verilerinde veya kuantum mesafe metrikleri kullanarak kümeler bulmayı amaçlar. Veri noktaları arasındaki mesafe, kuantum durum örtüşmesi kullanılarak hesaplanabilir ve potansiyel olarak mesafe hesaplama aşamasını hızlandırabilir. Kümelemeye yönelik diğer kuantum yaklaşımları, veri uzayının daha hızlı keşfi için kuantum yürüyüşlerini veya genlik amplifikasyonunu kullanmayı içerebilir.

#### 3.4. Uygulamalar
QML'nin potansiyel uygulamaları çeşitli alanları kapsar:
*   **İlaç Keşfi ve Malzeme Bilimi:** Moleküler etkileşimleri simüle etmek ve yeni malzemeler keşfetmek, bu sistemlerdeki kuantum mekaniğinden yararlanmak.
*   **Finansal Modelleme:** Portföy optimizasyonu, risk değerlendirmesi ve dolandırıcılık tespiti, karmaşık veri korelasyonlarının klasik yöntemler için zorlayıcı olabileceği yerlerde.
*   **Görüntü ve Örüntü Tanıma:** Özellikle yüksek boyutlu görüntü verileri için gelişmiş özellik çıkarma ve sınıflandırma.
*   **Optimizasyon Problemleri:** Karmaşık kombinatoryal optimizasyon problemlerini klasik algoritmalardan daha hızlı çözme.
*   **Yapay Zeka:** Daha güçlü öğrenme modelleri aracılığıyla temel yapay zeka yeteneklerini ilerletme.

### 4. Kod Örneği

Bu kısa Qiskit örneği, birçok QML algoritması için temel bir yapı taşı olan basit bir parametreli kuantum devresinin nasıl oluşturulacağını gösterir. İki kübitli bir devre oluşturur ve yer tutucu parametrelerle dönüş kapıları (Ry, Rz) uygular.

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# 2 kübitli bir Kuantum Devresi oluşturun
qc = QuantumCircuit(2)

# Dönüş kapıları için iki parametre tanımlayın
param_theta = Parameter('θ')
param_phi = Parameter('φ')

# Süperpozisyon oluşturmak için ilk kübite bir Hadamard kapısı uygulayın
qc.h(0)

# İlk kübite parametreli bir Ry dönüş kapısı uygulayın
qc.ry(param_theta, 0)

# İkinci kübite parametreli bir Rz dönüş kapısı uygulayın
qc.rz(param_phi, 1)

# İki kübit arasında bir CNOT kapısı (dolanıklık) uygulayın
qc.cx(0, 1)

# Devreyi çizin (Jupyter not defterlerinde görselleştirme veya metin olarak yazdırmak için)
print("Parametreli Kuantum Devresi:")
print(qc.draw(output='text'))

# Parametreleri bağlama örneği (daha sonraki yürütme için)
# Bunu tipik olarak bir VQA döngüsünde bir optimize edici ile yinelemeli olarak yaparsınız
# bound_qc = qc.assign_parameters({param_theta: 1.57, param_phi: 3.14})
# print("\nBağlı parametrelerle devre:")
# print(bound_qc.draw(output='text'))

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
**Kuantum Makine Öğrenimi (QML)**, yapay zekada dönüştürücü bir sınırı temsil eder ve klasik hesaplama paradigmalarında doğal olan sınırlamaların üstesinden gelme vaadini sunar. **Süperpozisyon**, **dolanıklık** ve **girişim** gibi kuantum mekaniğinin benzersiz prensiplerini kullanarak, QML, karmaşık veri analizinden gelişmiş optimizasyona kadar değişen görevler için eşi benzeri görülmemiş bir hesaplama gücünü ortaya çıkarmayı amaçlamaktadır. Alan, hala başlangıç yıllarında olsa da, **dekoherans**, **hata düzeltme** ve mevcut **NISQ cihazlarının** sınırlamaları gibi önemli zorluklarla karşı karşıyadır. Ancak, hem kuantum donanımında hem de algoritmik araştırmalardaki hızlı ilerlemeler, kuantum bilgisayarların makine öğrenimini güçlendirebileceği ve hatta devrim yaratabileceği bir geleceğin yolunu açmaktadır. Kuantum fizikçileri, bilgisayar bilimcileri ve makine öğrenimi uzmanları arasındaki sürekli disiplinlerarası işbirliği, QML'nin tüm potansiyelini gerçekleştirmek ve gerçekten güçlü ve yenilikçi bir yapay zeka çağını başlatmak için çok önemli olacaktır.
