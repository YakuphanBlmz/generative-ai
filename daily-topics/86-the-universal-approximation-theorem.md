# The Universal Approximation Theorem

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Theoretical Foundations](#2-theoretical-foundations)
    - [2.1. Statement of the Theorem](#21-statement-of-the-theorem)
    - [2.2. Key Conditions and Extensions](#22-key-conditions-and-extensions)
- [3. Implications for Neural Networks](#3-implications-for-neural-networks)
    - [3.1. Justification of Expressive Power](#31-justification-of-expressive-power)
    - [3.2. Approximation vs. Learning](#32-approximation-vs-learning)
    - [3.3. Relevance in Deep Learning](#33-relevance-in-deep-learning)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction
The **Universal Approximation Theorem (UAT)** stands as a cornerstone in the theoretical understanding of neural networks, providing a fundamental justification for their widespread use in machine learning. First articulated by George Cybenko in 1989 for sigmoid activation functions and later extended by Kurt Hornik, Maxwell Stinchcombe, and Hal White in 1991 to a broader class of activation functions, the theorem asserts that a **feedforward neural network** with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Euclidean space, given appropriate non-constant, bounded, and monotonically increasing activation functions.

This theorem is crucial because it theoretically proves the **expressive power** of neural networks, indicating that they are capable of representing a vast array of complex relationships and patterns found in data. It fundamentally suggests that, given enough computational resources (neurons) and proper weights and biases, a relatively simple neural network architecture can model virtually any function, regardless of its complexity. However, it is important to note that the UAT is an existence theorem; it guarantees that such a network *exists* but does not provide a constructive method for finding the required weights and biases, nor does it specify the optimal network architecture (e.g., number of hidden neurons or layers). Its significance lies in establishing the theoretical upper bound of what neural networks can achieve, laying the groundwork for subsequent research and applications in areas ranging from image recognition to natural language processing within the realm of **Generative AI**.

### 2. Theoretical Foundations
#### 2.1. Statement of the Theorem
The most common formulation of the Universal Approximation Theorem, often attributed to Cybenko (1989), states that for any continuous function $f: \mathbb{R}^n \to \mathbb{R}^m$ on a compact set $K \subset \mathbb{R}^n$, and any $\epsilon > 0$, there exists a **feedforward neural network** with one hidden layer that can approximate $f$ to within $\epsilon$ accuracy. More formally, there exist weights $w_{ij}$, $v_{ji}$, biases $b_j$, and a fixed non-constant, bounded, and monotonically increasing **activation function** $\sigma$ (e.g., the sigmoid function) such that for all $x \in K$:

$$ \left| f(x) - \sum_{j=1}^{N} v_{j} \sigma \left( \sum_{i=1}^{n} w_{ij} x_i + b_j \right) \right| < \epsilon $$

Here, $N$ represents the number of neurons in the hidden layer, $n$ is the dimension of the input space, and the output layer typically uses a linear activation function for regression tasks. This statement highlights that the network's ability to approximate does not depend on a multitude of hidden layers, but rather on the sufficient width of a single hidden layer.

#### 2.2. Key Conditions and Extensions
The theorem's validity relies on specific conditions and has been extended significantly since its initial proofs:

*   **Activation Function:** The original proofs largely focused on sigmoid-like activation functions. Hornik, Stinchcombe, and White (1989, 1991) later generalized the UAT, demonstrating that any **non-polynomial continuous activation function** (e.g., ReLU, tanh, softplus) can serve this purpose. The non-polynomial requirement is crucial because polynomial activations would only allow approximation of polynomials, not arbitrary continuous functions.
*   **Compact Domain:** The approximation guarantee is typically for functions defined on a **compact subset** of $\mathbb{R}^n$. A compact set is one that is closed and bounded. This condition ensures that the function's behavior does not become arbitrarily wild at infinity, which would make approximation intractable.
*   **Number of Hidden Neurons:** The theorem specifies that a *finite* number of hidden neurons are sufficient, but it does not provide an upper bound or a method to determine this number. In practice, the required number of neurons can be very large for highly complex functions or high-dimensional input spaces.
*   **Depth vs. Width:** While the initial UAT focused on a single hidden layer (width), subsequent research has explored the implications for **deep neural networks** (multiple hidden layers). It has been shown that deeper networks can often approximate certain functions more efficiently (i.e., with exponentially fewer neurons) than shallow networks, though shallow networks retain their universal approximation capability. This efficiency argument forms a key part of the justification for **deep learning**.
*   **Approximation of Derivatives:** Extensions of the UAT have also shown that neural networks can approximate not only the function itself but also its derivatives, which is important for applications like solving differential equations.

The UAT provides the "what" – that neural networks *can* approximate – but not the "how" or "how well" in practical terms. It underscores the theoretical capacity while leaving the practical challenges of learning and generalization to be addressed by optimization algorithms and architectural design choices.

### 3. Implications for Neural Networks
#### 3.1. Justification of Expressive Power
The most profound implication of the Universal Approximation Theorem is its role in justifying the **expressive power** of feedforward neural networks. Before the UAT, the ability of neural networks to model complex, non-linear relationships was largely empirical. The theorem provided a rigorous theoretical foundation, demonstrating that neural networks are not merely sophisticated curve-fitting tools but are **function approximators** of immense theoretical capability. This theoretical backing encouraged further research and development in the field, moving neural networks from a niche academic interest to a powerful tool in various domains. It essentially states that if a function exists, a neural network can, in principle, represent it.

#### 3.2. Approximation vs. Learning
It is critical to distinguish between **approximation** and **learning**. The UAT guarantees that a network *exists* that can approximate any given function. However, it does not provide an algorithm or method to *find* the specific weights and biases that constitute this ideal network. This is where the challenge of **learning** comes in. Training a neural network involves an optimization process (e.g., gradient descent) to adjust weights and biases based on training data. The UAT does not guarantee that these optimization algorithms will converge to the optimal set of weights, nor does it assure that the learned function will generalize well to unseen data (avoid **overfitting**). Therefore, while neural networks are universal approximators, successfully *learning* a good approximation from limited data is a separate, complex problem influenced by factors such as:
*   **Optimization algorithm choice:** Stochastic Gradient Descent (SGD), Adam, etc.
*   **Network architecture:** Number of layers, neurons per layer.
*   **Regularization techniques:** Dropout, L1/L2 regularization.
*   **Amount and quality of training data.**

#### 3.3. Relevance in Deep Learning
Initially, the UAT focused on **shallow networks** (a single hidden layer). However, the rise of **deep learning**, utilizing networks with many hidden layers, has led to further insights. While shallow networks are theoretically capable of universal approximation, deep networks have demonstrated significant practical advantages, including:
*   **Parameter Efficiency:** Deep networks can often approximate complex functions with exponentially fewer neurons than shallow networks. This is due to their ability to learn hierarchical representations, where early layers extract simple features which are then combined by later layers into more abstract and complex features.
*   **Generalization:** In many real-world scenarios, deeper architectures tend to generalize better and are less prone to overfitting, especially with large datasets.
*   **Feature Learning:** Deep networks excel at **feature learning**, automatically discovering relevant features from raw data, reducing the need for manual feature engineering.

Thus, the UAT serves as a foundational theoretical justification for the *potential* of neural networks, regardless of depth. The shift towards deep architectures is driven by efficiency and practical performance, leveraging the UAT's core message that "a neural network can do it," while adding the empirical and theoretical understanding that "a *deep* neural network can often do it *better*." This principle is fundamental to the advancements seen in **Generative AI**, where complex models like GANs and Transformers approximate highly intricate data distributions.

### 4. Code Example
This simple Python code snippet illustrates the conceptual forward pass through a basic feedforward neural network with a sigmoid activation function, a core component implied by the Universal Approximation Theorem. It shows how inputs are processed through weighted sums and non-linear activations to produce an output.

```python
import numpy as np

# Define a common activation function (e.g., Sigmoid)
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

# Define a simple two-layer neural network (conceptual forward pass)
def simple_neural_network_forward(inputs, weights_hidden, biases_hidden, weights_output, biases_output):
    """
    Performs a conceptual forward pass through a simple two-layer neural network.
    This demonstrates the basic operations (weighted sum + activation).
    """
    # Hidden layer calculation:
    # 1. Weighted sum of inputs plus biases for the hidden layer.
    hidden_layer_input = np.dot(inputs, weights_hidden) + biases_hidden
    # 2. Apply the activation function to the hidden layer's input.
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Output layer calculation:
    # 1. Weighted sum of hidden layer outputs plus biases for the output layer.
    output_layer_input = np.dot(hidden_layer_output, weights_output) + biases_output
    # 2. For a simple regression task, the output layer might not have an activation.
    output = output_layer_input

    return output

# --- Example Usage ---
# Assume a single input feature, 2 hidden neurons, and a single output.
# These weights and biases are arbitrary and would be 'learned' in a real scenario.
input_val = np.array([0.5]) # A single input value (e.g., normalized data point)

# Weights and biases for the hidden layer (1 input -> 2 hidden neurons)
weights_h = np.array([[0.1, 0.2]]) # Weight matrix for hidden layer
biases_h = np.array([0.3, 0.4])    # Bias vector for hidden layer

# Weights and biases for the output layer (2 hidden neurons -> 1 output neuron)
weights_o = np.array([[0.5], [0.6]]) # Weight matrix for output layer
biases_o = np.array([0.7])         # Bias vector for output layer

# Perform the forward pass to get a predicted output
predicted_output = simple_neural_network_forward(input_val, weights_h, biases_h, weights_o, biases_o)

print(f"Input value: {input_val[0]}")
print(f"Predicted Output: {predicted_output[0]:.4f}")

(End of code example section)
```

### 5. Conclusion
The Universal Approximation Theorem is a foundational mathematical statement that underpins the theoretical capabilities of artificial neural networks. It rigorously demonstrates that even a relatively simple **feedforward neural network** with a single hidden layer and appropriate non-linear activation functions possesses the inherent capacity to approximate any continuous function to an arbitrary degree of accuracy on a compact domain. This powerful theoretical guarantee provided the essential justification for exploring and investing in neural network research, moving the field beyond heuristic arguments.

While the UAT confirms the *existence* of such an approximating network, it deliberately sidesteps the practical challenges of **learning** and **generalization**. It does not prescribe how to find the optimal weights and biases, nor does it guarantee efficient learning or effective performance on unseen data. These are challenges addressed by advancements in optimization algorithms, regularization techniques, and architectural innovations, particularly in the domain of **deep learning**. The shift towards deeper networks, while still benefiting from the UAT's core message, highlights that while shallow networks are theoretically capable, deeper architectures often offer practical advantages in terms of efficiency, feature learning, and generalization for complex tasks common in **Generative AI**.

In essence, the Universal Approximation Theorem provides the "why" – why neural networks are worth exploring for complex function approximation – while the ongoing research in deep learning, optimization, and architectural design continues to address the "how" and "how well" in practical, real-world applications. Its legacy endures as a critical theoretical bedrock for the entire field of machine learning and, specifically, for the advanced capabilities witnessed in modern AI.

---
<br>

<a name="türkçe-içerik"></a>
## Evrensel Yaklaşım Teoremi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Teorik Temeller](#2-teorik-temeller)
    - [2.1. Teoremin İfadesi](#21-teoremin-ifadesi)
    - [2.2. Anahtar Koşullar ve Uzantılar](#22-anahtar-koşullar-ve-uzantılar)
- [3. Yapay Sinir Ağları İçin Çıkarımlar](#3-yapay-sinir-ağları-için-çıkarımlar)
    - [3.1. İfade Gücünün Gerekçelendirilmesi](#31-ifade-gücünün-gerekçelendirilmesi)
    - [3.2. Yaklaşım ve Öğrenme Arasındaki Fark](#32-yaklaşım-ve-öğrenme-arasındaki-fark)
    - [3.3. Derin Öğrenmedeki Önemi](#33-derin-öğrenmedeki-önemi)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş
**Evrensel Yaklaşım Teoremi (EYT)**, yapay sinir ağlarının teorik anlayışında bir köşe taşı olarak durmakta ve makine öğreniminde yaygın kullanımlarının temel bir gerekçesini sunmaktadır. İlk olarak 1989'da George Cybenko tarafından sigmoid aktivasyon fonksiyonları için ifade edilmiş, daha sonra 1991'de Kurt Hornik, Maxwell Stinchcombe ve Hal White tarafından daha geniş bir aktivasyon fonksiyonları sınıfına genişletilmiştir. Teorem, sonlu sayıda nöron içeren tek bir gizli katmana sahip bir **ileri beslemeli yapay sinir ağının**, uygun, sabit olmayan, sınırlı ve monotonik olarak artan aktivasyon fonksiyonları verildiğinde, Öklid uzayının kompakt alt kümeleri üzerindeki herhangi bir sürekli fonksiyonu yaklaşık olarak temsil edebileceğini ileri sürer.

Bu teorem çok önemlidir çünkü yapay sinir ağlarının **ifade gücünü** teorik olarak kanıtlar ve verilerde bulunan çok çeşitli karmaşık ilişkileri ve kalıpları temsil etme yeteneğine sahip olduklarını gösterir. Temel olarak, yeterli hesaplama kaynağına (nöron) ve uygun ağırlıklar ve önyargılara sahipse, nispeten basit bir yapay sinir ağı mimarisinin karmaşıklığı ne olursa olsun neredeyse her fonksiyonu modelleyebileceğini öne sürer. Ancak, EYT'nin bir varoluş teoremi olduğunu belirtmek önemlidir; böyle bir ağın *var olduğunu* garanti eder, ancak gerekli ağırlıkları ve önyargıları bulmak için yapıcı bir yöntem sağlamaz, ayrıca en uygun ağ mimarisini (örn. gizli nöron veya katman sayısı) belirtmez. Önemi, yapay sinir ağlarının neler başarabileceğinin teorik üst sınırını belirlemesinde yatar ve **Üretken Yapay Zeka** alanında görüntü tanımadan doğal dil işlemeye kadar çeşitli alanlardaki sonraki araştırmalar ve uygulamalar için temel oluşturur.

### 2. Teorik Temeller
#### 2.1. Teoremin İfadesi
Evrensel Yaklaşım Teoreminin en yaygın formülasyonu, genellikle Cybenko'ya (1989) atfedilen şekliyle, $\mathbb{R}^n$'nin kompakt bir kümesi olan $K \subset \mathbb{R}^n$ üzerinde tanımlanmış herhangi bir $f: \mathbb{R}^n \to \mathbb{R}^m$ sürekli fonksiyonu ve herhangi bir $\epsilon > 0$ için, $f$'yi $\epsilon$ doğrulukla yaklaşık olarak temsil edebilen tek bir gizli katmana sahip bir **ileri beslemeli yapay sinir ağı** olduğunu belirtir. Daha resmi olarak, $w_{ij}$, $v_{ji}$ ağırlıkları, $b_j$ önyargıları ve sabit, sabit olmayan, sınırlı ve monotonik olarak artan bir **aktivasyon fonksiyonu** $\sigma$ (örn. sigmoid fonksiyonu) mevcuttur, öyle ki tüm $x \in K$ için:

$$ \left| f(x) - \sum_{j=1}^{N} v_{j} \sigma \left( \sum_{i=1}^{n} w_{ij} x_i + b_j \right) \right| < \epsilon $$

Burada, $N$ gizli katmandaki nöron sayısını, $n$ giriş uzayının boyutunu temsil eder ve çıkış katmanı genellikle regresyon görevleri için doğrusal bir aktivasyon fonksiyonu kullanır. Bu ifade, ağın yaklaşım yeteneğinin çok sayıda gizli katmana değil, tek bir gizli katmanın yeterli genişliğine bağlı olduğunu vurgular.

#### 2.2. Anahtar Koşullar ve Uzantılar
Teoremin geçerliliği belirli koşullara dayanır ve ilk ispatlarından bu yana önemli ölçüde genişletilmiştir:

*   **Aktivasyon Fonksiyonu:** Orijinal ispatlar büyük ölçüde sigmoid benzeri aktivasyon fonksiyonlarına odaklanmıştır. Hornik, Stinchcombe ve White (1989, 1991) daha sonra EYT'yi genelleştirerek, herhangi bir **polinom olmayan sürekli aktivasyon fonksiyonunun** (örn. ReLU, tanh, softplus) bu amaca hizmet edebileceğini göstermiştir. Polinom olmayan gereksinim, polinom aktivasyonlarının yalnızca polinomları yaklaşık olarak temsil etmesine izin vereceği, keyfi sürekli fonksiyonları değil, bu nedenle kritik öneme sahiptir.
*   **Kompakt Alan:** Yaklaşım garantisi tipik olarak $\mathbb{R}^n$'nin **kompakt bir alt kümesi** üzerinde tanımlanmış fonksiyonlar içindir. Kompakt bir küme, kapalı ve sınırlı olan bir kümedir. Bu koşul, fonksiyonun sonsuzda keyfi olarak vahşi davranışlar sergilememesini sağlar, bu da yaklaşımı zorlaştırırdı.
*   **Gizli Nöron Sayısı:** Teorem, *sonlu* sayıda gizli nöronun yeterli olduğunu belirtir, ancak bu sayıyı belirlemek için bir üst sınır veya yöntem sağlamaz. Pratikte, çok karmaşık fonksiyonlar veya yüksek boyutlu giriş uzayları için gereken nöron sayısı çok büyük olabilir.
*   **Derinlik ve Genişlik:** İlk EYT tek bir gizli katmana (genişlik) odaklanırken, sonraki araştırmalar **derin sinir ağları** (birden çok gizli katman) için çıkarımları incelemiştir. Daha derin ağların, sığ ağlara göre genellikle belirli fonksiyonları daha verimli (yani, katlanarak daha az nöronla) yaklaşık olarak temsil edebildiği gösterilmiştir, ancak sığ ağlar evrensel yaklaşım yeteneklerini korur. Bu verimlilik argümanı, **derin öğrenmenin** gerekçesinin önemli bir parçasını oluşturur.
*   **Türevlerin Yaklaşımı:** EYT'nin uzantıları, yapay sinir ağlarının sadece fonksiyonu değil, aynı zamanda türevlerini de yaklaşık olarak temsil edebileceğini göstermiştir; bu, diferansiyel denklemleri çözme gibi uygulamalar için önemlidir.

EYT, yapay sinir ağlarının *yaklaşık olarak temsil edebildiğini* - "ne" olduğunu - sağlar, ancak pratik terimlerle "nasıl" veya "ne kadar iyi" olduğunu sağlamaz. Teorik kapasiteyi vurgularken, öğrenme ve genelleştirmenin pratik zorluklarını optimizasyon algoritmaları ve mimari tasarım seçimleriyle ele alınmak üzere bırakır.

### 3. Yapay Sinir Ağları İçin Çıkarımlar
#### 3.1. İfade Gücünün Gerekçelendirilmesi
Evrensel Yaklaşım Teoreminin en derin çıkarımı, ileri beslemeli yapay sinir ağlarının **ifade gücünü** gerekçelendirmedeki rolüdür. EYT'den önce, yapay sinir ağlarının karmaşık, doğrusal olmayan ilişkileri modelleme yeteneği büyük ölçüde deneyseldi. Teorem, yapay sinir ağlarının sadece sofistike eğri uydurma araçları olmadığını, aynı zamanda muazzam teorik kapasiteye sahip **fonksiyon yaklaştırıcılar** olduğunu gösteren titiz bir teorik temel sağladı. Bu teorik destek, yapay sinir ağlarını niş bir akademik ilgiden çeşitli alanlarda güçlü bir araca dönüştürerek, alandaki daha fazla araştırma ve geliştirmeyi teşvik etti. Temelde, bir fonksiyon varsa, bir yapay sinir ağının prensipte onu temsil edebileceğini belirtir.

#### 3.2. Yaklaşım ve Öğrenme Arasındaki Fark
**Yaklaşım** ve **öğrenme** arasında ayrım yapmak kritik öneme sahiptir. EYT, herhangi bir verilen fonksiyonu yaklaşık olarak temsil edebilecek bir ağın *var olduğunu* garanti eder. Ancak, bu ideal ağı oluşturan belirli ağırlıkları ve önyargıları *bulmak* için bir algoritma veya yöntem sağlamaz. İşte burada **öğrenmenin** zorluğu ortaya çıkar. Bir yapay sinir ağını eğitmek, eğitim verilerine dayanarak ağırlıkları ve önyargıları ayarlamak için bir optimizasyon süreci (örn. gradyan inişi) içerir. EYT, bu optimizasyon algoritmalarının optimal ağırlıklar kümesine yakınsayacağını garanti etmez, ayrıca öğrenilen fonksiyonun görülmemiş verilere iyi genelleşeceğini ( **aşırı uyumu** önleyeceğini) de garanti etmez. Bu nedenle, yapay sinir ağları evrensel yaklaştırıcılar olsa da, sınırlı veriden iyi bir yaklaşımı başarıyla *öğrenmek*, aşağıdaki gibi faktörlerden etkilenen ayrı, karmaşık bir sorundur:
*   **Optimizasyon algoritması seçimi:** Stokastik Gradyan İnişi (SGD), Adam vb.
*   **Ağ mimarisi:** Katman sayısı, katman başına nöron sayısı.
*   **Düzenlileştirme teknikleri:** Dropout, L1/L2 düzenlileştirme.
*   **Eğitim verilerinin miktarı ve kalitesi.**

#### 3.3. Derin Öğrenmedeki Önemi
Başlangıçta, EYT **sığ ağlara** (tek bir gizli katman) odaklanmıştı. Ancak, birçok gizli katmana sahip ağları kullanan **derin öğrenmenin** yükselişi, daha fazla içgörüye yol açmıştır. Sığ ağlar teorik olarak evrensel yaklaşıma yetenekli olsa da, derin ağlar önemli pratik avantajlar göstermiştir:
*   **Parametre Verimliliği:** Derin ağlar genellikle karmaşık fonksiyonları sığ ağlara göre katlanarak daha az nöronla yaklaşık olarak temsil edebilir. Bu, hiyerarşik temsilleri öğrenme yeteneklerinden kaynaklanır; erken katmanlar basit özellikler çıkarırken, daha sonraki katmanlar bunları daha soyut ve karmaşık özellikler halinde birleştirir.
*   **Genelleştirme:** Birçok gerçek dünya senaryosunda, daha derin mimariler genellikle daha iyi genelleşir ve özellikle büyük veri kümelerinde aşırı uyuşmaya daha az eğilimlidir.
*   **Özellik Öğrenimi:** Derin ağlar, ham verilerden ilgili özellikleri otomatik olarak keşfederek, manuel özellik mühendisliği ihtiyacını azaltarak **özellik öğrenmede** başarılıdır.

Böylece, EYT, derinlikten bağımsız olarak yapay sinir ağlarının *potansiyeli* için temel bir teorik gerekçe görevi görür. Derin mimarilere geçiş, EYT'nin temel mesajı olan "bir yapay sinir ağı bunu yapabilir" mesajından yararlanırken, "bir *derin* yapay sinir ağının genellikle bunu *daha iyi* yapabileceği" ampirik ve teorik anlayışını ekleyerek verimlilik ve pratik performans tarafından yönlendirilir. Bu ilke, GAN'lar ve Transformatörler gibi karmaşık modellerin oldukça karmaşık veri dağılımlarını yaklaşık olarak temsil ettiği **Üretken Yapay Zeka**'da görülen ilerlemelerin temelini oluşturur.

### 4. Kod Örneği
Bu basit Python kod parçacığı, Evrensel Yaklaşım Teoremi'nin ima ettiği temel bir bileşen olan sigmoid aktivasyon fonksiyonuna sahip basit bir ileri beslemeli yapay sinir ağından geçen kavramsal ileri besleme geçişini göstermektedir. Girdilerin, ağırlıklı toplamlar ve doğrusal olmayan aktivasyonlar aracılığıyla nasıl işlenerek bir çıktı ürettiğini gösterir.

```python
import numpy as np

# Yaygın bir aktivasyon fonksiyonunu tanımlayın (örn. Sigmoid)
def sigmoid(x):
    """Sigmoid aktivasyon fonksiyonu."""
    return 1 / (1 + np.exp(-x))

# Basit bir iki katmanlı yapay sinir ağı tanımlayın (kavramsal ileri besleme geçişi)
def simple_neural_network_forward(inputs, weights_hidden, biases_hidden, weights_output, biases_output):
    """
    Basit bir iki katmanlı yapay sinir ağından kavramsal bir ileri besleme geçişi yapar.
    Bu, temel işlemleri (ağırlıklı toplam + aktivasyon) gösterir.
    """
    # Gizli katman hesaplaması:
    # 1. Girişlerin ağırlıklı toplamı artı gizli katman için önyargılar.
    hidden_layer_input = np.dot(inputs, weights_hidden) + biases_hidden
    # 2. Gizli katmanın girdisine aktivasyon fonksiyonunu uygulayın.
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Çıkış katmanı hesaplaması:
    # 1. Gizli katman çıktılarının ağırlıklı toplamı artı çıkış katmanı için önyargılar.
    output_layer_input = np.dot(hidden_layer_output, weights_output) + biases_output
    # 2. Basit bir regresyon görevi için, çıkış katmanında aktivasyon olmayabilir.
    output = output_layer_input

    return output

# --- Örnek Kullanım ---
# Tek bir giriş özelliği, 2 gizli nöron ve tek bir çıkış varsayalım.
# Bu ağırlıklar ve önyargılar keyfidir ve gerçek bir senaryoda 'öğrenilecektir'.
input_val = np.array([0.5]) # Tek bir giriş değeri (örn. normalize edilmiş veri noktası)

# Gizli katman için ağırlıklar ve önyargılar (1 giriş -> 2 gizli nöron)
weights_h = np.array([[0.1, 0.2]]) # Gizli katman için ağırlık matrisi
biases_h = np.array([0.3, 0.4])    # Gizli katman için önyargı vektörü

# Çıkış katmanı için ağırlıklar ve önyargılar (2 gizli nöron -> 1 çıkış nöronu)
weights_o = np.array([[0.5], [0.6]]) # Çıkış katmanı için ağırlık matrisi
biases_o = np.array([0.7])         # Çıkış katmanı için önyargı vektörü

# Tahmin edilen bir çıktı elde etmek için ileri besleme geçişini gerçekleştirin
predicted_output = simple_neural_network_forward(input_val, weights_h, biases_h, weights_o, biases_o)

print(f"Giriş değeri: {input_val[0]}")
print(f"Tahmini Çıktı: {predicted_output[0]:.4f}")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
Evrensel Yaklaşım Teoremi, yapay sinir ağlarının teorik yeteneklerinin temelini oluşturan matematiksel bir ifadedir. Tek bir gizli katmana ve uygun doğrusal olmayan aktivasyon fonksiyonlarına sahip nispeten basit bir **ileri beslemeli yapay sinir ağının** bile, kompakt bir alan üzerinde herhangi bir sürekli fonksiyonu keyfi bir doğruluk derecesine kadar yaklaşık olarak temsil etme içsel kapasitesine sahip olduğunu titizlikle gösterir. Bu güçlü teorik garanti, sinir ağı araştırmalarını keşfetmek ve yatırım yapmak için gerekli gerekçeyi sağlamış, alanı sezgisel argümanların ötesine taşımıştır.

EYT, böyle bir yaklaşık ağın *varlığını* doğrulamakla birlikte, **öğrenme** ve **genelleştirmenin** pratik zorluklarını kasten göz ardı eder. Optimal ağırlıkları ve önyargıları nasıl bulacağını belirtmez, ne de verimli öğrenmeyi veya görülmemiş verilerde etkili performansı garanti etmez. Bunlar, optimizasyon algoritmalarındaki, düzenlileştirme tekniklerindeki ve mimari yeniliklerdeki gelişmelerle, özellikle de **derin öğrenme** alanında ele alınan zorluklardır. Daha derin ağlara doğru kayış, EYT'nin temel mesajından yararlanırken, sığ ağlar teorik olarak yetenekli olsa da, derin mimarilerin **Üretken Yapay Zeka**'da yaygın olan karmaşık görevler için verimlilik, özellik öğrenme ve genelleştirme açısından genellikle pratik avantajlar sunduğunu vurgulamaktadır.

Özünde, Evrensel Yaklaşım Teoremi "neden"i - neden sinir ağlarının karmaşık fonksiyon yaklaşımı için keşfedilmeye değer olduğunu - sağlarken, derin öğrenme, optimizasyon ve mimari tasarım alanındaki devam eden araştırmalar, pratik, gerçek dünya uygulamalarında "nasıl" ve "ne kadar iyi" sorularını ele almaya devam etmektedir. Mirası, makine öğreniminin tüm alanı ve özellikle modern yapay zekada tanık olunan gelişmiş yetenekler için kritik bir teorik temel taşı olarak yaşamaya devam etmektedir.


