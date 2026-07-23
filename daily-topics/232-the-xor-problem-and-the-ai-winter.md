# The XOR Problem and the AI Winter

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The XOR Problem: A Fundamental Challenge](#2-the-xor-problem-a-fundamental-challenge)
- [3. The Perceptron's Limitation and Minsky & Papert's Critique](#3-the-perceptrons-limitation-and-minsky--paperts-critique)
- [4. The AI Winter: A Period of Disillusionment](#4-the-ai-winter-a-period-of-disillusionment)
- [5. Overcoming Limitations: The Path to Modern AI](#5-overcoming-limitations-the-path-to-modern-ai)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

## 1. Introduction
The early history of Artificial Intelligence (AI) is characterized by cycles of immense optimism followed by periods of disillusionment, famously dubbed "AI winters." One of the pivotal moments contributing to the first AI winter was the **XOR Problem**, a seemingly simple logical challenge that exposed a fundamental limitation of the prevailing neural network architecture of the time: the **single-layer perceptron**. This document delves into the nature of the XOR Problem, the architectural constraints it highlighted, the critical role played by Minsky and Papert's analysis, and the subsequent impact on AI research and funding, ultimately paving the way for the sophisticated deep learning models we utilize today. Understanding this historical episode is crucial for appreciating the iterative nature of scientific progress and the resilience required in developing complex computational systems.

## 2. The XOR Problem: A Fundamental Challenge
The **XOR (exclusive OR)** logical operation is a basic binary function that takes two binary inputs and produces a binary output. Its truth table is defined as follows:
- Input (0, 0) -> Output 0
- Input (0, 1) -> Output 1
- Input (1, 0) -> Output 1
- Input (1, 1) -> Output 0

From this truth table, it is evident that the XOR function yields a true (1) output if and only if its inputs are different. If the inputs are the same, the output is false (0). When plotted on a 2D Cartesian plane, with the inputs (x1, x2) as coordinates, the data points for XOR are (0,0) and (1,1) labeled as 0, and (0,1) and (1,0) labeled as 1.

The critical characteristic of the XOR problem is its **non-linear separability**. This means that there is no single straight line (or hyperplane in higher dimensions) that can separate the data points classified as '0' from those classified as '1'. The '0' points (0,0) and (1,1) are diagonally opposed, as are the '1' points (0,1) and (1,0). Any straight line drawn to separate (0,0) from (0,1) and (1,0) would inevitably misclassify (1,1), or vice-versa. This geometric interpretation is central to understanding why early neural networks struggled with this specific task.

## 3. The Perceptron's Limitation and Minsky & Papert's Critique
The **perceptron**, developed by Frank Rosenblatt in 1957, was one of the earliest models of an artificial neuron. A single-layer perceptron is designed to classify linearly separable data. It computes a weighted sum of its inputs, adds a bias, and then passes this result through an activation function (often a step function) to produce an output. If the weighted sum exceeds a certain threshold, it outputs one class; otherwise, it outputs another. This effectively creates a linear decision boundary in the input space.

For tasks like **AND** or **OR**, a single-layer perceptron works perfectly because their truth tables represent linearly separable data. For example, for the AND function, a line can easily separate (1,1) (output 1) from (0,0), (0,1), and (1,0) (all output 0).

However, because the XOR problem is not linearly separable, a single-layer perceptron is fundamentally incapable of learning and representing the XOR function. It cannot draw a single straight line to correctly classify all four input combinations.

This limitation was prominently highlighted by **Marvin Minsky and Seymour Papert** in their influential 1969 book, "Perceptrons." Their work provided a rigorous mathematical analysis of the limitations of single-layer perceptrons, unequivocally demonstrating their inability to solve non-linearly separable problems like XOR. While their analysis also hinted at the potential of **multi-layer perceptrons** (networks with hidden layers) to overcome these limitations, this aspect was largely overlooked or downplayed by the broader research community and funding agencies at the time. The emphasis was instead placed on the perceived inherent limitations of neural networks.

## 4. The AI Winter: A Period of Disillusionment
The publication of "Perceptrons," coupled with other factors, contributed significantly to the first major **AI winter**. This period, roughly spanning from the mid-1970s to the mid-1980s, saw a drastic reduction in funding, research interest, and public confidence in AI. The disillusionment stemmed from several converging issues:

1.  **Over-optimistic Promises:** Early AI researchers often made ambitious predictions about AI's capabilities that far exceeded the technology's actual readiness. This led to a gap between expectation and reality.
2.  **Computational Limitations:** The computing power available at the time was severely limited. Even simple AI tasks required significant resources, and complex neural networks (like multi-layer perceptrons) were computationally expensive to train and implement.
3.  **Lack of Data:** The concept of big data was non-existent. AI models, especially neural networks, thrive on vast amounts of data for effective training, which was largely unavailable.
4.  **The XOR Problem and Minsky & Papert's Critique:** This was arguably the most direct and academically rigorous blow to neural network research. The clear demonstration of the perceptron's inability to solve basic non-linear problems cast a long shadow over the entire field of connectionism (neural networks). Funding for neural network research essentially evaporated.

Governments and private institutions, having invested heavily with little immediate return, withdrew funding. Many researchers left the field, and AI became a marginalized academic discipline. The prevailing paradigm shifted towards symbolic AI, expert systems, and logic-based approaches, which seemed more promising for the computational resources and theoretical understanding of the era.

## 5. Overcoming Limitations: The Path to Modern AI
Despite the bleak outlook of the AI winter, a dedicated group of researchers continued to explore the potential of neural networks. The key to overcoming the XOR problem and other non-linear challenges lay in the concept of **multi-layer perceptrons (MLPs)**, also known as feedforward neural networks with one or more **hidden layers**.

A hidden layer allows the network to learn intermediate representations of the input data. Essentially, it transforms the input into a new, higher-dimensional feature space where the classes *can* be linearly separated. For the XOR problem, a multi-layer perceptron can learn to create two linear boundaries that, when combined, effectively separate the non-linearly separable points.

The crucial algorithmic breakthrough that enabled the training of MLPs was **backpropagation**, popularized by Rumelhart, Hinton, and Williams in 1986. Backpropagation is an algorithm that efficiently computes the gradients of the loss function with respect to the weights of a neural network, allowing for iterative adjustment of these weights using gradient descent. This meant that networks with multiple layers could finally be trained effectively to learn complex, non-linear mappings.

The re-discovery and popularization of backpropagation, combined with increasing computational power (especially with GPUs later on) and the availability of larger datasets, gradually led to a resurgence of interest in neural networks. This eventually blossomed into the modern era of **deep learning**, where complex multi-layered architectures solve problems far more intricate than XOR, from image recognition and natural language processing to drug discovery and autonomous driving. The XOR problem, once a symbol of AI's limitations, became a fundamental pedagogical example for demonstrating the power of hidden layers and non-linear activation functions in neural networks.

## 6. Code Example
Here's a simple Python code snippet illustrating how a basic single-layer perceptron *fails* to solve the XOR problem. It attempts to learn the weights, but cannot find a linear boundary.

```python
import numpy as np

# XOR inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Simple Perceptron class (without a bias term for simplicity,
# but the concept of linear separation limitation still holds)
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X, y):
        # Initialize weights randomly, adding 1 for the 'bias' equivalent effect if needed
        # For a truly single-layer perceptron without explicit bias, weights would match input features
        self.weights = np.random.rand(X.shape[1]) # Two weights for two inputs

        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                # Calculate the predicted output
                prediction = self.predict_step(X[i])
                
                # Update weights based on error
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]

    def predict_step(self, inputs):
        # Step function activation: 1 if weighted sum >= 0.5, else 0
        # (Using 0.5 as threshold for binary classification based on typical activation output range)
        return 1 if np.dot(inputs, self.weights) >= 0.5 else 0

# Train the perceptron
perceptron = Perceptron()
perceptron.fit(X, y)

# Test the perceptron
print("Single-Layer Perceptron Predictions for XOR:")
for i in range(X.shape[0]):
    prediction = perceptron.predict_step(X[i])
    print(f"Input: {X[i]}, Expected: {y[i]}, Predicted: {prediction}")

# The output will show incorrect predictions for XOR, demonstrating its failure.
# A multi-layer perceptron with a hidden layer would be required to solve this.

(End of code example section)
```

## 7. Conclusion
The XOR Problem stands as a monumental cautionary tale and a pivotal turning point in the history of Artificial Intelligence. It vividly exposed the limitations of early neural network architectures, specifically the single-layer perceptron, and played a significant role in ushering in the first AI winter. However, this period of reduced funding and skepticism was not without its merits; it fostered deeper theoretical understanding and spurred the development of more sophisticated models and algorithms, such as multi-layer perceptrons and backpropagation. The ultimate triumph over the XOR problem, from a theoretical curiosity to a solvable benchmark, symbolizes the iterative nature of scientific discovery. It underscores the importance of persistent research, even in times of disillusionment, and serves as a foundational lesson in understanding the power of non-linear transformations and layered architectures that are central to the success of modern Generative AI and deep learning. The journey from the XOR dilemma to today's complex neural networks exemplifies how overcoming seemingly simple challenges can unlock revolutionary advancements.

---
<br>

<a name="türkçe-içerik"></a>
## XOR Problemi ve Yapay Zeka Kışı

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. XOR Problemi: Temel Bir Zorluk](#2-xor-problemi-temel-bir-zorluk)
- [3. Perceptron'un Sınırlaması ve Minsky & Papert'in Eleştirisi](#3-perceptronun-sınırlaması-ve-minsky--papertin-eleştirisi)
- [4. Yapay Zeka Kışı: Hayal Kırıklığı Dönemi](#4-yapay-zeka-kışı-hayal-kırıklığı-dönemi)
- [5. Sınırlamaların Aşılması: Modern Yapay Zekaya Giden Yol](#5-sınırlamaların-aşılması-modern-yapay-zekaya-giden-yol)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

## 1. Giriş
Yapay Zeka (YZ) tarihin ilk dönemleri, büyük iyimserlik dalgalarının ardından, meşhur "YZ kışları" olarak adlandırılan hayal kırıklığı dönemleriyle karakterize edilir. İlk YZ kışına katkıda bulunan önemli dönüm noktalarından biri, dönemin baskın sinir ağı mimarisinin, yani **tek katmanlı perceptron'un** temel bir sınırlamasını ortaya çıkaran, görünüşte basit bir mantıksal zorluk olan **XOR Problemi** idi. Bu belge, XOR Problemi'nin doğasını, vurguladığı mimari kısıtlamaları, Minsky ve Papert'in analizinin kritik rolünü ve YZ araştırmaları ile fonları üzerindeki sonraki etkisini derinlemesine inceleyerek, nihayetinde günümüzde kullandığımız sofistike derin öğrenme modellerine giden yolu açmaktadır. Bu tarihi dönemi anlamak, bilimsel ilerlemenin yinelemeli doğasını ve karmaşık hesaplama sistemleri geliştirmek için gereken esnekliği takdir etmek açısından hayati önem taşımaktadır.

## 2. XOR Problemi: Temel Bir Zorluk
**XOR (özel VEYA)** mantık işlemi, iki ikili girdi alan ve ikili bir çıktı üreten temel bir ikili fonksiyondur. Doğruluk tablosu şu şekilde tanımlanır:
- Girdi (0, 0) -> Çıktı 0
- Girdi (0, 1) -> Çıktı 1
- Girdi (1, 0) -> Çıktı 1
- Girdi (1, 1) -> Çıktı 0

Bu doğruluk tablosundan, XOR fonksiyonunun ancak ve ancak girdileri farklıysa doğru (1) bir çıktı verdiği açıktır. Girdiler aynıysa, çıktı yanlış (0) olur. Girdileri (x1, x2) koordinatlar olarak kullanarak 2 boyutlu bir Kartezyen düzlemde çizildiğinde, XOR için veri noktaları (0,0) ve (1,1) '0' olarak, (0,1) ve (1,0) ise '1' olarak etiketlenir.

XOR probleminin kritik özelliği, **doğrusal olarak ayrılamaz** olmasıdır. Bu, '0' olarak sınıflandırılan veri noktalarını '1' olarak sınıflandırılanlardan ayırabilecek tek bir düz çizgi (veya daha yüksek boyutlarda hiperdüzlem) olmadığı anlamına gelir. '0' noktaları (0,0) ve (1,1) çapraz olarak birbirine zıttır, tıpkı '1' noktaları (0,1) ve (1,0) gibi. (0,0)'ı (0,1) ve (1,0)'dan ayırmak için çizilen herhangi bir düz çizgi, kaçınılmaz olarak (1,1)'i yanlış sınıflandıracaktır veya tam tersi. Bu geometrik yorum, erken sinir ağlarının bu özel görevde neden zorlandığını anlamanın merkezindedir.

## 3. Perceptron'un Sınırlaması ve Minsky & Papert'in Eleştirisi
Frank Rosenblatt tarafından 1957'de geliştirilen **perceptron**, yapay bir nöronun en eski modellerinden biriydi. Tek katmanlı bir perceptron, doğrusal olarak ayrılabilir verileri sınıflandırmak için tasarlanmıştır. Girdilerinin ağırlıklı toplamını hesaplar, bir sapma (bias) ekler ve ardından bu sonucu bir aktivasyon fonksiyonundan (genellikle bir basamak fonksiyonu) geçirerek bir çıktı üretir. Ağırlıklı toplam belirli bir eşiği aşarsa, bir sınıfı; aksi takdirde başka bir sınıfı çıktı olarak verir. Bu, girdi uzayında etkili bir şekilde doğrusal bir karar sınırı oluşturur.

**AND** veya **OR** gibi görevler için tek katmanlı bir perceptron mükemmel çalışır, çünkü doğruluk tabloları doğrusal olarak ayrılabilir verileri temsil eder. Örneğin, AND fonksiyonu için, (1,1) (çıktı 1) ile (0,0), (0,1) ve (1,0) (tümü çıktı 0) arasında kolayca bir çizgi çekilebilir.

Ancak, XOR problemi doğrusal olarak ayrılamaz olduğundan, tek katmanlı bir perceptron, XOR fonksiyonunu öğrenme ve temsil etme konusunda temel olarak yetersizdir. Dört girdi kombinasyonunun tamamını doğru şekilde sınıflandıracak tek bir düz çizgi çizemez.

Bu sınırlama, **Marvin Minsky ve Seymour Papert** tarafından 1969 tarihli etkili kitapları "Perceptrons"ta belirgin bir şekilde vurgulanmıştır. Çalışmaları, tek katmanlı perceptron'ların sınırlamalarına dair titiz bir matematiksel analiz sunarak, XOR gibi doğrusal olarak ayrılamaz problemleri çözme yeteneklerinin olmadığını kesin olarak göstermiştir. Analizleri aynı zamanda **çok katmanlı perceptron'ların** (gizli katmanlara sahip ağlar) bu sınırlamaları aşma potansiyeline de işaret etse de, bu yön o dönemdeki genel araştırma topluluğu ve fon sağlayıcı kurumlar tarafından büyük ölçüde göz ardı edildi veya küçümsendi. Bunun yerine, sinir ağlarının algılanan içsel sınırlamalarına odaklanıldı.

## 4. Yapay Zeka Kışı: Hayal Kırıklığı Dönemi
"Perceptrons" kitabının yayımlanması, diğer faktörlerle birlikte, ilk büyük **YZ kışına** önemli ölçüde katkıda bulundu. Yaklaşık olarak 1970'lerin ortalarından 1980'lerin ortalarına kadar süren bu dönem, YZ'ye olan fonlarda, araştırma ilgisinde ve kamu güveninde büyük bir düşüş gördü. Hayal kırıklığı, birkaç birleşen sorundan kaynaklanıyordu:

1.  **Aşırı İyimser Vaatler:** Erken YZ araştırmacıları, YZ'nin yetenekleri hakkında teknolojinin gerçek hazır bulunuşluğunu çok aşan iddialı tahminlerde bulundular. Bu, beklenti ile gerçeklik arasında bir boşluğa yol açtı.
2.  **Hesaplama Sınırlamaları:** O dönemdeki hesaplama gücü ciddi şekilde sınırlıydı. En basit YZ görevleri bile önemli kaynaklar gerektiriyordu ve karmaşık sinir ağları (çok katmanlı perceptronlar gibi) eğitilmesi ve uygulanması açısından hesaplama açısından pahalıydı.
3.  **Veri Eksikliği:** Büyük veri kavramı mevcut değildi. YZ modelleri, özellikle sinir ağları, etkili eğitim için büyük miktarda veriye ihtiyaç duyar, ki bu büyük ölçüde mevcut değildi.
4.  **XOR Problemi ve Minsky & Papert'in Eleştirisi:** Bu, sinir ağı araştırmalarına şüphesiz en doğrudan ve akademik olarak sağlam darbe oldu. Perceptron'un temel doğrusal olmayan problemleri çözme yeteneksizliğinin açıkça gösterilmesi, tüm bağlantıcılık (sinir ağları) alanı üzerinde uzun bir gölge bıraktı. Sinir ağı araştırmaları için fonlar esasen buharlaştı.

Hükümetler ve özel kurumlar, az anında getiri ile ağır yatırım yapmış olmalarına rağmen, fonları çektiler. Birçok araştırmacı alanı terk etti ve YZ marjinalleşmiş bir akademik disiplin haline geldi. Baskın paradigma, dönemin hesaplama kaynakları ve teorik anlayışı için daha umut vadeden sembolik YZ, uzman sistemler ve mantık tabanlı yaklaşımlara kaydı.

## 5. Sınırlamaların Aşılması: Modern Yapay Zekaya Giden Yol
YZ kışının kasvetli görünümüne rağmen, özel bir araştırmacı grubu sinir ağlarının potansiyelini keşfetmeye devam etti. XOR problemini ve diğer doğrusal olmayan zorlukları aşmanın anahtarı, bir veya daha fazla **gizli katmana** sahip ileri beslemeli sinir ağları olarak da bilinen **çok katmanlı perceptronlar (MLP'ler)** kavramında yatıyordu.

Bir gizli katman, ağın girdi verilerinin ara temsillerini öğrenmesine olanak tanır. Esasen, girdiyi, sınıfların doğrusal olarak ayrılabildiği yeni, daha yüksek boyutlu bir özellik uzayına dönüştürür. XOR problemi için, çok katmanlı bir perceptron, bir araya geldiğinde doğrusal olarak ayrılamaz noktaları etkili bir şekilde ayıran iki doğrusal sınır oluşturmayı öğrenebilir.

MLP'lerin eğitimini mümkün kılan kritik algoritmik atılım, Rumelhart, Hinton ve Williams tarafından 1986'da popülerleştirilen **geri yayılım (backpropagation)** idi. Geri yayılım, bir sinir ağının ağırlıklarına göre kayıp fonksiyonunun gradyanlarını verimli bir şekilde hesaplayan bir algoritmadır, bu da gradyan inişi kullanarak bu ağırlıkların yinelemeli olarak ayarlanmasına olanak tanır. Bu, birden çok katmana sahip ağların nihayet karmaşık, doğrusal olmayan eşlemeleri öğrenmek için etkili bir şekilde eğitilebileceği anlamına geliyordu.

Geri yayılımın yeniden keşfedilmesi ve popülerleşmesi, artan hesaplama gücü (özellikle daha sonra GPU'larla) ve daha büyük veri kümelerinin mevcudiyeti ile birleşerek, sinir ağlarına olan ilginin kademeli olarak yeniden canlanmasına yol açtı. Bu, nihayetinde görüntü tanıma ve doğal dil işlemeden ilaç keşfi ve otonom sürüşe kadar XOR'dan çok daha karmaşık sorunları çözen karmaşık çok katmanlı mimarilerin olduğu modern **derin öğrenme** çağına dönüştü. Bir zamanlar YZ'nin sınırlamalarının sembolü olan XOR problemi, sinir ağlarında gizli katmanların ve doğrusal olmayan aktivasyon fonksiyonlarının gücünü göstermek için temel bir pedagojik örnek haline geldi.

## 6. Kod Örneği
İşte temel bir tek katmanlı perceptron'un XOR problemini çözmede nasıl *başarısız olduğunu* gösteren basit bir Python kod parçacığı. Ağırlıkları öğrenmeye çalışır, ancak doğrusal bir sınır bulamaz.

```python
import numpy as np

# XOR girdileri ve çıktıları
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Basit Perceptron sınıfı (basitlik için bias terimi olmadan,
# ancak doğrusal ayrılma sınırlaması kavramı hala geçerlidir)
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X, y):
        # Ağırlıkları rastgele başlat, gerekirse 'bias' eşdeğeri etki için 1 ekle
        # Açık bir bias terimi olmayan gerçekten tek katmanlı bir perceptron için, ağırlıklar girdi özellikleriyle eşleşmelidir
        self.weights = np.random.rand(X.shape[1]) # İki girdi için iki ağırlık

        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                # Tahmin edilen çıktıyı hesapla
                prediction = self.predict_step(X[i])
                
                # Hataya göre ağırlıkları güncelle
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]

    def predict_step(self, inputs):
        # Basamak fonksiyonu aktivasyonu: ağırlıklı toplam >= 0.5 ise 1, aksi takdirde 0
        # (Tipik aktivasyon çıktı aralığına göre ikili sınıflandırma için eşik olarak 0.5 kullanılır)
        return 1 if np.dot(inputs, self.weights) >= 0.5 else 0

# Perceptron'u eğit
perceptron = Perceptron()
perceptron.fit(X, y)

# Perceptron'u test et
print("Tek Katmanlı Perceptron'un XOR İçin Tahminleri:")
for i in range(X.shape[0]):
    prediction = perceptron.predict_step(X[i])
    print(f"Girdi: {X[i]}, Beklenen: {y[i]}, Tahmin Edilen: {prediction}")

# Çıktı, XOR için yanlış tahminler gösterecek ve başarısızlığını kanıtlayacaktır.
# Bunu çözmek için gizli katmana sahip çok katmanlı bir perceptron gereklidir.

(Kod örneği bölümünün sonu)
```

## 7. Sonuç
XOR Problemi, Yapay Zeka tarihinde anıtsal bir uyarıcı hikaye ve önemli bir dönüm noktası olarak durmaktadır. Erken sinir ağı mimarilerinin, özellikle tek katmanlı perceptron'un sınırlamalarını canlı bir şekilde ortaya koymuş ve ilk YZ kışını başlatmada önemli bir rol oynamıştır. Ancak, fonların azalması ve şüphecilikle geçen bu dönem, kendi yararları olmadan değildi; daha derin teorik anlayışı besledi ve çok katmanlı perceptronlar ve geri yayılım gibi daha sofistike modellerin ve algoritmaların geliştirilmesini teşvik etti. XOR probleminin, teorik bir merak olmaktan çözülebilir bir kıyaslama haline gelmesi, bilimsel keşfin yinelemeli doğasını simgelemektedir. Hayal kırıklığı zamanlarında bile ısrarlı araştırmanın önemini vurgular ve modern Üretken YZ ve derin öğrenmenin başarısının merkezinde yer alan doğrusal olmayan dönüşümlerin ve katmanlı mimarilerin gücünü anlamak için temel bir ders görevi görür. XOR ikileminden günümüzün karmaşık sinir ağlarına uzanan yolculuk, görünüşte basit zorlukların üstesinden gelmenin nasıl devrim niteliğinde ilerlemelerin kilidini açabileceğini göstermektedir.

