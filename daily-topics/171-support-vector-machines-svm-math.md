# Support Vector Machines (SVM) Math

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Mathematical Concepts](#2-core-mathematical-concepts)
  - [2.1. The Hyperplane](#21-the-hyperplane)
  - [2.2. Support Vectors and the Margin](#22-support-vectors-and-the-margin)
  - [2.3. Functional vs. Geometric Margin](#23-functional-vs-geometric-margin)
- [3. The Optimization Problem](#3-the-optimization-problem)
  - [3.1. Hard Margin SVM](#31-hard-margin-svm)
  - [3.2. Soft Margin SVM (Linear Separability with Noise)](#32-soft-margin-svm-linear-separability-with-noise)
  - [3.3. The Kernel Trick (Non-linear Separability)](#33-the-kernel-trick-non-linear-separability)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction
**Support Vector Machines (SVMs)** constitute a powerful and versatile class of supervised learning models widely used for classification and regression tasks. At their core, SVMs are linear models, but through the application of the **kernel trick**, they can effectively model non-linear decision boundaries. The fundamental premise of SVMs lies in finding an optimal **hyperplane** that best separates data points belonging to different classes in a high-dimensional feature space. This "optimal" hyperplane is defined as the one that maximizes the **margin** between the closest data points of different classes, known as **support vectors**. This document delves into the rigorous mathematical foundations that underpin SVMs, outlining the key concepts, the optimization problem, and the elegant solution offered by the kernel trick. Understanding these mathematical principles is crucial for grasping the power and limitations of SVMs and for their effective application in various machine learning scenarios.

### 2. Core Mathematical Concepts

#### 2.1. The Hyperplane
In a **d-dimensional feature space**, a hyperplane is a (d-1)-dimensional subspace that divides the space into two half-spaces. For a binary classification problem, an SVM seeks to find such a hyperplane to separate the data points. Mathematically, a hyperplane can be represented by the equation:
$$ \mathbf{w} \cdot \mathbf{x} + b = 0 $$
where:
*   $\mathbf{w}$ is the **weight vector** (normal vector to the hyperplane).
*   $\mathbf{x}$ is a data point in the feature space.
*   $b$ is the **bias term** (or intercept).

For classification, given a new data point $\mathbf{x}$, its class is predicted based on the sign of the function $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$. If $f(\mathbf{x}) > 0$, it belongs to one class; if $f(\mathbf{x}) < 0$, it belongs to the other.

#### 2.2. Support Vectors and the Margin
The data points that lie closest to the separating hyperplane and directly influence its position and orientation are called **support vectors**. These are the critical elements of the dataset, as removing any other data point would not alter the optimal hyperplane.

The **margin** is the distance between the hyperplane and the closest data points from each class (the support vectors). SVMs aim to maximize this margin. Intuitively, a larger margin indicates a more robust separation, making the classifier less susceptible to misclassification on unseen data.

Two parallel hyperplanes define this margin:
$$ \mathbf{w} \cdot \mathbf{x} + b = 1 \quad \text{for class 1} $$
$$ \mathbf{w} \cdot \mathbf{x} + b = -1 \quad \text{for class -1} $$
The data points lying on these planes are the support vectors. All other data points must satisfy $\mathbf{w} \cdot \mathbf{x} + b \ge 1$ for class 1 and $\mathbf{w} \cdot \mathbf{x} + b \le -1$ for class -1.

#### 2.3. Functional vs. Geometric Margin
*   **Functional Margin:** For a single data point $(\mathbf{x}_i, y_i)$, where $y_i \in \{-1, 1\}$, the functional margin is defined as $y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$. For a dataset, the functional margin of the hyperplane $(\mathbf{w}, b)$ with respect to the training set is $\min_i y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$. SVMs require this functional margin to be at least 1 for all training points, which can be achieved by scaling $\mathbf{w}$ and $b$.

*   **Geometric Margin:** The geometric margin, denoted $\gamma$, is the actual Euclidean distance from a data point to the hyperplane. For a data point $\mathbf{x}_i$ and a hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$, the geometric margin is given by:
    $$ \gamma_i = \frac{y_i(\mathbf{w} \cdot \mathbf{x}_i + b)}{\| \mathbf{w} \|} $$
    The goal of SVM is to maximize the geometric margin. With the scaling constraint $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1$, the geometric margin for any point is at least $1/\| \mathbf{w} \|$. The total geometric margin between the two separating hyperplanes ($ \mathbf{w} \cdot \mathbf{x} + b = 1 $ and $ \mathbf{w} \cdot \mathbf{x} + b = -1 $) is $2/\| \mathbf{w} \|$. Maximizing this value is equivalent to minimizing $\| \mathbf{w} \|^2$.

### 3. The Optimization Problem

#### 3.1. Hard Margin SVM
For linearly separable data, the goal is to find $\mathbf{w}$ and $b$ that maximize the geometric margin. This translates into the following constrained optimization problem:

**Minimize:**
$$ \frac{1}{2} \| \mathbf{w} \|^2 $$
**Subject to:**
$$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 \quad \text{for all } i=1, \dots, N $$
This is a **convex optimization problem** (specifically, a quadratic programming problem) with a unique solution. It can be solved using **Lagrangian duality**. The dual problem is typically easier to solve and introduces the Lagrange multipliers $\alpha_i$, which are non-zero only for the support vectors. The solution for $\mathbf{w}$ in terms of $\alpha_i$ is $\mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i$. The bias $b$ can be found using any support vector.

#### 3.2. Soft Margin SVM (Linear Separability with Noise)
In most real-world scenarios, data is not perfectly linearly separable, or there might be outliers. The **Soft Margin SVM** introduces **slack variables**, $\xi_i \ge 0$, to allow some training points to violate the margin constraint or even be misclassified.

**Minimize:**
$$ \frac{1}{2} \| \mathbf{w} \|^2 + C \sum_{i=1}^N \xi_i $$
**Subject to:**
$$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 - \xi_i \quad \text{for all } i=1, \dots, N $$
$$ \xi_i \ge 0 \quad \text{for all } i=1, \dots, N $$
Here, $C > 0$ is a **regularization parameter** that controls the trade-off between maximizing the margin (minimizing $\| \mathbf{w} \|^2$) and minimizing the training error (minimizing $\sum \xi_i$).
*   A small $C$ allows for a larger margin but potentially more misclassifications.
*   A large $C$ enforces stricter adherence to the margin constraints, leading to a smaller margin but fewer misclassifications.

The dual formulation for the soft margin SVM is similar to the hard margin case, with the additional constraint $0 \le \alpha_i \le C$.

#### 3.3. The Kernel Trick (Non-linear Separability)
One of the most powerful aspects of SVMs is their ability to handle non-linearly separable data through the **kernel trick**. Instead of explicitly transforming the data into a higher-dimensional feature space where it might become linearly separable, the kernel trick performs this transformation implicitly.

Let $\phi(\mathbf{x})$ be a mapping function that transforms the original input $\mathbf{x}$ into a higher-dimensional space. The decision function in this new space would be $f(\mathbf{x}) = \mathbf{w} \cdot \phi(\mathbf{x}) + b$. The key insight is that the optimization problem (both primal and dual) for SVMs only depends on inner products of data points, i.e., $\mathbf{x}_i \cdot \mathbf{x}_j$. In the transformed space, this becomes $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$.

A **kernel function** $K(\mathbf{x}_i, \mathbf{x}_j)$ is a function that directly computes this inner product in the higher-dimensional space without explicitly performing the mapping $\phi(\mathbf{x})$.
$$ K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j) $$
By replacing $\mathbf{x}_i \cdot \mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$ in the dual problem, SVMs can learn non-linear decision boundaries efficiently. Common kernel functions include:
*   **Linear Kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$ (Equivalent to linear SVM).
*   **Polynomial Kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$.
*   **Radial Basis Function (RBF) / Gaussian Kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \| \mathbf{x}_i - \mathbf{x}_j \|^2)$. This is a very popular choice due to its flexibility.
*   **Sigmoid Kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)$.

The kernel trick elegantly allows SVMs to handle complex, non-linear relationships in data without incurring the computational cost of explicit high-dimensional transformations.

### 4. Code Example
This Python code snippet demonstrates how to use a Support Vector Classifier (`SVC`) from `scikit-learn` to classify a simple dataset, using the RBF kernel which is commonly applied for non-linear problems.

```python
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate a synthetic dataset for classification
# We use make_moons to create a non-linearly separable dataset
X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=42)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and train an SVM classifier with an RBF kernel
# C is the regularization parameter, gamma defines the influence of a single training example
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = svm_model.predict(X_test)

# 5. Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with RBF Kernel: {accuracy:.4f}")

# The support vectors are the data points closest to the hyperplane
# print(f"Number of Support Vectors: {len(svm_model.support_vectors_)}")
# print(f"First 5 Support Vectors:\n{svm_model.support_vectors_[:5]}")

(End of code example section)
```

### 5. Conclusion
The mathematical elegance of Support Vector Machines positions them as a cornerstone in the field of machine learning. From the fundamental concept of maximizing the geometric margin between classes, through the formulation of convex optimization problems for both hard and soft margin scenarios, to the ingenious application of the kernel trick for handling non-linear data, SVMs offer a robust and theoretically sound framework for classification and regression. Their reliance on support vectors makes them particularly efficient with high-dimensional data, as the complexity depends on the number of support vectors rather than the total number of training examples. While newer deep learning approaches have gained prominence, SVMs remain highly effective for many tasks, especially when data is limited, features are well-defined, or interpretability of the decision boundary is important. A deep understanding of their underlying mathematics is invaluable for any practitioner aiming to apply or further develop machine learning algorithms.

---
<br>

<a name="türkçe-içerik"></a>
## Destek Vektör Makineleri (SVM) Matematiği

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Matematiksel Kavramlar](#2-temel-matematiksel-kavramlar)
  - [2.1. Hiperdüzlem](#21-hiperdüzlem)
  - [2.2. Destek Vektörleri ve Marj](#22-destek-vektörleri-ve-marj)
  - [2.3. Fonksiyonel ve Geometrik Marj](#23-fonksiyonel-ve-geometrik-marj)
- [3. Optimizasyon Problemi](#3-optimizasyon-problemi)
  - [3.1. Sert Marj SVM](#31-sert-marj-svm)
  - [3.2. Yumuşak Marj SVM (Gürültülü Doğrusal Ayrılabilirlik)](#32-yumuşak-marj-svm-gürültülü-doğrusal-ayrılabilirlik)
  - [3.3. Çekirdek Hilesi (Doğrusal Olmayan Ayrılabilirlik)](#33-çekirdek-hilesi-doğrusal-olmayan-ayrılabilirlik)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş
**Destek Vektör Makineleri (SVM'ler)**, sınıflandırma ve regresyon görevlerinde yaygın olarak kullanılan güçlü ve çok yönlü bir denetimli öğrenme modelleri sınıfını oluşturur. Özünde, SVM'ler doğrusal modellerdir, ancak **çekirdek hilesi** uygulamasıyla doğrusal olmayan karar sınırlarını etkili bir şekilde modelleyebilirler. SVM'lerin temel prensibi, yüksek boyutlu bir özellik uzayında farklı sınıflara ait veri noktalarını en iyi şekilde ayıran optimal bir **hiperdüzlem** bulmaktır. Bu "optimal" hiperdüzlem, farklı sınıflardaki en yakın veri noktaları (yani **destek vektörleri**) arasındaki **marjı** maksimize eden hiperdüzlem olarak tanımlanır. Bu belge, SVM'lerin temelini oluşturan titiz matematiksel temelleri, anahtar kavramları, optimizasyon problemini ve çekirdek hilesinin sunduğu zarif çözümü detaylandırmaktadır. Bu matematiksel prensipleri anlamak, SVM'lerin gücünü ve sınırlamalarını kavramak ve çeşitli makine öğrenimi senaryolarında etkili bir şekilde uygulamak için kritik öneme sahiptir.

### 2. Temel Matematiksel Kavramlar

#### 2.1. Hiperdüzlem
**d boyutlu bir özellik uzayında**, bir hiperdüzlem, uzayı iki yarı uzaya bölen (d-1) boyutlu bir alt uzaydır. İkili bir sınıflandırma problemi için, bir SVM, veri noktalarını ayırmak üzere böyle bir hiperdüzlemi bulmaya çalışır. Matematiksel olarak, bir hiperdüzlem şu denklemle temsil edilebilir:
$$ \mathbf{w} \cdot \mathbf{x} + b = 0 $$
burada:
*   $\mathbf{w}$ **ağırlık vektörüdür** (hiperdüzleme normal vektör).
*   $\mathbf{x}$ özellik uzayındaki bir veri noktasıdır.
*   $b$ **önyargı terimidir** (veya kesişim noktası).

Sınıflandırma için, yeni bir veri noktası $\mathbf{x}$ verildiğinde, sınıfı $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$ fonksiyonunun işaretine göre tahmin edilir. Eğer $f(\mathbf{x}) > 0$ ise bir sınıfa; $f(\mathbf{x}) < 0$ ise diğer sınıfa aittir.

#### 2.2. Destek Vektörleri ve Marj
Ayırıcı hiperdüzleme en yakın olan ve onun konumunu ve yönünü doğrudan etkileyen veri noktalarına **destek vektörleri** denir. Bunlar veri kümesinin kritik elemanlarıdır, çünkü başka herhangi bir veri noktasının kaldırılması optimal hiperdüzlemi değiştirmez.

**Marj**, hiperdüzlem ile her sınıftan en yakın veri noktaları (destek vektörleri) arasındaki mesafedir. SVM'ler bu marjı maksimize etmeyi amaçlar. Sezgisel olarak, daha büyük bir marj, daha sağlam bir ayrımı gösterir ve sınıflandırıcının görünmeyen verilerde yanlış sınıflandırmaya daha az eğilimli olmasını sağlar.

Bu marjı tanımlayan iki paralel hiperdüzlem vardır:
$$ \mathbf{w} \cdot \mathbf{x} + b = 1 \quad \text{1. sınıf için} $$
$$ \mathbf{w} \cdot \mathbf{x} + b = -1 \quad \text{-1. sınıf için} $$
Bu düzlemler üzerinde yer alan veri noktaları destek vektörleridir. Diğer tüm veri noktaları, 1. sınıf için $\mathbf{w} \cdot \mathbf{x} + b \ge 1$ ve -1. sınıf için $\mathbf{w} \cdot \mathbf{x} + b \le -1$ koşulunu sağlamalıdır.

#### 2.3. Fonksiyonel ve Geometrik Marj
*   **Fonksiyonel Marj:** Bir veri noktası $(\mathbf{x}_i, y_i)$ için, burada $y_i \in \{-1, 1\}$, fonksiyonel marj $y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$ olarak tanımlanır. Bir veri kümesi için, $(\mathbf{w}, b)$ hiperdüzleminin eğitim kümesine göre fonksiyonel marjı $\min_i y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$ şeklindedir. SVM'ler, bu fonksiyonel marjın tüm eğitim noktaları için en az 1 olmasını gerektirir, bu da $\mathbf{w}$ ve $b$'yi ölçekleyerek sağlanabilir.

*   **Geometrik Marj:** $\gamma$ ile gösterilen geometrik marj, bir veri noktasından hiperdüzleme olan gerçek Öklid uzaklığıdır. Bir $\mathbf{x}_i$ veri noktası ve $\mathbf{w} \cdot \mathbf{x} + b = 0$ hiperdüzlemi için geometrik marj şu şekilde verilir:
    $$ \gamma_i = \frac{y_i(\mathbf{w} \cdot \mathbf{x}_i + b)}{\| \mathbf{w} \|} $$
    SVM'nin amacı geometrik marjı maksimize etmektir. $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1$ ölçekleme kısıtlamasıyla, herhangi bir nokta için geometrik marj en az $1/\| \mathbf{w} \|$ olur. İki ayırıcı hiperdüzlem ($ \mathbf{w} \cdot \mathbf{x} + b = 1 $ ve $ \mathbf{w} \cdot \mathbf{x} + b = -1 $) arasındaki toplam geometrik marj $2/\| \mathbf{w} \|$'dir. Bu değeri maksimize etmek, $\| \mathbf{w} \|^2$'yi minimize etmeye eşdeğerdir.

### 3. Optimizasyon Problemi

#### 3.1. Sert Marj SVM
Doğrusal olarak ayrılabilir veriler için amaç, geometrik marjı maksimize eden $\mathbf{w}$ ve $b$'yi bulmaktır. Bu, aşağıdaki kısıtlı optimizasyon problemine dönüşür:

**Minimize et:**
$$ \frac{1}{2} \| \mathbf{w} \|^2 $$
**Şuna tabidir:**
$$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 \quad \text{tüm } i=1, \dots, N \text{ için} $$
Bu bir **konveks optimizasyon problemi**dir (özel olarak, bir karesel programlama problemi) ve tek bir çözümü vardır. **Lagrange dualitesi** kullanılarak çözülebilir. Dual problem genellikle daha kolay çözülür ve sadece destek vektörleri için sıfır olmayan Lagrange çarpanları $\alpha_i$'yi tanıtır. $\mathbf{w}$'nin $\alpha_i$ cinsinden çözümü $\mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i$'dir. Önyargı $b$ herhangi bir destek vektörü kullanılarak bulunabilir.

#### 3.2. Yumuşak Marj SVM (Gürültülü Doğrusal Ayrılabilirlik)
Çoğu gerçek dünya senaryosunda, veriler mükemmel bir şekilde doğrusal olarak ayrılabilir değildir veya aykırı değerler olabilir. **Yumuşak Marj SVM**, bazı eğitim noktalarının marj kısıtlamasını ihlal etmesine veya hatta yanlış sınıflandırılmasına izin vermek için **gevşek değişkenler** (slack variables), $\xi_i \ge 0$, tanıtır.

**Minimize et:**
$$ \frac{1}{2} \| \mathbf{w} \|^2 + C \sum_{i=1}^N \xi_i $$
**Şuna tabidir:**
$$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 - \xi_i \quad \text{tüm } i=1, \dots, N \text{ için} $$
$$ \xi_i \ge 0 \quad \text{tüm } i=1, \dots, N \text{ için} $$
Burada $C > 0$, marjı maksimize etme (yani $\| \mathbf{w} \|^2$'yi minimize etme) ile eğitim hatasını minimize etme (yani $\sum \xi_i$'yi minimize etme) arasındaki değişimi kontrol eden bir **düzenlileştirme parametresi**dir.
*   Küçük bir $C$ değeri, daha büyük bir marja ancak potansiyel olarak daha fazla yanlış sınıflandırmaya izin verir.
*   Büyük bir $C$ değeri, marj kısıtlamalarına daha sıkı uyulmasını sağlar, bu da daha küçük bir marj ve daha az yanlış sınıflandırma ile sonuçlanır.

Yumuşak marj SVM için dual formülasyon, sert marj durumuna benzerdir, ancak $0 \le \alpha_i \le C$ ek kısıtlamasıyla.

#### 3.3. Çekirdek Hilesi (Doğrusal Olmayan Ayrılabilirlik)
SVM'lerin en güçlü yönlerinden biri, **çekirdek hilesi** aracılığıyla doğrusal olarak ayrılabilir olmayan verileri işleme yeteneğidir. Verileri, doğrusal olarak ayrılabilir hale gelebileceği yüksek boyutlu bir özellik uzayına açıkça dönüştürmek yerine, çekirdek hilesi bu dönüşümü dolaylı olarak gerçekleştirir.

$\phi(\mathbf{x})$, orijinal giriş $\mathbf{x}$'i daha yüksek boyutlu bir uzaya dönüştüren bir eşleme fonksiyonu olsun. Bu yeni uzaydaki karar fonksiyonu $f(\mathbf{x}) = \mathbf{w} \cdot \phi(\mathbf{x}) + b$ olacaktır. Temel fikir, SVM'ler için optimizasyon probleminin (hem primal hem de dual) yalnızca veri noktalarının iç çarpımlarına, yani $\mathbf{x}_i \cdot \mathbf{x}_j$'ye bağlı olmasıdır. Dönüştürülmüş uzayda bu, $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$ haline gelir.

Bir **çekirdek fonksiyonu** $K(\mathbf{x}_i, \mathbf{x}_j)$, bu iç çarpımı yüksek boyutlu uzayda, $\phi(\mathbf{x})$ eşlemesini açıkça gerçekleştirmeden doğrudan hesaplayan bir fonksiyondur.
$$ K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j) $$
Dual problemde $\mathbf{x}_i \cdot \mathbf{x}_j$ yerine $K(\mathbf{x}_i, \mathbf{x}_j)$ kullanılarak, SVM'ler doğrusal olmayan karar sınırlarını verimli bir şekilde öğrenebilirler. Yaygın çekirdek fonksiyonları şunlardır:
*   **Doğrusal Çekirdek:** $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$ (Doğrusal SVM'ye eşdeğerdir).
*   **Polinom Çekirdek:** $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$.
*   **Radyal Temel Fonksiyon (RBF) / Gauss Çekirdeği:** $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \| \mathbf{x}_i - \mathbf{x}_j \|^2)$. Esnekliği nedeniyle çok popüler bir seçimdir.
*   **Sigmoid Çekirdek:** $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)$.

Çekirdek hilesi, SVM'lerin veri içindeki karmaşık, doğrusal olmayan ilişkileri, açık yüksek boyutlu dönüşümlerin hesaplama maliyetine katlanmadan zarif bir şekilde ele almasına olanak tanır.

### 4. Kod Örneği
Bu Python kod parçacığı, `scikit-learn` kütüphanesinden bir Destek Vektör Sınıflandırıcısını (`SVC`) kullanarak basit bir veri kümesini sınıflandırmayı göstermektedir. Özellikle doğrusal olmayan problemler için sıkça kullanılan RBF çekirdeği kullanılmıştır.

```python
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Sınıflandırma için sentetik bir veri kümesi oluştur
# Doğrusal olmayan şekilde ayrılabilir bir veri kümesi oluşturmak için make_moons kullanıyoruz
X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=42)

# 2. Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. RBF çekirdeği ile bir SVM sınıflandırıcısını başlat ve eğit
# C düzenlileştirme parametresi, gamma tek bir eğitim örneğinin etkisini tanımlar
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 4. Test kümesinde tahminler yap
y_pred = svm_model.predict(X_test)

# 5. Modelin doğruluğunu değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF Çekirdeği ile Model Doğruluğu: {accuracy:.4f}")

# Destek vektörleri, hiperdüzleme en yakın olan veri noktalarıdır
# print(f"Destek Vektörlerinin Sayısı: {len(svm_model.support_vectors_)}")
# print(f"İlk 5 Destek Vektörü:\n{svm_model.support_vectors_[:5]}")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
Destek Vektör Makinelerinin matematiksel zarafeti, onları makine öğrenimi alanında bir köşe taşı olarak konumlandırır. Sınıflar arasındaki geometrik marjı maksimize etme temel kavramından, hem sert hem de yumuşak marj senaryoları için konveks optimizasyon problemlerinin formülasyonuna, doğrusal olmayan verileri işlemek için çekirdek hilesinin ustaca uygulanmasına kadar, SVM'ler sınıflandırma ve regresyon için sağlam ve teorik olarak sağlam bir çerçeve sunar. Destek vektörlerine bağımlılıkları, yüksek boyutlu verilerle özellikle verimli olmalarını sağlar, çünkü karmaşıklık toplam eğitim örneği sayısından ziyade destek vektörlerinin sayısına bağlıdır. Yeni derin öğrenme yaklaşımları öne çıkmış olsa da, SVM'ler birçok görev için, özellikle verilerin sınırlı olduğu, özelliklerin iyi tanımlandığı veya karar sınırının yorumlanabilirliğinin önemli olduğu durumlarda oldukça etkili olmaya devam etmektedir. Temel matematiklerinin derinlemesine anlaşılması, makine öğrenimi algoritmalarını uygulamayı veya daha da geliştirmeyi amaçlayan her uygulayıcı için paha biçilmezdir.