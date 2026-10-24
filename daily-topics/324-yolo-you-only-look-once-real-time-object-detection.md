# YOLO (You Only Look Once): Real-Time Object Detection

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Evolution of Object Detection Paradigms](#2-the-evolution-of-object-detection-paradigms)
- [3. YOLO's Core Principles and Architecture](#3-yolos-core-principles-and-architecture)
- [4. Key YOLO Versions and Advancements](#4-key-yolo-versions-and-advancements)
- [5. Advantages, Limitations, and Applications](#5-advantages-limitations-and-applications)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)
- [8. References and Further Reading](#8-references-and-further-reading)

<a name="1-introduction"></a>
## 1. Introduction

Object detection is a fundamental task in computer vision, aiming to identify and localize objects of interest within an image or video frame. This involves not only classifying what objects are present but also drawing **bounding boxes** around their exact locations. The field has seen remarkable progress, driven largely by the advent of deep learning. Among the most influential and widely adopted approaches is **YOLO (You Only Look Once)**, a revolutionary family of models renowned for its exceptional speed and competitive accuracy in real-time object detection.

Introduced by Joseph Redmon et al. in 2015, YOLO presented a paradigm shift by reframing object detection as a single regression problem, directly predicting bounding box coordinates and class probabilities from full images in one pass. This contrasts sharply with previous multi-stage detectors that relied on region proposal networks. YOLO's "one-shot" approach made real-time applications viable for the first time, paving the way for its integration into diverse fields such as autonomous driving, surveillance, robotics, and augmented reality. This document explores the core principles, architectural evolution, advantages, and limitations of the YOLO framework.

<a name="2-the-evolution-of-object-detection-paradigms"></a>
## 2. The Evolution of Object Detection Paradigms

Before YOLO, object detection largely followed a two-stage approach. Early methods combined handcrafted features (e.g., SIFT, HOG) with classifiers (e.g., SVMs). The deep learning era brought about significant improvements, primarily with the **Region-based Convolutional Neural Network (R-CNN)** family.

**R-CNN (Regions with CNN features)**, proposed in 2014, marked a breakthrough by integrating CNNs. It involved three main steps:
1.  **Region Proposal:** A selective search algorithm generated approximately 2000 region proposals (potential object locations).
2.  **Feature Extraction:** Each proposed region was warped to a fixed size and passed through a pre-trained CNN to extract features.
3.  **Classification and Bounding Box Regression:** Support Vector Machines (SVMs) classified the features, and a separate regressor refined the bounding box coordinates.
While accurate, R-CNN was computationally expensive and slow due to processing each region independently.

**Fast R-CNN** addressed some of R-CNN's inefficiencies by processing the entire image with a CNN once and then projecting region proposals onto the feature map. This reduced redundant computations. **Faster R-CNN** further improved speed by replacing the selective search algorithm with a **Region Proposal Network (RPN)**, which was trained jointly with the detector. This created an end-to-end, albeit still two-stage, detection system that was significantly faster and more accurate than its predecessors.

Despite these advancements, two-stage detectors still suffered from inherent latency due to their sequential nature. They first proposed regions and then classified and refined them. This fundamental limitation spurred the development of **one-stage detectors**, where detection was formulated as a single regression problem, predicting all bounding boxes and class probabilities simultaneously. YOLO emerged as a leading exponent of this new paradigm.

<a name="3-yolos-core-principles-and-architecture"></a>
## 3. YOLO's Core Principles and Architecture

YOLO's fundamental innovation lies in its unified architecture, which performs object localization and classification in a single pass of a convolutional neural network. This "one-shot" approach directly predicts bounding boxes and class probabilities without the need for a separate region proposal step.

The core principles can be summarized as follows:

### Grid System
The input image is divided into an $S \times S$ grid. If the center of an object falls into a grid cell, that cell becomes responsible for detecting the object.

### Bounding Box Prediction
Each grid cell predicts $B$ bounding boxes. For each bounding box, it predicts:
*   **$x, y$ coordinates:** The center of the box relative to the grid cell boundaries.
*   **$w, h$ dimensions:** The width and height relative to the full image dimensions.
*   **Confidence score:** This score reflects two aspects: the probability that the box contains an object (objectness) and the **Intersection Over Union (IOU)** between the predicted box and any ground-truth box. A high confidence score indicates a high probability of an object and accurate localization.

### Class Probability Prediction
Each grid cell also predicts $C$ **conditional class probabilities** ($P(\text{Class}_i | \text{Object})$). These are probabilities of the object belonging to a specific class, given that an object is detected in the cell. These probabilities are conditional because they only make sense if an object is present.

### Final Detection
For each bounding box, the final class-specific confidence score is calculated by multiplying the bounding box confidence with the conditional class probabilities:
$$ \text{Confidence} = P(\text{Object}) * \text{IOU}_{\text{pred}}^{\text{truth}} \times P(\text{Class}_i | \text{Object}) = P(\text{Class}_i) * \text{IOU}_{\text{pred}}^{\text{truth}} $$
This score represents both the probability of that class appearing in the box and how well the predicted box fits the object. Boxes with scores below a certain threshold are discarded.

### Non-Maximum Suppression (NMS)
YOLO often predicts multiple overlapping bounding boxes for the same object. **Non-Maximum Suppression (NMS)** is applied to eliminate redundant detections. It selects the bounding box with the highest confidence score among overlapping boxes and suppresses others that have a high IOU with the selected box.

### Network Architecture (YOLOv1)
The original YOLOv1 network typically consisted of 24 convolutional layers followed by 2 fully connected layers. It used **Leaky ReLU** activations and a **linear activation function** for the final output layer. The output of the network is a tensor of size $S \times S \times (B \times 5 + C)$, encapsulating all predictions for bounding boxes, confidence scores, and class probabilities across the entire image.

This integrated approach makes YOLO extremely fast, as it processes the image globally and simultaneously predicts all detection components.

<a name="4-key-yolo-versions-and-advancements"></a>
## 4. Key YOLO Versions and Advancements

The YOLO framework has undergone continuous evolution, with each subsequent version introducing significant architectural and training improvements.

### YOLOv1 (2015)
The foundational version, as described above, introduced the concept of one-stage object detection. While revolutionary in speed, it struggled with detecting small objects and had difficulties generalizing to novel aspect ratios or configurations. It also suffered from relatively lower localization accuracy compared to two-stage detectors.

### YOLOv2 (YOLO9000) (2016)
YOLOv2 introduced several enhancements to improve both speed and accuracy. Key changes included:
*   **Batch Normalization:** Added to all convolutional layers for faster convergence and regularization.
*   **Anchor Boxes:** Adopted from Faster R-CNN, anchor boxes improved bounding box prediction by predicting offsets relative to pre-defined anchor box shapes. This allowed the network to learn to predict shapes better.
*   **Darknet-19:** A new, more powerful backbone network.
*   **Dimension Clusters:** Used k-means clustering on the training dataset's bounding boxes to automatically find good prior anchor box dimensions, leading to better starting points for predictions.
*   **Direct Location Prediction:** Predicted location offsets in a way that constrains them to fall within a grid cell, stabilizing training.
*   **Multi-scale Training:** The network was trained at various input image sizes, making it robust to different scales.
*   **YOLO9000:** An extension that could detect over 9000 object categories by jointly training on detection and classification datasets using a hierarchical tree structure.

### YOLOv3 (2018)
YOLOv3 further pushed the boundaries of accuracy while maintaining speed. Significant improvements included:
*   **Darknet-53:** An even deeper and more powerful backbone network, featuring **residual connections** for better feature propagation.
*   **Multi-scale Detection:** Predictions were made at three different scales (layers), allowing the model to detect objects of various sizes more effectively. Smaller objects were detected at finer-grained feature maps, while larger objects were detected at coarser maps.
*   **Logistic Classifiers:** Replaced softmax for class prediction, allowing for multi-label classification (e.g., an object could be both "person" and "woman").
*   **Increased Anchor Boxes:** Used 9 anchor boxes in total (3 for each scale).

### YOLOv4 (2020)
YOLOv4 focused on "bag of freebies" and "bag of specials" techniques to optimize training and inference without adding significant computational cost. It introduced:
*   **CSPDarknet53:** A **Cross Stage Partial Network** (CSP) backbone for more efficient feature learning.
*   **PANet (Path Aggregation Network):** For improved feature aggregation from different scales.
*   **Mish activation function:** Replaced Leaky ReLU for better gradient flow.
*   **Data Augmentation:** Extensive use of techniques like Mosaic data augmentation.
*   **Various Regularization Techniques:** Such as DropBlock and SAM.
*   **CIoU Loss:** Used for bounding box regression, leading to more accurate localization.

### YOLOv5 (2020)
Developed by Ultralytics, YOLOv5 offered a family of models (e.g., YOLOv5s, m, l, x) with different sizes and performance characteristics. While not an official paper, its highly optimized PyTorch implementation, ease of use, and strong performance led to its widespread adoption. Key aspects include:
*   **PyTorch Framework:** Full implementation in PyTorch, leveraging its ecosystem.
*   **Auto-Anchor:** Automatically determines optimal anchor boxes for custom datasets.
*   **Focus module:** An efficient way to downsample images.
*   **Simplified Architecture:** Often leaner and faster to train than YOLOv4 while achieving comparable accuracy.

### YOLOv6 (2022) & YOLOv7 (2022)
These versions continued the trend of optimizing the architecture for various devices and tasks, often focusing on balancing throughput and accuracy. YOLOv6, developed by Meituan, and YOLOv7, from a separate research group, both brought novel architectural components and training strategies that pushed the state-of-the-art in real-time detection further.

### YOLOv8 (2023)
The latest major release from Ultralytics, YOLOv8, is a cutting-edge, state-of-the-art (SOTA) model that offers improved performance across all sizes and a more user-friendly interface. It's an **anchor-free** model, simplifying the detection head and making it more efficient. Key features include:
*   **Anchor-Free Detection:** Directly predicts object centers and dimensions, simplifying the network and improving generalization.
*   **C2F (CSP-Darknet to Focus) module:** Enhances feature extraction.
*   **Decoupled Head:** Separates classification and detection heads for improved learning.
*   **New Loss Functions:** Incorporates advanced loss functions for better training stability and accuracy.
*   **Classification and Segmentation Models:** Extends the YOLO framework beyond just detection.

The continuous evolution of YOLO demonstrates its adaptability and the ongoing research efforts to achieve faster, more accurate, and more efficient real-time object detection systems.

<a name="5-advantages-limitations-and-applications"></a>
## 5. Advantages, Limitations, and Applications

YOLO's rise to prominence is largely due to its distinct advantages, though it also possesses inherent limitations. Its unique characteristics make it suitable for a wide array of real-world applications.

### Advantages
1.  **Speed and Real-Time Performance:** This is YOLO's most significant advantage. By treating object detection as a single regression problem, it achieves extremely high frame rates, making it suitable for real-time video processing, autonomous systems, and interactive applications.
2.  **Global Context:** Unlike region proposal-based methods that analyze patches independently, YOLO sees the entire image during training and testing. This allows it to encode contextual information about classes and their appearance, reducing false positives for background elements.
3.  **End-to-End Training:** The entire network is trained end-to-end, directly optimizing detection performance, which simplifies the training pipeline.
4.  **Generality:** YOLO models are highly generalizable. When trained on various detection tasks, they learn generalized representations of objects, making them robust when applied to new domains or unexpected inputs.
5.  **Simplicity:** Compared to multi-stage pipelines, YOLO's unified architecture is conceptually simpler and often easier to implement and deploy.

### Limitations
1.  **Localization Accuracy:** Early YOLO versions (YOLOv1) were known to have lower localization accuracy compared to two-stage detectors, especially for precisely placing bounding boxes. While subsequent versions have significantly improved this, it can still be a trade-off for extreme speed.
2.  **Small Object Detection:** YOLO's grid system means that each grid cell can only predict a limited number of objects. If multiple small objects appear within a single grid cell, YOLO struggles to detect all of them. This limitation has been mitigated in later versions through multi-scale predictions but remains a challenge.
3.  **Novel Aspect Ratios/Configurations:** Due to its reliance on learning from the training data, YOLO can struggle with objects that appear in unusual aspect ratios or novel configurations that are not well-represented in the training set.
4.  **Batch Processing Preference:** While optimized for speed, YOLO still benefits from batch processing. For truly single-image, ultra-low-latency scenarios, it might require specialized hardware or model compression.

### Applications
The unique blend of speed and accuracy makes YOLO invaluable across numerous domains:
*   **Autonomous Driving:** Detecting pedestrians, vehicles, traffic signs, and lanes in real-time.
*   **Robotics:** Enabling robots to perceive and interact with their environment, including object grasping and navigation.
*   **Surveillance and Security:** Monitoring crowds, identifying suspicious activities, and tracking individuals or objects.
*   **Retail Analytics:** Tracking customer behavior, monitoring shelf inventory, and analyzing product placement.
*   **Sports Analytics:** Tracking player movements, ball trajectories, and analyzing game strategies.
*   **Agriculture:** Monitoring crop health, detecting pests, and automating harvesting tasks.
*   **Medical Imaging:** Assisting in the detection of anomalies or features in medical scans (though often specialized models are preferred for critical applications).
*   **Augmented Reality (AR):** Real-time object recognition for overlaying digital information onto the real world.

The continuous improvements in YOLO's accuracy and efficiency ensure its continued relevance and expansion into new and challenging real-time vision tasks.

<a name="6-code-example"></a>
## 6. Code Example

This short Python snippet demonstrates how to use a pre-trained YOLOv8 model from the `ultralytics` library to perform object detection on an image. It showcases loading the model and processing its predictions.

```python
from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8n model (the 'n' stands for nano, a smaller, faster version)
# This will download the model weights if not already present.
model = YOLO('yolov8n.pt')

# 2. Perform inference on an example image.
# You can provide a local image path (e.g., 'path/to/your/image.jpg')
# or a URL to an image.
print("Performing inference on an example image...")
results = model('https://ultralytics.com/images/bus.jpg')

# 3. Process the results from the inference.
# The 'results' object contains detailed predictions including bounding boxes, classes, and confidence scores.
for r in results:
    print("\n--- Detection Results ---")
    # Iterate through each detected bounding box
    for box in r.boxes:
        # Get the detected class name using the model's names mapping
        class_name = model.names[int(box.cls[0])]
        # Get the bounding box coordinates (x1, y1, x2, y2)
        # .tolist() converts the tensor to a standard Python list for easy printing
        coords = box.xyxy[0].tolist()
        # Get the confidence score
        confidence = box.conf[0].item()

        print(f"Detected: {class_name}")
        print(f"  Coordinates: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f}]")
        print(f"  Confidence: {confidence:.2f}")

# The results object also contains methods to visualize detections,
# e.g., r.show() to display the image with bounding boxes.
# This example focuses on printing the raw detection data.
print("\nInference complete. The results object holds detailed detections.")

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion

YOLO (You Only Look Once) has undeniably revolutionized the field of real-time object detection. By framing the complex task of object localization and classification as a single regression problem, it bypassed the computational bottlenecks of earlier two-stage detectors, ushering in an era of truly real-time computer vision applications. From its initial inception, YOLO has continuously evolved through multiple iterations, each building upon its predecessors to enhance speed, accuracy, and efficiency.

The architectural advancements, such as the introduction of anchor boxes, multi-scale detection, more powerful backbones, and sophisticated training "bag of freebies," have consistently pushed the boundaries of what is achievable. While early versions faced limitations in detecting small objects and precise localization, modern YOLO variants have largely addressed these concerns, offering a highly robust and versatile solution for a myriad of industrial and research challenges.

YOLO's impact extends across diverse sectors, empowering innovations in autonomous systems, surveillance, robotics, and augmented reality. Its ongoing development, exemplified by the anchor-free approach and improved performance of YOLOv8, ensures its enduring relevance and positions it as a cornerstone technology in the ever-expanding landscape of artificial intelligence and computer vision.

<a name="8-references-and-further-reading"></a>
## 8. References and Further Reading

*   Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779-788.
*   Redmon, J., & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7263-7271.
*   Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv preprint arXiv:1804.02767*.
*   Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. *arXiv preprint arXiv:2004.10934*.
*   Jocher, G., et al. (2020). *Ultralytics YOLOv5*. GitHub. Available at: https://github.com/ultralytics/yolov5
*   Jocher, G., et al. (2023). *Ultralytics YOLOv8*. GitHub. Available at: https://github.com/ultralytics/ultralytics

---
<br>

<a name="türkçe-içerik"></a>
## YOLO (You Only Look Once): Gerçek Zamanlı Nesne Algılama

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Nesne Algılama Paradiglarının Evrimi](#2-nesne-algılama-paradiglarının-evrimi)
- [3. YOLO'nun Temel Prensipleri ve Mimarisi](#3-yolonun-temel-prensipleri-ve-mimarisi)
- [4. Başlıca YOLO Versiyonları ve Geliştirmeler](#4-başlıca-yolo-versiyonları-ve-geliştirmeler)
- [5. Avantajlar, Sınırlamalar ve Uygulamalar](#5-avantajlar-sınırlamalar-ve-uygulamalar)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)
- [8. Kaynaklar ve İleri Okuma](#8-kaynaklar-ve-ileri-okuma)

<a name="1-giriş"></a>
## 1. Giriş

Nesne algılama, bir görüntü veya video karesi içindeki ilgi çekici nesneleri tanımlamayı ve konumlarını belirlemeyi amaçlayan bilgisayar görüşünün temel bir görevidir. Bu, yalnızca hangi nesnelerin mevcut olduğunu sınıflandırmakla kalmaz, aynı zamanda tam konumları etrafına **sınırlayıcı kutular (bounding boxes)** çizmeyi de içerir. Alan, büyük ölçüde derin öğrenmenin ortaya çıkmasıyla dikkate değer ilerlemeler kaydetmiştir. En etkili ve yaygın olarak benimsenen yaklaşımlardan biri, gerçek zamanlı nesne algılamadaki olağanüstü hızı ve rekabetçi doğruluğu ile bilinen devrim niteliğindeki bir model ailesi olan **YOLO (You Only Look Once)**'dur.

2015 yılında Joseph Redmon ve arkadaşları tarafından tanıtılan YOLO, nesne algılamayı, tam görüntülerden tek geçişte doğrudan sınırlayıcı kutu koordinatlarını ve sınıf olasılıklarını tahmin eden tek bir regresyon problemi olarak yeniden çerçevelendirerek bir paradigma değişikliği sundu. Bu durum, bölge öneri ağlarına dayanan önceki çok aşamalı dedektörlerle keskin bir tezat oluşturuyordu. YOLO'nun "tek atışlık" yaklaşımı, gerçek zamanlı uygulamaları ilk kez uygulanabilir hale getirerek otonom sürüş, gözetim, robotik ve artırılmış gerçeklik gibi çeşitli alanlara entegrasyonunun önünü açtı. Bu belge, YOLO çerçevesinin temel prensiplerini, mimari evrimini, avantajlarını ve sınırlamalarını incelemektedir.

<a name="2-nesne-algılama-paradiglarının-evrimi"></a>
## 2. Nesne Algılama Paradiglarının Evrimi

YOLO'dan önce, nesne algılama büyük ölçüde iki aşamalı bir yaklaşım izliyordu. Erken dönem yöntemler, el yapımı özelliklerle (örn. SIFT, HOG) sınıflandırıcıları (örn. SVM'ler) birleştiriyordu. Derin öğrenme çağı, özellikle **Bölge tabanlı Evrişimsel Sinir Ağı (R-CNN)** ailesiyle önemli gelişmeler getirdi.

2014'te önerilen **R-CNN (CNN özelliklerine sahip Bölgeler)**, CNN'leri entegre ederek bir atılım gerçekleştirdi. Üç ana adımı içeriyordu:
1.  **Bölge Önerisi:** Seçici bir arama algoritması yaklaşık 2000 bölge önerisi (potansiyel nesne konumları) üretiyordu.
2.  **Özellik Çıkarma:** Her önerilen bölge sabit bir boyuta dönüştürülür ve özellik çıkarmak için önceden eğitilmiş bir CNN'den geçiriliyordu.
3.  **Sınıflandırma ve Sınırlayıcı Kutu Regresyonu:** Destek Vektör Makineleri (SVM'ler) özellikleri sınıflandırıyor ve ayrı bir regressor sınırlayıcı kutu koordinatlarını iyileştiriyordu.
Doğru olmasına rağmen, R-CNN, her bölgeyi bağımsız olarak işlemesi nedeniyle hesaplama açısından pahalı ve yavaştı.

**Fast R-CNN**, R-CNN'in bazı verimsizliklerini, tüm görüntüyü bir CNN ile bir kez işleyerek ve ardından bölge önerilerini özellik haritasına yansıtarak giderdi. Bu, gereksiz hesaplamaları azalttı. **Faster R-CNN** ise seçici arama algoritmasını bir **Bölge Öneri Ağı (RPN)** ile değiştirerek hızı daha da artırdı; RPN dedektör ile birlikte eğitiliyordu. Bu, öncekilerden önemli ölçüde daha hızlı ve daha doğru, uçtan uca, ancak yine de iki aşamalı bir algılama sistemi yarattı.

Bu gelişmelere rağmen, iki aşamalı dedektörler, sıralı yapıları nedeniyle doğasında var olan gecikmeden muzdaripti. Önce bölgeleri öneriyor, sonra sınıflandırıp iyileştiriyorlardı. Bu temel sınırlama, algılamanın tek bir regresyon problemi olarak formüle edildiği, tüm sınırlayıcı kutuları ve sınıf olasılıklarını eşzamanlı olarak tahmin eden **tek aşamalı dedektörlerin** geliştirilmesini teşvik etti. YOLO, bu yeni paradigmanın önde gelen temsilcisi olarak ortaya çıktı.

<a name="3-yolonun-temel-prensipleri-ve-mimarisi"></a>
## 3. YOLO'nun Temel Prensipleri ve Mimarisi

YOLO'nun temel yeniliği, nesne konumlandırmayı ve sınıflandırmayı evrişimsel bir sinir ağının tek bir geçişinde gerçekleştiren birleşik mimarisinde yatmaktadır. Bu "tek atışlık" yaklaşım, ayrı bir bölge öneri adımına ihtiyaç duymadan doğrudan sınırlayıcı kutuları ve sınıf olasılıklarını tahmin eder.

Temel prensipleri şu şekilde özetlenebilir:

### Izgara Sistemi
Giriş görüntüsü $S \times S$ boyutunda bir ızgaraya bölünür. Bir nesnenin merkezi bir ızgara hücresine düşerse, o hücre nesneyi algılamaktan sorumlu olur.

### Sınırlayıcı Kutu Tahmini
Her ızgara hücresi $B$ sınırlayıcı kutu tahmin eder. Her sınırlayıcı kutu için şunları tahmin eder:
*   **$x, y$ koordinatları:** Kutunun merkezinin ızgara hücresi sınırlarına göre konumu.
*   **$w, h$ boyutları:** Kutunun tam görüntü boyutlarına göre genişliği ve yüksekliği.
*   **Güven skoru:** Bu skor iki hususu yansıtır: kutunun bir nesne içerdiği olasılığı (nesnellik) ve tahmin edilen kutu ile herhangi bir gerçek kutu arasındaki **Kesit Üzeri Birleşim (IOU)**. Yüksek bir güven skoru, yüksek nesne olasılığına ve doğru konumlandırmaya işaret eder.

### Sınıf Olasılığı Tahmini
Her ızgara hücresi ayrıca $C$ **koşullu sınıf olasılığı** ($P(\text{Sınıf}_i | \text{Nesne})$) tahmin eder. Bunlar, bir nesnenin hücrede algılanması koşuluyla, nesnenin belirli bir sınıfa ait olma olasılıklarıdır. Bu olasılıklar koşulludur çünkü yalnızca bir nesne mevcutsa anlam ifade ederler.

### Nihai Algılama
Her sınırlayıcı kutu için, nihai sınıfa özgü güven skoru, sınırlayıcı kutu güveninin koşullu sınıf olasılıklarıyla çarpılmasıyla hesaplanır:
$$ \text{Güven} = P(\text{Nesne}) * \text{IOU}_{\text{tahmin}}^{\text{gerçek}} \times P(\text{Sınıf}_i | \text{Nesne}) = P(\text{Sınıf}_i) * \text{IOU}_{\text{tahmin}}^{\text{gerçek}} $$
Bu skor, hem o sınıfın kutuda görünme olasılığını hem de tahmin edilen kutunun nesneye ne kadar iyi oturduğunu temsil eder. Belirli bir eşiğin altındaki skorlara sahip kutular atılır.

### Maksimum Olmayan Bastırma (Non-Maximum Suppression - NMS)
YOLO genellikle aynı nesne için birden fazla çakışan sınırlayıcı kutu tahmin eder. Fazla algılamaları ortadan kaldırmak için **Maksimum Olmayan Bastırma (NMS)** uygulanır. Çakışan kutular arasından en yüksek güven skoruna sahip sınırlayıcı kutuyu seçer ve seçilen kutuyla yüksek IOU'ya sahip diğerlerini bastırır.

### Ağ Mimarisi (YOLOv1)
Orijinal YOLOv1 ağı genellikle 24 evrişim katmanından ve ardından 2 tam bağlı katmandan oluşuyordu. **Leaky ReLU** aktivasyonlarını ve son çıktı katmanı için **doğrusal bir aktivasyon fonksiyonu** kullanıyordu. Ağın çıktısı, $S \times S \times (B \times 5 + C)$ boyutunda bir tensördür ve tüm görüntüdeki sınırlayıcı kutular, güven skorları ve sınıf olasılıkları için tüm tahminleri kapsar.

Bu entegre yaklaşım, görüntüyü global olarak işlediği ve tüm algılama bileşenlerini eşzamanlı olarak tahmin ettiği için YOLO'yu son derece hızlı hale getirir.

<a name="4-başlıca-yolo-versiyonları-ve-geliştirmeler"></a>
## 4. Başlıca YOLO Versiyonları ve Geliştirmeler

YOLO çerçevesi sürekli bir evrim geçirmiş, sonraki her versiyon önemli mimari ve eğitim geliştirmeleri sunmuştur.

### YOLOv1 (2015)
Yukarıda açıklanan temel versiyon, tek aşamalı nesne algılama kavramını tanıttı. Hız açısından devrim niteliğinde olsa da, küçük nesneleri algılamakta zorlandı ve yeni en boy oranlarına veya konfigürasyonlara genelleme yapmakta güçlük çekti. Ayrıca, iki aşamalı dedektörlere kıyasla nispeten düşük konumlandırma doğruluğuna sahipti.

### YOLOv2 (YOLO9000) (2016)
YOLOv2, hem hızı hem de doğruluğu artırmak için çeşitli geliştirmeler tanıttı. Temel değişiklikler şunları içeriyordu:
*   **Toplu Normalizasyon (Batch Normalization):** Daha hızlı yakınsama ve düzenlileşme için tüm evrişim katmanlarına eklendi.
*   **Çapa Kutuları (Anchor Boxes):** Faster R-CNN'den uyarlanan çapa kutuları, önceden tanımlanmış çapa kutusu şekillerine göre ofsetleri tahmin ederek sınırlayıcı kutu tahminini iyileştirdi. Bu, ağın şekilleri daha iyi tahmin etmeyi öğrenmesini sağladı.
*   **Darknet-19:** Yeni, daha güçlü bir omurga ağı.
*   **Boyut Kümeleri (Dimension Clusters):** Eğitim veri kümesinin sınırlayıcı kutuları üzerinde k-ortalamalar kümelemesi kullanarak iyi başlangıç çapa kutusu boyutlarını otomatik olarak bularak, tahminler için daha iyi başlangıç noktaları sağladı.
*   **Doğrudan Konum Tahmini:** Konum ofsetlerini, bir ızgara hücresi içinde kalmalarını sağlayacak şekilde tahmin ederek eğitimi dengeledi.
*   **Çok Ölçekli Eğitim:** Ağ, çeşitli giriş görüntü boyutlarında eğitilerek farklı ölçeklere karşı sağlam hale getirildi.
*   **YOLO9000:** Hiyerarşik bir ağaç yapısı kullanarak algılama ve sınıflandırma veri kümeleri üzerinde ortaklaşa eğitim yaparak 9000'den fazla nesne kategorisini algılayabilen bir uzantı.

### YOLOv3 (2018)
YOLOv3, hızı korurken doğruluk sınırlarını daha da zorladı. Önemli geliştirmeler şunları içeriyordu:
*   **Darknet-53:** Daha iyi özellik yayılımı için **artık bağlantılar (residual connections)** içeren daha derin ve daha güçlü bir omurga ağı.
*   **Çok Ölçekli Algılama:** Tahminler üç farklı ölçekte (katmanda) yapıldı ve modelin çeşitli boyutlardaki nesneleri daha etkili bir şekilde algılamasını sağladı. Daha küçük nesneler daha ince taneli özellik haritalarında, daha büyük nesneler ise daha kaba haritalarda algılandı.
*   **Lojistik Sınıflandırıcılar:** Sınıf tahmini için softmax yerine lojistik sınıflandırıcılar kullanıldı ve çok etiketli sınıflandırmaya izin verildi (örn. bir nesne hem "kişi" hem de "kadın" olabilir).
*   **Artan Çapa Kutuları:** Toplamda 9 çapa kutusu kullanıldı (her ölçek için 3).

### YOLOv4 (2020)
YOLOv4, önemli bir hesaplama maliyeti eklemeden eğitimi ve çıkarımı optimize etmek için "çantada bedavalar" (bag of freebies) ve "çantada özel ürünler" (bag of specials) tekniklerine odaklandı. Şunları tanıttı:
*   **CSPDarknet53:** Daha verimli özellik öğrenimi için **Çapraz Aşama Parçalı Ağ (Cross Stage Partial Network - CSP)** omurgası.
*   **PANet (Yol Toplama Ağı - Path Aggregation Network):** Farklı ölçeklerden özellik toplamasını iyileştirmek için.
*   **Mish aktivasyon fonksiyonu:** Daha iyi gradyan akışı için Leaky ReLU'nun yerini aldı.
*   **Veri Çoğaltma (Data Augmentation):** Mozaik veri çoğaltma gibi tekniklerin yoğun kullanımı.
*   **Çeşitli Düzenlileşme Teknikleri:** DropBlock ve SAM gibi.
*   **CIoU Kaybı:** Sınırlayıcı kutu regresyonu için kullanılarak daha doğru konumlandırma sağladı.

### YOLOv5 (2020)
Ultralytics tarafından geliştirilen YOLOv5, farklı boyutlarda ve performans özelliklerine sahip bir model ailesi (örn. YOLOv5s, m, l, x) sundu. Resmi bir makalesi olmasa da, son derece optimize edilmiş PyTorch uygulaması, kullanım kolaylığı ve güçlü performansı yaygın olarak benimsenmesine yol açtı. Temel özellikler şunları içerir:
*   **PyTorch Çerçevesi:** PyTorch ekosisteminden yararlanan tam PyTorch uygulaması.
*   **Otomatik Çapa (Auto-Anchor):** Özel veri kümeleri için optimal çapa kutularını otomatik olarak belirler.
*   **Odak modülü (Focus module):** Görüntüleri örneklemek için verimli bir yol.
*   **Basitleştirilmiş Mimari:** Genellikle YOLOv4'ten daha yalın ve daha hızlı eğitilebilirken benzer doğruluk elde eder.

### YOLOv6 (2022) & YOLOv7 (2022)
Bu versiyonlar, çeşitli cihazlar ve görevler için mimariyi optimize etme eğilimini sürdürerek genellikle çıktı ve doğruluk dengesine odaklandı. Meituan tarafından geliştirilen YOLOv6 ve ayrı bir araştırma grubu tarafından geliştirilen YOLOv7, gerçek zamanlı algılamada en ileri teknolojiyi daha da ileri taşıyan yeni mimari bileşenler ve eğitim stratejileri getirdi.

### YOLOv8 (2023)
Ultralytics'in en son büyük sürümü olan YOLOv8, tüm boyutlarda iyileştirilmiş performans ve daha kullanıcı dostu bir arayüz sunan son teknoloji (SOTA) bir modeldir. Bu, algılama başlığını basitleştiren ve daha verimli hale getiren **çapa kutusuz (anchor-free)** bir modeldir. Temel özellikler şunları içerir:
*   **Çapa Kutusuz Algılama:** Nesne merkezlerini ve boyutlarını doğrudan tahmin ederek ağı basitleştirir ve genellemeyi iyileştirir.
*   **C2F (CSP-Darknet'ten Odak'a) modülü:** Özellik çıkarımını geliştirir.
*   **Ayrık Başlık (Decoupled Head):** Daha iyi öğrenme için sınıflandırma ve algılama başlıklarını ayırır.
*   **Yeni Kayıp Fonksiyonları:** Daha iyi eğitim kararlılığı ve doğruluğu için gelişmiş kayıp fonksiyonlarını içerir.
*   **Sınıflandırma ve Segmentasyon Modelleri:** YOLO çerçevesini sadece algılamanın ötesine genişletir.

YOLO'nun sürekli evrimi, uyarlanabilirliğini ve daha hızlı, daha doğru ve daha verimli gerçek zamanlı nesne algılama sistemleri elde etmek için devam eden araştırma çabalarını göstermektedir.

<a name="5-avantajlar-sınırlamalar-ve-uygulamalar"></a>
## 5. Avantajlar, Sınırlamalar ve Uygulamalar

YOLO'nun öne çıkışı büyük ölçüde kendine özgü avantajlarından kaynaklanırken, aynı zamanda doğasında var olan sınırlamalara da sahiptir. Benzersiz özellikleri, onu çok çeşitli gerçek dünya uygulamaları için uygun kılar.

### Avantajlar
1.  **Hız ve Gerçek Zamanlı Performans:** Bu, YOLO'nun en önemli avantajıdır. Nesne algılamayı tek bir regresyon problemi olarak ele alarak, son derece yüksek kare hızlarına ulaşır ve bu da onu gerçek zamanlı video işleme, otonom sistemler ve etkileşimli uygulamalar için uygun hale getirir.
2.  **Küresel Bağlam:** Yamaları bağımsız olarak analiz eden bölge öneri tabanlı yöntemlerin aksine, YOLO eğitim ve test sırasında görüntünün tamamını görür. Bu, sınıflar ve görünümleri hakkında bağlamsal bilgileri kodlamasına olanak tanır ve arka plan öğeleri için yanlış pozitifleri azaltır.
3.  **Uçtan Uca Eğitim:** Tüm ağ uçtan uca eğitilir ve algılama performansını doğrudan optimize ederek eğitim sürecini basitleştirir.
4.  **Genellenebilirlik:** YOLO modelleri yüksek düzeyde genellenebilir. Çeşitli algılama görevlerinde eğitildiğinde, nesnelerin genelleştirilmiş temsillerini öğrenirler, bu da onları yeni alanlara veya beklenmedik girdilere uygulandığında sağlam hale getirir.
5.  **Basitlik:** Çok aşamalı işlem hatlarına kıyasla, YOLO'nun birleşik mimarisi kavramsal olarak daha basittir ve uygulaması ve dağıtımı genellikle daha kolaydır.

### Sınırlamalar
1.  **Konumlandırma Doğruluğu:** Erken YOLO versiyonlarının (YOLOv1), özellikle sınırlayıcı kutuları hassas bir şekilde yerleştirme konusunda iki aşamalı dedektörlere kıyasla daha düşük konumlandırma doğruluğuna sahip olduğu biliniyordu. Sonraki versiyonlar bunu önemli ölçüde iyileştirmiş olsa da, aşırı hız için yine de bir ödünleşme olabilir.
2.  **Küçük Nesne Algılama:** YOLO'nun ızgara sistemi, her ızgara hücresinin yalnızca sınırlı sayıda nesne tahmin edebileceği anlamına gelir. Birden fazla küçük nesne tek bir ızgara hücresinde görünürse, YOLO hepsini algılamakta zorlanır. Bu sınırlama, sonraki versiyonlarda çok ölçekli tahminlerle hafifletilmiş olsa da bir zorluk olmaya devam etmektedir.
3.  **Yeni En Boy Oranları/Konfigürasyonlar:** Eğitim verilerinden öğrenmeye bağımlı olması nedeniyle, YOLO, eğitim kümesinde iyi temsil edilmeyen alışılmadık en boy oranlarına veya yeni konfigürasyonlara sahip nesnelerle zorlanabilir.
4.  **Toplu İşleme Tercihi:** Hız için optimize edilmiş olsa da, YOLO yine de toplu işlemeden faydalanır. Gerçekten tek görüntü, ultra düşük gecikmeli senaryolar için özel donanım veya model sıkıştırma gerektirebilir.

### Uygulamalar
YOLO'nun hız ve doğruluğun benzersiz karışımı, onu sayısız alanda paha biçilmez kılar:
*   **Otonom Sürüş:** Yayaları, araçları, trafik işaretlerini ve şeritleri gerçek zamanlı olarak algılama.
*   **Robotik:** Robotların çevrelerini algılamalarını ve onlarla etkileşim kurmalarını sağlama, nesne yakalama ve navigasyon dahil.
*   **Gözetim ve Güvenlik:** Kalabalıkları izleme, şüpheli faaliyetleri belirleme ve bireyleri veya nesneleri takip etme.
*   **Perakende Analizi:** Müşteri davranışını izleme, raf envanterini izleme ve ürün yerleşimini analiz etme.
*   **Spor Analizi:** Oyuncu hareketlerini, top yörüngelerini takip etme ve oyun stratejilerini analiz etme.
*   **Tarım:** Mahsul sağlığını izleme, zararlıları tespit etme ve hasat görevlerini otomatikleştirme.
*   **Tıbbi Görüntüleme:** Tıbbi taramalarda anormallikleri veya özellikleri tespit etmede yardımcı olma (ancak kritik uygulamalar için genellikle özel modeller tercih edilir).
*   **Artırılmış Gerçeklik (AR):** Dijital bilgileri gerçek dünyaya bindirmek için gerçek zamanlı nesne tanıma.

YOLO'nun doğruluğundaki ve verimliliğindeki sürekli iyileştirmeler, onun devam eden önemini ve yeni ve zorlu gerçek zamanlı görüş görevlerine genişlemesini sağlamaktadır.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği

Bu kısa Python kodu, `ultralytics` kütüphanesinden önceden eğitilmiş bir YOLOv8 modelini kullanarak bir görüntü üzerinde nesne algılama işlemini nasıl gerçekleştireceğinizi gösterir. Modelin yüklenmesini ve tahminlerinin işlenmesini sergiler.

```python
from ultralytics import YOLO

# 1. Önceden eğitilmiş bir YOLOv8n modelini yükleyin ('n' nano, yani daha küçük ve daha hızlı bir versiyonu temsil eder)
# Bu işlem, model ağırlıkları henüz mevcut değilse indirilecektir.
model = YOLO('yolov8n.pt')

# 2. Örnek bir görüntü üzerinde çıkarım (inference) yapın.
# Yerel bir görüntü yolu (örn. 'path/to/your/image.jpg') veya bir görüntü URL'si sağlayabilirsiniz.
print("Örnek bir görüntü üzerinde çıkarım gerçekleştiriliyor...")
results = model('https://ultralytics.com/images/bus.jpg')

# 3. Çıkarım sonuçlarını işleyin.
# 'results' nesnesi, sınırlayıcı kutular, sınıflar ve güven skorları dahil olmak üzere ayrıntılı tahminler içerir.
for r in results:
    print("\n--- Algılama Sonuçları ---")
    # Tespit edilen her sınırlayıcı kutu üzerinde döngü yapın
    for box in r.boxes:
        # Modelin isim haritasını kullanarak tespit edilen sınıfın adını alın
        class_name = model.names[int(box.cls[0])]
        # Sınırlayıcı kutu koordinatlarını (x1, y1, x2, y2) alın
        # .tolist(), tensörü kolay yazdırma için standart bir Python listesine dönüştürür
        coords = box.xyxy[0].tolist()
        # Güven skorunu alın
        confidence = box.conf[0].item()

        print(f"Tespit Edilen: {class_name}")
        print(f"  Koordinatlar: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f}]")
        print(f"  Güven: {confidence:.2f}")

# 'results' nesnesi ayrıca algılamaları görselleştirmek için yöntemler içerir,
# örneğin, sınırlayıcı kutularla görüntüyü göstermek için r.show().
# Bu örnek, ham algılama verilerini yazdırmaya odaklanmıştır.
print("\nÇıkarım tamamlandı. Sonuçlar nesnesi ayrıntılı algılamaları tutar.")

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç

YOLO (You Only Look Once), gerçek zamanlı nesne algılama alanında inkar edilemez bir devrim yaratmıştır. Nesne konumlandırma ve sınıflandırmanın karmaşık görevini tek bir regresyon problemi olarak çerçeveleyerek, önceki iki aşamalı dedektörlerin hesaplama darboğazlarını aşmış ve gerçekten gerçek zamanlı bilgisayar görüşü uygulamaları çağını başlatmıştır. İlk ortaya çıkışından itibaren, YOLO, hızı, doğruluğu ve verimliliği artırmak için her biri seleflerinin üzerine inşa edilmiş birden fazla iterasyonla sürekli olarak evrim geçirmiştir.

Çapa kutularının tanıtımı, çok ölçekli algılama, daha güçlü omurgalar ve sofistike eğitim "bedava çantaları" gibi mimari gelişmeler, başarılabilir olanın sınırlarını sürekli olarak zorlamıştır. Erken versiyonlar küçük nesneleri algılama ve hassas konumlandırma konusunda sınırlamalarla karşılaşsa da, modern YOLO varyantları bu endişeleri büyük ölçüde ele almış, sayısız endüstriyel ve araştırma zorluğu için son derece sağlam ve çok yönlü bir çözüm sunmuştur.

YOLO'nun etkisi, otonom sistemler, gözetim, robotik ve artırılmış gerçeklik alanlarındaki yenilikleri güçlendirerek çeşitli sektörlere yayılmaktadır. YOLOv8'in çapa kutusuz yaklaşımı ve geliştirilmiş performansı ile örneklenen devam eden gelişimi, onun kalıcı önemini garanti etmekte ve yapay zeka ve bilgisayar görüşünün sürekli genişleyen ortamında onu bir köşe taşı teknolojisi olarak konumlandırmaktadır.

<a name="8-kaynaklar-ve-ileri-okuma"></a>
## 8. Kaynaklar ve İleri Okuma

*   Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779-788.
*   Redmon, J., & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7263-7271.
*   Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv preprint arXiv:1804.02767*.
*   Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. *arXiv preprint arXiv:2004.10934*.
*   Jocher, G., et al. (2020). *Ultralytics YOLOv5*. GitHub. Erişim adresi: https://github.com/ultralytics/yolov5
*   Jocher, G., et al. (2023). *Ultralytics YOLOv8*. GitHub. Erişim adresi: https://github.com/ultralytics/ultralytics



