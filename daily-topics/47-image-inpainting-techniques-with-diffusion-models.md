# Image Inpainting Techniques with Diffusion Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Diffusion Models](#2-understanding-diffusion-models)
    - [2.1 Forward Diffusion Process](#21-forward-diffusion-process)
    - [2.2 Reverse Diffusion Process](#22-reverse-diffusion-process)
- [3. Image Inpainting with Diffusion Models](#3-image-inpainting-with-diffusion-models)
    - [3.1 Core Methodology: Masked Conditional Diffusion](#31-core-methodology-masked-conditional-diffusion)
    - [3.2 Latent Diffusion for Inpainting](#32-latent-diffusion-for-inpainting)
    - [3.3 Advantages and Challenges](#33-advantages-and-challenges)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
**Image inpainting**, a fundamental task in computer vision and digital image processing, refers to the process of filling in missing or corrupted regions of an image in a way that is visually plausible and consistent with the surrounding content. Traditionally, this problem has been addressed using methods ranging from simple exemplar-based approaches to more sophisticated patch-based algorithms and, more recently, deep learning techniques such as Generative Adversarial Networks (GANs). While these methods have achieved remarkable success, they often struggle with generating diverse and semantically coherent content, particularly for large missing regions or complex scenes.

The advent of **Diffusion Models (DMs)** has ushered in a new era for generative AI, demonstrating unparalleled capabilities in image synthesis, style transfer, and super-resolution. Their probabilistic framework and iterative denoising process allow for the generation of high-fidelity, diverse samples that often surpass the quality of GANs. This document delves into the application of diffusion models for image inpainting, exploring the underlying principles, key methodologies, and practical considerations that make them a powerful tool for this challenging task. We will discuss how diffusion models leverage their generative power to infer and synthesize missing pixels, thereby restoring damaged images or enabling creative content generation.

## 2. Understanding Diffusion Models
Diffusion Models are a class of generative models that learn to reverse a gradual noising process. They operate by progressively adding Gaussian noise to an image until it becomes pure noise, and then learning to reverse this process to generate new data from noise.

### 2.1 Forward Diffusion Process
The **forward diffusion process**, also known as the noising process, gradually transforms an original data sample (e.g., an image $x_0$) into a noisy version ($x_t$) over a series of $T$ time steps. At each step $t$, a small amount of Gaussian noise is added to $x_{t-1}$ to produce $x_t$. This process is typically defined by a fixed Markov chain:

$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$

where $\beta_t$ is a small variance schedule. An important property of this process is that $x_t$ can be directly sampled from $x_0$ at any time step $t$:

$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. This allows sampling $x_t$ without iterating through all intermediate steps, simplifying training.

### 2.2 Reverse Diffusion Process
The core of a diffusion model lies in learning the **reverse diffusion process**, which aims to reverse the noise addition and restore the original data. This reverse process is also a Markov chain, starting from pure Gaussian noise ($x_T \sim \mathcal{N}(0, \mathbf{I})$) and iteratively denoising it to produce a sample from the data distribution ($x_0$). However, the true reverse conditional probability $p(x_{t-1} | x_t)$ is intractable.

Diffusion models approximate this intractable distribution using a neural network, typically a **U-Net architecture**, denoted as $\theta$. The network learns to predict the noise added at each step $t$, or more commonly, the mean of the denoising distribution. The predicted mean $\mu_{\theta}(x_t, t)$ and variance $\Sigma_{\theta}(x_t, t)$ allow sampling $x_{t-1}$ from $x_t$:

$p_{\theta}(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t))$

The network is trained by minimizing a loss function that encourages it to predict the true noise added to $x_0$ to get $x_t$. The **denoising diffusion probabilistic models (DDPMs)** framework, for instance, focuses on learning to predict the noise $\epsilon$ from $x_t$ and $t$.

## 3. Image Inpainting with Diffusion Models
Applying diffusion models to image inpainting primarily involves modifying the reverse diffusion process to condition the generation on the known (unmasked) regions of the image.

### 3.1 Core Methodology: Masked Conditional Diffusion
The most direct approach to inpainting with diffusion models involves performing **conditional sampling** during the reverse diffusion process. Given an image $x_0$ with a missing region defined by a binary mask $M$ (where $M_{ij}=1$ for known pixels and $M_{ij}=0$ for unknown pixels), the goal is to generate the unknown pixels while preserving the known ones.

During each step $t$ of the reverse sampling process, when sampling $x_{t-1}$ from $x_t$, the strategy is to combine the neural network's prediction for the entire image with the known original pixels. Specifically, after the network predicts the denoised version of $x_t$ (let's call it $\hat{x}_0$), the known regions of $\hat{x}_0$ are replaced with the corresponding known regions from the original noisy image $x_t$. This can be formulated as:

1.  **Generate a preliminary denoised image:** Using the U-Net, predict the noise or directly predict $\hat{x}_0$ from $x_t$.
2.  **Combine with known content:** The known pixels are "pushed back" into the prediction. Let $x_{0, \text{known}}$ be the original known pixels. The combined image $x'_{0}$ would be:
    $x'_{0} = M \odot x_{0, \text{known}} + (1-M) \odot \hat{x}_0$
    where $\odot$ denotes element-wise multiplication.
3.  **Resample noisy image:** From this combined $x'_{0}$, a new noisy $x_t$ is generated using the forward diffusion process (e.g., $q(x_t | x'_{0})$).
4.  **Update $x_t$ for next step:** The noisy $x_t$ is then updated by combining the known parts of the initial $x_t$ (which were maintained throughout the process) with the newly sampled noisy unknown parts. This ensures the known regions remain anchored to their original values while the unknown regions are filled. This is essentially creating a new $x_t$ that is consistent with both the model's prediction and the original known data.

This iterative process ensures that the generated content for the masked region is semantically consistent with the existing image data, as the diffusion model constantly adapts its generation based on the fixed boundary conditions provided by the mask.

### 3.2 Latent Diffusion for Inpainting
**Latent Diffusion Models (LDMs)**, such as Stable Diffusion, significantly improve the efficiency of diffusion models by performing the diffusion process in a compressed latent space rather than directly in pixel space. This approach is particularly beneficial for high-resolution images.

For inpainting with LDMs:
1.  The input image $x_0$ and its mask $M$ are first encoded into a lower-dimensional latent representation $z_0$ using an encoder network (e.g., a VAE encoder). The mask might also be compressed or propagated to the latent space.
2.  The forward and reverse diffusion processes then occur entirely within this latent space.
3.  During the reverse sampling, the conditioning on known pixels happens in the latent space. The encoder's output for the known image regions can be combined with the denoised latent representation generated by the U-Net.
4.  Finally, the denoised latent $z_0$ is passed through a decoder network (e.g., a VAE decoder) to reconstruct the full inpainted image in pixel space.

LDMs often utilize **classifier-free guidance**, a technique that combines an unconditional score estimate (generating without any specific input) with a conditional score estimate (generating based on the input image and mask). This guidance boosts the adherence of the generated content to the conditioning, leading to higher quality and more contextually relevant inpainted results.

### 3.3 Advantages and Challenges
**Advantages:**
*   **High Quality and Realism:** Diffusion models excel at generating photo-realistic and high-fidelity images, often outperforming GANs in perceptual quality.
*   **Diversity of Outputs:** Their probabilistic nature allows for generating multiple plausible completions for the same masked region, offering creative flexibility.
*   **Semantic Understanding:** By learning complex data distributions, DMs can fill in large, semantically challenging gaps in a coherent manner.
*   **Robustness:** The iterative denoising process can be robust to various types of masks and missing data patterns.

**Challenges:**
*   **Computational Cost:** The iterative nature of diffusion sampling, especially in pixel space, can be computationally expensive and slow compared to single-pass generative models. Latent diffusion helps mitigate this but still requires multiple steps.
*   **Memory Footprint:** Training and inference for very high-resolution images can demand significant GPU memory.
*   **Consistency:** While generally good, ensuring perfect local and global consistency, especially for extremely large masks or intricate textures, remains an active research area.
*   **Training Data:** High-quality and diverse datasets are crucial for training effective diffusion models for inpainting.

## 4. Code Example
This conceptual Python snippet illustrates how one might prepare an image and mask for inpainting, and conceptualizes the iterative denoising process. It simplifies the actual diffusion model's U-Net and noise prediction for clarity.

```python
import numpy as np
from PIL import Image

def load_image(image_path):
    """Loads an image and converts it to a NumPy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img) / 255.0  # Normalize to [0, 1]

def create_mask(image_shape, mask_center=(100, 100), mask_size=50):
    """Creates a simple square mask."""
    mask = np.ones(image_shape[:2], dtype=bool) # True for known (unmasked)
    y_start, x_start = mask_center[0] - mask_size // 2, mask_center[1] - mask_size // 2
    y_end, x_end = y_start + mask_size, x_start + mask_size
    mask[y_start:y_end, x_start:x_end] = False # False for unknown (masked)
    return mask

def conceptual_inpainting_step(noisy_image_t, original_known_region, mask, denoiser_model, t_step):
    """
    Conceptual single step of inpainting during reverse diffusion.
    In a real DM, 'denoiser_model' would be a U-Net predicting noise or original image.
    """
    # 1. Model predicts a denoised version of the noisy image
    # For illustration, let's just use the noisy image itself for the unknown part
    # In a real model, denoiser_model(noisy_image_t, t_step) would give a sophisticated prediction
    predicted_x0 = denoiser_model(noisy_image_t, t_step) # This is a placeholder for actual U-Net prediction

    # 2. Combine with the original known region (apply mask)
    # The known parts are taken from the original image (or its noisy version at step t)
    # The unknown parts are taken from the model's prediction
    combined_x0_estimation = np.copy(predicted_x0)
    combined_x0_estimation[mask] = original_known_region[mask]

    # 3. Simulate adding noise back to combine_x0_estimation to get next_noisy_image_t_minus_1
    # This is a simplification; actual DMs have a precise way to do this.
    # Here, we just blend it for demonstration.
    # In a real DM, one would sample x_{t-1} using p(x_{t-1}|x_t) where x_t is conditioned.
    next_noisy_image_t_minus_1 = combined_x0_estimation + np.random.normal(0, 0.01, noisy_image_t.shape) # A bit of noise

    # In actual inpainting, the known regions from original_known_region are implicitly
    # or explicitly re-inserted or conditioned upon at each reverse step.
    return next_noisy_image_t_minus_1

# Dummy denoiser model for illustration
def dummy_denoiser(image, t):
    """A dummy denoiser that just returns the input image for simplicity."""
    return image # In reality, this would be a trained U-Net that predicts noise or x_0

if __name__ == "__main__":
    # This part would typically be part of a larger diffusion inference loop.
    # Simulate a noisy image at time t and the original image's known regions
    # (replace with actual image loading and mask creation for real use)

    # Example: create a dummy image and mask
    dummy_img_shape = (128, 128, 3)
    original_image = np.zeros(dummy_img_shape)
    original_image[30:70, 30:70] = [1.0, 0.0, 0.0] # Red square
    original_image[50:90, 50:90] = [0.0, 1.0, 0.0] # Green square

    # Create a mask covering part of the green square
    mask = create_mask(original_image.shape, mask_center=(60, 60), mask_size=30)
    
    # Initialize a noisy image (e.g., pure noise at T)
    noisy_image_at_T = np.random.rand(*dummy_img_shape) 
    
    # For illustration, let's consider a single inpainting step
    # The 'original_known_region' would be derived from the actual input image
    # and passed through a forward process to match the noise level of noisy_image_at_T
    
    # A simplified 'original_known_region' is just the original image where the mask is True.
    # In a real setup, this would be the known parts of the *noisy* image at step t.
    current_noisy_image = np.copy(noisy_image_at_T)
    
    # Apply the mask to 'current_noisy_image' to simulate the input to the denoiser
    # For initial steps, the unknown region is noise, known region is noisy version of original known.
    
    # This loop conceptually represents the reverse diffusion steps (e.g., 1000 steps down to 0)
    num_inference_steps = 5
    print(f"Starting conceptual inpainting simulation for {num_inference_steps} steps...")
    for t_step in range(num_inference_steps, 0, -1):
        print(f"Processing step: {t_step}")
        
        # Here, original_known_region needs to be the known part of the original image,
        # appropriately scaled and noised to be compatible with current_noisy_image.
        # For this conceptual example, let's use the actual original image's known part.
        
        # A more accurate representation would involve:
        # 1. Extract known part from original_image.
        # 2. Add noise to this known part to match the noise level of current_noisy_image.
        # This creates 'original_known_region_at_t'.
        
        # Simplification: we directly use original_image's known part for blending
        # In a real diffusion model, this 'original_known_region' would be derived carefully
        # to ensure consistency with the current noise level 't_step'.
        
        current_noisy_image = conceptual_inpainting_step(
            current_noisy_image, original_image, mask, dummy_denoiser, t_step
        )
        # Visually inspect or save current_noisy_image if desired
        # For simplicity, we just print a message here.

    print("Conceptual inpainting simulation finished. Final image (highly simplified):")
    # To see the effect:
    # final_inpainted_image = Image.fromarray((current_noisy_image * 255).astype(np.uint8))
    # final_inpainted_image.save("conceptual_inpainted_result.png")
    # print("Result saved to conceptual_inpainted_result.png")


(End of code example section)
```
## 5. Conclusion
Diffusion Models have emerged as a transformative technology for image generation, and their adaptation to the task of image inpainting represents a significant leap forward. By leveraging their ability to learn complex data distributions and perform iterative denoising, DMs can generate highly realistic, semantically coherent, and diverse completions for missing image regions. While challenges related to computational cost and perfect consistency in all scenarios persist, techniques like Latent Diffusion and classifier-free guidance have substantially improved efficiency and quality. As research continues to advance, diffusion models are poised to become the cornerstone of next-generation image editing and restoration tools, offering unprecedented control and creative freedom in visual content manipulation.

---
<br>

<a name="türkçe-içerik"></a>
## Difüzyon Modelleri ile Görüntü Tamamlama Teknikleri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Difüzyon Modellerini Anlamak](#2-difüzyon-modellerini-anlamak)
    - [2.1 İleri Difüzyon Süreci](#21-ileri-difüzyon-süreci)
    - [2.2 Ters Difüzyon Süreci](#22-ters-difüzyon-süreci)
- [3. Difüzyon Modelleri ile Görüntü Tamamlama](#3-difüzyon-modelleri-ile-görüntü-tamamlama)
    - [3.1 Temel Metodoloji: Maskelenmiş Koşullu Difüzyon](#31-temel-metodoloji-maskelenmiş-koşullu-difüzyon)
    - [3.2 Latent Difüzyon ile Görüntü Tamamlama](#32-latent-difüzyon-ile-görüntü-tamamlama)
    - [3.3 Avantajlar ve Zorluklar](#33-avantajlar-ve-zorluklar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
Bilgisayar görüşü ve dijital görüntü işlemede temel bir görev olan **görüntü tamamlama (image inpainting)**, bir görüntüdeki eksik veya bozuk bölgelerin, çevresindeki içerikle görsel olarak makul ve tutarlı bir şekilde doldurulması sürecini ifade eder. Geleneksel olarak, bu problem basit örnek tabanlı yaklaşımlardan daha sofistike yama tabanlı algoritmalara ve son zamanlarda Üretken Çekişmeli Ağlar (GAN'lar) gibi derin öğrenme tekniklerine kadar çeşitli yöntemlerle ele alınmıştır. Bu yöntemler kayda değer başarılar elde etmiş olsa da, özellikle büyük eksik bölgeler veya karmaşık sahneler için çeşitli ve anlamsal olarak tutarlı içerik üretmede genellikle zorlanırlar.

**Difüzyon Modellerinin (DM'ler)** ortaya çıkışı, üretken yapay zeka için yeni bir çağ başlatmış ve görüntü sentezi, stil transferi ve süper çözünürlük alanlarında eşi benzeri görülmemiş yetenekler sergilemiştir. Olasılıksal çerçeveleri ve yinelemeli gürültü giderme süreçleri, genellikle GAN'ların kalitesini aşan yüksek doğrulukta, çeşitli örneklerin üretilmesine olanak tanır. Bu belge, difüzyon modellerinin görüntü tamamlama için uygulanmasını, bu zorlu görev için onları güçlü bir araç haline getiren temel ilkeleri, ana metodolojileri ve pratik hususları incelemektedir. Difüzyon modellerinin, eksik pikselleri çıkarıp sentezlemek için üretken güçlerini nasıl kullandığını ve böylece hasarlı görüntüleri onardığını veya yaratıcı içerik üretilmesini sağladığını tartışacağız.

## 2. Difüzyon Modellerini Anlamak
Difüzyon Modelleri, kademeli bir gürültüleme sürecini tersine çevirmeyi öğrenen bir üretken model sınıfıdır. Bir görüntüye (veya genel olarak veriye) kademeli olarak Gauss gürültüsü ekleyerek onu saf gürültüye dönüştürürler ve ardından bu süreci tersine çevirerek gürültüden yeni veri üretmeyi öğrenirler.

### 2.1 İleri Difüzyon Süreci
**İleri difüzyon süreci**, diğer adıyla gürültüleme süreci, orijinal bir veri örneğini (örn. bir görüntü $x_0$) $T$ zaman adımı boyunca kademeli olarak gürültülü bir versiyona ($x_t$) dönüştürür. Her $t$ adımında, $x_{t-1}$'e az miktarda Gauss gürültüsü eklenerek $x_t$ üretilir. Bu süreç genellikle sabit bir Markov zinciri ile tanımlanır:

$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$

Burada $\beta_t$ küçük bir varyans programıdır. Bu sürecin önemli bir özelliği, herhangi bir $t$ zaman adımında $x_t$'nin $x_0$'dan doğrudan örneklenmesidir:

$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$

Burada $\alpha_t = 1 - \beta_t$ ve $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Bu, tüm ara adımları yinelemeden $x_t$'yi örneklemeyi mümkün kılar ve eğitimi basitleştirir.

### 2.2 Ters Difüzyon Süreci
Bir difüzyon modelinin özü, gürültü eklemeyi tersine çevirmeyi ve orijinal verileri geri yüklemeyi amaçlayan **ters difüzyon sürecini** öğrenmekte yatar. Bu ters süreç de bir Markov zinciridir; saf Gauss gürültüsünden ($x_T \sim \mathcal{N}(0, \mathbf{I})$) başlayarak, verilerin dağılımından bir örnek ($x_0$) üretmek için kademeli olarak gürültüyü giderir. Ancak, gerçek ters koşullu olasılık $p(x_{t-1} | x_t)$ hesaplanamazdır.

Difüzyon modelleri, bu hesaplanamayan dağılımı bir sinir ağı, tipik olarak bir **U-Net mimarisi** kullanarak yaklaştırır ve bunu $\theta$ ile gösteririz. Ağ, her $t$ adımında eklenen gürültüyü veya daha yaygın olarak gürültü giderme dağılımının ortalamasını tahmin etmeyi öğrenir. Tahmin edilen ortalama $\mu_{\theta}(x_t, t)$ ve varyans $\Sigma_{\theta}(x_t, t)$, $x_t$'den $x_{t-1}$'i örneklemeye izin verir:

$p_{\theta}(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t))$

Ağ, $x_0$'a $x_t$'yi elde etmek için eklenen gerçek gürültüyü $\epsilon$ tahmin etmesini teşvik eden bir kayıp fonksiyonunu minimize ederek eğitilir. Örneğin, **gürültü giderici difüzyon olasılıksal modelleri (DDPM'ler)** çerçevesi, $x_t$ ve $t$'den gürültü $\epsilon$ tahmin etmeyi öğrenmeye odaklanır.

## 3. Difüzyon Modelleri ile Görüntü Tamamlama
Difüzyon modellerini görüntü tamamlamaya uygulamak, esasen ters difüzyon sürecini görüntünün bilinen (maskelenmemiş) bölgelerine göre koşullandırmayı içerir.

### 3.1 Temel Metodoloji: Maskelenmiş Koşullu Difüzyon
Difüzyon modelleriyle tamamlama için en doğrudan yaklaşım, ters difüzyon süreci sırasında **koşullu örnekleme** yapmaktır. İkili bir maske $M$ ile tanımlanmış eksik bir bölgeye sahip bir görüntü $x_0$ verildiğinde (burada $M_{ij}=1$ bilinen pikseller ve $M_{ij}=0$ bilinmeyen pikseller içindir), amaç, bilinen pikselleri korurken bilinmeyen pikselleri üretmektir.

Ters örnekleme sürecinin her $t$ adımında, $x_t$'den $x_{t-1}$ örneklenirken, strateji, sinir ağının tüm görüntü için tahminini bilinen orijinal piksellerle birleştirmektir. Özellikle, ağ $x_t$'nin gürültüsü giderilmiş versiyonunu ($\hat{x}_0$ diyelim) tahmin ettikten sonra, $\hat{x}_0$'ın bilinen bölgeleri, orijinal gürültülü görüntü $x_t$'den gelen karşılık gelen bilinen bölgelerle değiştirilir. Bu şu şekilde formüle edilebilir:

1.  **Ön gürültüsü giderilmiş bir görüntü oluşturun:** U-Net kullanarak gürültüyü tahmin edin veya $\hat{x}_0$'ı doğrudan $x_t$'den tahmin edin.
2.  **Bilinen içerikle birleştirin:** Bilinen pikseller tahmine "geri itilir". $x_{0, \text{bilinen}}$ orijinal bilinen pikseller olsun. Birleştirilmiş görüntü $x'_{0}$ şöyle olacaktır:
    $x'_{0} = M \odot x_{0, \text{bilinen}} + (1-M) \odot \hat{x}_0$
    Burada $\odot$ eleman bazında çarpımı ifade eder.
3.  **Gürültülü görüntüyü yeniden örnekle:** Bu birleştirilmiş $x'_{0}$'dan, ileri difüzyon süreci kullanılarak yeni bir gürültülü $x_t$ oluşturulur (örn. $q(x_t | x'_{0})$).
4.  **Bir sonraki adım için $x_t$'yi güncelleyin:** Gürültülü $x_t$, başlangıçtaki $x_t$'nin bilinen kısımları (süreç boyunca korunan) ile yeni örneklenmiş gürültülü bilinmeyen kısımları birleştirilerek güncellenir. Bu, bilinen bölgelerin orijinal değerlerine bağlı kalmasını sağlarken, bilinmeyen bölgelerin doldurulmasını sağlar. Bu aslında hem modelin tahminiyle hem de orijinal bilinen verilerle tutarlı yeni bir $x_t$ oluşturmaktır.

Bu yinelemeli süreç, maskelenen bölge için üretilen içeriğin, maske tarafından sağlanan sabit sınır koşullarına göre difüzyon modelinin üretimini sürekli olarak uyarlaması nedeniyle mevcut görüntü verileriyle anlamsal olarak tutarlı olmasını sağlar.

### 3.2 Latent Difüzyon ile Görüntü Tamamlama
**Latent Difüzyon Modelleri (LDM'ler)**, örneğin Stable Diffusion, difüzyon sürecini doğrudan piksel uzayında değil, sıkıştırılmış bir latent uzayda gerçekleştirerek difüzyon modellerinin verimliliğini önemli ölçüde artırır. Bu yaklaşım, özellikle yüksek çözünürlüklü görüntüler için faydalıdır.

LDM'ler ile tamamlama için:
1.  Girdi görüntüsü $x_0$ ve maskesi $M$ öncelikle bir kodlayıcı ağı (örn. bir VAE kodlayıcı) kullanılarak daha düşük boyutlu bir latent gösterim $z_0$'a kodlanır. Maske de sıkıştırılabilir veya latent uzaya aktarılabilir.
2.  İleri ve geri difüzyon süreçleri daha sonra tamamen bu latent uzayda gerçekleşir.
3.  Ters örnekleme sırasında, bilinen pikseller üzerindeki koşullandırma latent uzayında gerçekleşir. Kodlayıcının bilinen görüntü bölgeleri için çıktısı, U-Net tarafından üretilen gürültüsü giderilmiş latent gösterimle birleştirilebilir.
4.  Son olarak, gürültüsü giderilmiş latent $z_0$, tam doldurulmuş görüntüyü piksel uzayında yeniden yapılandırmak için bir kod çözücü ağı (örn. bir VAE kod çözücü) aracılığıyla geçirilir.

LDM'ler genellikle, koşulsuz bir skor tahmini (belirli bir girdi olmadan üretme) ile koşullu bir skor tahminini (girdi görüntüsüne ve maskeye göre üretme) birleştiren bir teknik olan **sınıflandırıcısız yönlendirme (classifier-free guidance)** kullanır. Bu yönlendirme, üretilen içeriğin koşullandırmaya bağlılığını artırarak daha yüksek kaliteli ve bağlamsal olarak daha alakalı tamamlanmış sonuçlar elde edilmesini sağlar.

### 3.3 Avantajlar ve Zorluklar
**Avantajlar:**
*   **Yüksek Kalite ve Gerçekçilik:** Difüzyon modelleri, foto-gerçekçi ve yüksek doğrulukta görüntüler üretmede mükemmeldir ve algısal kalitede genellikle GAN'ları geride bırakır.
*   **Çeşitli Çıktılar:** Olasılıksal doğaları, aynı maskelenmiş bölge için birden fazla olası tamamlama üretmeye olanak tanır ve yaratıcı esneklik sunar.
*   **Anlamsal Anlama:** Karmaşık veri dağılımlarını öğrenerek, DM'ler büyük, anlamsal olarak zorlayıcı boşlukları tutarlı bir şekilde doldurabilir.
*   **Sağlamlık:** Yinelemeli gürültü giderme süreci, çeşitli maske türlerine ve eksik veri desenlerine karşı sağlam olabilir.

**Zorluklar:**
*   **Hesaplama Maliyeti:** Özellikle piksel uzayında difüzyon örneklemesinin yinelemeli doğası, tek geçişli üretken modellere kıyasla hesaplama açısından pahalı ve yavaş olabilir. Latent difüzyon bunu hafifletmeye yardımcı olur ancak yine de birden fazla adım gerektirir.
*   **Bellek Ayak İzi:** Çok yüksek çözünürlüklü görüntüler için eğitim ve çıkarım, önemli GPU belleği gerektirebilir.
*   **Tutarlılık:** Genellikle iyi olsa da, özellikle son derece büyük maskeler veya karmaşık dokular için mükemmel yerel ve küresel tutarlılığı sağlamak aktif bir araştırma alanı olmaya devam etmektedir.
*   **Eğitim Verisi:** Tamamlama için etkili difüzyon modellerini eğitmek için yüksek kaliteli ve çeşitli veri kümeleri çok önemlidir.

## 4. Kod Örneği
Bu kavramsal Python kodu, tamamlama için bir görüntüyü ve maskeyi nasıl hazırlayacağınızı gösterir ve yinelemeli gürültü giderme sürecini kavramsallaştırır. Gerçek difüzyon modelinin U-Net ve gürültü tahminini netlik için basitleştirir.

```python
import numpy as np
from PIL import Image

def load_image(image_path):
    """Bir görüntüyü yükler ve bir NumPy dizisine dönüştürür."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img) / 255.0  # [0, 1] aralığına normalleştirir

def create_mask(image_shape, mask_center=(100, 100), mask_size=50):
    """Basit bir kare maske oluşturur."""
    mask = np.ones(image_shape[:2], dtype=bool) # Bilinen (maskelenmemiş) pikseller için True
    y_start, x_start = mask_center[0] - mask_size // 2, mask_center[1] - mask_size // 2
    y_end, x_end = y_start + mask_size, x_start + mask_size
    mask[y_start:y_end, x_start:x_end] = False # Bilinmeyen (maskelenmiş) pikseller için False
    return mask

def conceptual_inpainting_step(noisy_image_t, original_known_region, mask, denoiser_model, t_step):
    """
    Ters difüzyon sırasında tamamlamanın kavramsal tek adımı.
    Gerçek bir DM'de, 'denoiser_model' gürültü veya orijinal görüntüyü tahmin eden bir U-Net olurdu.
    """
    # 1. Model, gürültülü görüntünün gürültüsü giderilmiş bir versiyonunu tahmin eder
    # Gösterim için, bilinmeyen kısım için sadece gürültülü görüntüyü kullanalım
    # Gerçek bir modelde, denoiser_model(noisy_image_t, t_step) sofistike bir tahmin verirdi
    predicted_x0 = denoiser_model(noisy_image_t, t_step) # Bu, gerçek U-Net tahmini için bir yer tutucudur

    # 2. Orijinal bilinen bölge ile birleştirin (maskeyi uygulayın)
    # Bilinen kısımlar orijinal görüntüden (veya t adımındaki gürültülü versiyonundan) alınır
    # Bilinmeyen kısımlar modelin tahmininden alınır
    combined_x0_estimation = np.copy(predicted_x0)
    combined_x0_estimation[mask] = original_known_region[mask]

    # 3. next_noisy_image_t_minus_1 elde etmek için combine_x0_estimation'a geri gürültü eklemeyi simüle edin
    # Bu bir basitleştirmedir; gerçek DM'lerin bunu yapmanın kesin bir yolu vardır.
    # Burada, sadece gösterim için karıştırıyoruz.
    # Gerçek bir DM'de, x_{t-1} p(x_{t-1}|x_t) kullanılarak örneklenir, burada x_t koşulludur.
    next_noisy_image_t_minus_1 = combined_x0_estimation + np.random.normal(0, 0.01, noisy_image_t.shape) # Biraz gürültü

    # Gerçek tamamlamada, original_known_region'dan bilinen bölgeler dolaylı olarak
    # veya açıkça her ters adımda yeniden eklenir veya koşullandırılır.
    return next_noisy_image_t_minus_1

# Gösterim için kukla gürültü giderici model
def dummy_denoiser(image, t):
    """Basitlik için sadece girdi görüntüsünü döndüren bir kukla gürültü giderici."""
    return image # Gerçekte, bu gürültüyü veya x_0'ı tahmin eden eğitilmiş bir U-Net olurdu

if __name__ == "__main__":
    # Bu bölüm tipik olarak daha büyük bir difüzyon çıkarım döngüsünün parçası olacaktır.
    # T anında gürültülü bir görüntüyü ve orijinal görüntünün bilinen bölgelerini simüle edin
    # (gerçek kullanım için gerçek görüntü yükleme ve maske oluşturma ile değiştirin)

    # Örnek: kukla bir görüntü ve maske oluşturun
    dummy_img_shape = (128, 128, 3)
    original_image = np.zeros(dummy_img_shape)
    original_image[30:70, 30:70] = [1.0, 0.0, 0.0] # Kırmızı kare
    original_image[50:90, 50:90] = [0.0, 1.0, 0.0] # Yeşil kare

    # Yeşil karenin bir kısmını kapsayan bir maske oluşturun
    mask = create_mask(original_image.shape, mask_center=(60, 60), mask_size=30)
    
    # T zamanında gürültülü bir görüntü başlatın (örn. T'de saf gürültü)
    noisy_image_at_T = np.random.rand(*dummy_img_shape) 
    
    # Gösterim için, tek bir tamamlama adımını ele alalım
    # 'original_known_region' gerçek girdi görüntüsünden türetilecek
    # ve noisy_image_at_T'nin gürültü seviyesiyle eşleşmesi için bir ileri süreçten geçirilecekti
    
    # Basitleştirilmiş bir 'original_known_region' sadece maskenin True olduğu orijinal görüntüdür.
    # Gerçek bir kurulumda, bu, t adımındaki *gürültülü* görüntünün bilinen kısımları olurdu.
    current_noisy_image = np.copy(noisy_image_at_T)
    
    # Denoiser'a girdi olarak simüle etmek için 'current_noisy_image'a maskeyi uygulayın
    # İlk adımlar için bilinmeyen bölge gürültü, bilinen bölge ise orijinal bilinenin gürültülü versiyonudur.
    
    # Bu döngü kavramsal olarak ters difüzyon adımlarını temsil eder (örn. 1000 adımdan 0'a kadar)
    num_inference_steps = 5
    print(f"{num_inference_steps} adımlık kavramsal tamamlama simülasyonu başlıyor...")
    for t_step in range(num_inference_steps, 0, -1):
        print(f"Adım işleniyor: {t_step}")
        
        # Burada, original_known_region, orijinal görüntünün bilinen kısmı olmalı,
        # current_noisy_image ile uyumlu olması için uygun şekilde ölçeklendirilmiş ve gürültülendirilmiş.
        # Bu kavramsal örnek için, orijinal görüntünün gerçek bilinen kısmını kullanalım.
        
        # Daha doğru bir temsil şunları içerir:
        # 1. original_image'dan bilinen kısmı çıkarın.
        # 2. current_noisy_image'ın gürültü seviyesiyle eşleşmesi için bu bilinen kısma gürültü ekleyin.
        # Bu, 'original_known_region_at_t'yi oluşturur.
        
        # Basitleştirme: Karıştırma için doğrudan original_image'ın bilinen kısmını kullanıyoruz
        # Gerçek bir difüzyon modelinde, bu 'original_known_region' dikkatlice türetilirdi
        # mevcut gürültü seviyesi 't_step' ile tutarlılığı sağlamak için.
        
        current_noisy_image = conceptual_inpainting_step(
            current_noisy_image, original_image, mask, dummy_denoiser, t_step
        )
        # İstenirse current_noisy_image'ı görsel olarak inceleyin veya kaydedin
        # Basitlik için burada sadece bir mesaj yazdırıyoruz.

    print("Kavramsal tamamlama simülasyonu tamamlandı. Son görüntü (çok basitleştirilmiş):")
    # Etkiyi görmek için:
    # final_inpainted_image = Image.fromarray((current_noisy_image * 255).astype(np.uint8))
    # final_inpainted_image.save("conceptual_inpainted_result.png")
    # print("Sonuç conceptual_inpainted_result.png dosyasına kaydedildi")

(Kod örneği bölümünün sonu)
```
## 5. Sonuç
Difüzyon Modelleri, görüntü üretimi için dönüştürücü bir teknoloji olarak ortaya çıkmış ve görüntü tamamlama görevine adaptasyonları önemli bir ilerlemeyi temsil etmektedir. Karmaşık veri dağılımlarını öğrenme ve yinelemeli gürültü giderme yeteneklerini kullanarak, DM'ler eksik görüntü bölgeleri için son derece gerçekçi, anlamsal olarak tutarlı ve çeşitli tamamlamalar üretebilir. Hesaplama maliyeti ve tüm senaryolarda mükemmel tutarlılık ile ilgili zorluklar devam etse de, Latent Difüzyon ve sınıflandırıcısız yönlendirme gibi teknikler verimliliği ve kaliteyi önemli ölçüde artırmıştır. Araştırmalar ilerlemeye devam ettikçe, difüzyon modelleri, görsel içerik manipülasyonunda eşi benzeri görülmemiş kontrol ve yaratıcı özgürlük sunarak, yeni nesil görüntü düzenleme ve restorasyon araçlarının temel taşı olmaya adaydır.