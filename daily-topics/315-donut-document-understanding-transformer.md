# Donut: Document Understanding Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. Architectural Design and Working Principle](#3-architectural-design-and-working-principle)
  - [3.1. Vision Transformer Encoder](#31-vision-transformer-encoder)
  - [3.2. Transformer Decoder](#32-transformer-decoder)
  - [3.3. Self-Supervised Pre-training](#33-self-supervised-pre-training)
- [4. Key Advantages and Applications](#4-key-advantages-and-applications)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The digital transformation of enterprise operations has led to an exponential increase in the volume of digital documents. Effectively extracting structured information from these diverse, often visually complex, documents remains a significant challenge. Traditional approaches frequently rely on a pipeline of **Optical Character Recognition (OCR)** followed by **Natural Language Processing (NLP)** techniques. While somewhat effective, this multi-stage process is prone to error propagation, struggles with visual variations, and often fails to capture the intricate spatial and semantic relationships inherent in document layouts.

**Donut (Document Understanding Transformer)** represents a paradigm shift in document AI by introducing an end-to-end, OCR-free approach to **Document Visual Question Answering (DocVQA)** and document information extraction. Developed by a team at NAVER Cloud, Donut re-frames document understanding as an image-to-text generation task, leveraging the power of **Vision Transformers (ViT)** and standard Transformer decoders. By directly consuming raw document images and generating structured JSON or text outputs, Donut bypasses the limitations of explicit OCR, offering a more robust, efficient, and visually-aware solution for complex document understanding tasks. This document provides a comprehensive overview of Donut's architecture, its underlying principles, key advantages, and practical applications.

<a name="2-background-and-motivation"></a>
## 2. Background and Motivation
Historically, document understanding systems have largely followed a modular pipeline. The initial step involves OCR engines converting pixels into textual representations. Subsequently, NLP models are applied to this extracted text for tasks such as named entity recognition, relation extraction, or classification. This approach, however, suffers from several inherent drawbacks:
*   **Error Propagation:** OCR errors, particularly on low-quality scans or documents with complex layouts, directly impact the performance of subsequent NLP models.
*   **Loss of Visual Information:** Separating the visual and textual processing stages means crucial layout and spatial information, which often conveys significant semantic meaning (e.g., the proximity of a value to a label on an invoice), is either lost or requires complex, heuristic-based re-integration.
*   **Pipeline Complexity:** Managing and optimizing separate OCR and NLP components adds complexity to development and deployment.
*   **Domain Specificity:** OCR engines might require fine-tuning for specific fonts, languages, or document types, and NLP models need extensive training on domain-specific corpora.

The advent of **multimodal learning** and **Transformer architectures** has opened new avenues for tackling these challenges. Models like LayoutLM and its successors began to incorporate layout information alongside text and visual features. However, most still relied on an underlying OCR step.

Donut's motivation stems from the desire to create a truly end-to-end system that can understand documents holistically, much like humans do, by simultaneously processing both visual and textual cues without explicit OCR. This generative approach, directly mapping document images to structured textual output, marks a significant departure from conventional methods and aligns with the success of end-to-end image captioning and visual question answering systems in other domains.

<a name="3-architectural-design-and-working-principle"></a>
## 3. Architectural Design and Working Principle
Donut's architecture is fundamentally a **sequence-to-sequence (seq2seq)** model, comprising a Vision Transformer encoder and a text-based Transformer decoder. Its design allows it to directly process an input document image and generate a sequence of tokens representing the desired structured output, often in a **JSON** format.

### <a name="31-vision-transformer-encoder"></a>3.1. Vision Transformer Encoder
The encoder component of Donut is typically a pre-trained **Vision Transformer (ViT)**. The ViT model takes a document image as input and processes it by dividing the image into fixed-size non-overlapping patches. Each patch is then linearly embedded, and positional embeddings are added to retain spatial information. These patch embeddings, along with a special `[CLS]` token, are fed into a standard Transformer encoder stack. The output of the ViT encoder is a rich, contextualized representation of the input image, encapsulating both visual features and implicit textual content, without explicitly performing character recognition. This means the encoder learns to "see" the text and its layout directly from the pixels.

### <a name="32-transformer-decoder"></a>3.2. Transformer Decoder
The decoder is a standard **Transformer decoder** architecture, similar to those used in natural language generation tasks. It takes the output representations from the ViT encoder (cross-attention) and generates a sequence of tokens autoregressively. During training, the decoder is conditioned on a target sequence, which is typically a serialized JSON string representing the information to be extracted from the document. For instance, for an invoice, the target could be `{"invoice_number": "INV-123", "total_amount": "100.00"}`. The decoder learns to generate these tokens, including special tokens for JSON structure (e.g., `<s>`, `<pad>`, `</s>`, `<unk>`), key names, and their corresponding values.

### <a name="33-self-supervised-pre-training"></a>3.3. Self-Supervised Pre-training
A critical aspect of Donut's success lies in its **self-supervised pre-training** strategy. Donut is pre-trained on a massive dataset of document images and their corresponding text (e.g., from public datasets like DocVQA or synthetic document generation). The pre-training task is essentially an image-to-text generation task: given a document image, the model is trained to generate its entire textual content, including structural information. This process allows the model to learn a strong correlation between visual features (characters, words, layouts) and their semantic meaning without explicit OCR supervision. The decoder is trained to reconstruct the original text content, effectively internalizing the "reading" process.

After this extensive pre-training, Donut can be fine-tuned for various downstream tasks, such as:
*   **Document Visual Question Answering (DocVQA):** Answering questions about the content of a document image.
*   **Form Understanding:** Extracting specific fields from structured forms like invoices, receipts, or application forms.
*   **Information Extraction:** Identifying and extracting key-value pairs or entities from less structured documents.

<a name="4-key-advantages-and-applications"></a>
## 4. Key Advantages and Applications
Donut offers several significant advantages over traditional OCR-based document understanding pipelines:

*   **End-to-End, OCR-Free:** The most prominent advantage is its ability to directly map pixels to structured output without an intermediate OCR step. This eliminates error propagation from OCR and simplifies the overall system.
*   **Visual-Textual Integration:** By processing images directly with a Vision Transformer, Donut inherently captures the spatial and layout information alongside visual cues, which is crucial for understanding complex documents where visual context is paramount.
*   **Robustness to Visual Noise:** Without reliance on perfect OCR, Donut can be more robust to variations in document quality, font styles, colors, and minor distortions.
*   **Generative Flexibility:** The generative nature of Donut allows it to produce highly structured output (e.g., JSON) tailored to specific extraction needs, making it adaptable to various information extraction tasks.
*   **Reduced Pipeline Complexity:** A single model handles both visual interpretation and text generation, reducing the complexity of development, deployment, and maintenance.

Donut finds its applications across a wide range of industries and use cases:

*   **Automated Data Entry:** Automatically extracting data from invoices, receipts, purchase orders, and other business documents, streamlining financial processes.
*   **Customer Service:** Enhancing chatbots and virtual assistants with the ability to "read" and understand submitted document images (e.g., utility bills, ID cards).
*   **Healthcare:** Extracting patient information from medical records, prescriptions, and lab reports.
*   **Legal and Compliance:** Automating the review and extraction of clauses and entities from contracts and legal documents.
*   **Digital Archiving:** Creating structured, searchable metadata from scanned historical documents.

<a name="5-code-example"></a>
## 5. Code Example
The following Python snippet demonstrates how to use a pre-trained Donut model from the Hugging Face `transformers` library to parse an invoice image and extract structured information.

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import os

# 1. Load pre-trained processor and model
# Using a model specifically fine-tuned for invoice parsing (e.g., from Hugging Face Hub)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# 2. Define the image path (replace with your actual image path)
# For practical execution, ensure 'image_path' points to a valid PIL Image object.
image_path = "sample_invoice.png" # Replace with your actual invoice image path

# Attempt to load the image, or create a dummy image if not found
try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded image from {image_path}")
except FileNotFoundError:
    print(f"Sample image not found at {image_path}. Creating a dummy image for demonstration.")
    from PIL import ImageDraw, ImageFont
    image = Image.new('RGB', (800, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(image)
    try: # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError: # Fallback to default if arial.ttf is not found
        font = ImageFont.load_default()
    d.text((100, 100), "Invoice No: INV-2023-001\nTotal: $123.45\nDate: 2023-10-27", fill=(0, 0, 0), font=font)
    print("Using a generated dummy image for demonstration.")


# 3. Prepare the image for the model
pixel_values = processor(image, return_tensors="pt").pixel_values

# 4. Generate the output sequence
# The model will generate tokens based on the image, representing the structured data.
# Max_length can be adjusted based on expected output length.
# No decoder input IDs are provided for the initial generation step as it's image-to-text.
output_ids = model.generate(
    pixel_values,
    max_length=processor.tokenizer.model_max_length,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    num_beams=1, # For simpler output; can increase for higher quality
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# 5. Decode the generated IDs into text
# The processor can handle special tokens and convert the generated sequence into a dictionary.
result_text = processor.batch_decode(output_ids.sequences, skip_special_tokens=True)[0]

# 6. Post-process the output to get a structured JSON
# Donut models often output a JSON string, which needs to be parsed.
try:
    parsed_output = json.loads(result_text)
    print("\nExtracted Information (JSON):")
    print(json.dumps(parsed_output, indent=2))
except json.JSONDecodeError:
    print("\nFailed to decode JSON from output:", result_text)
    print("Raw model output:", result_text)


(End of code example section)
```
<a name="6-conclusion"></a>
## 6. Conclusion
Donut represents a significant leap forward in the field of **document AI**. By adopting an end-to-end, OCR-free, image-to-text generation approach, it effectively addresses many of the limitations inherent in traditional pipeline-based document understanding systems. Its unique architecture, combining Vision Transformers with standard decoders and leveraging extensive self-supervised pre-training, enables it to holistically capture both visual and semantic information directly from document images. This capability makes Donut exceptionally robust to visual noise and diverse document layouts, paving the way for more accurate and efficient information extraction. From automating data entry in finance to enhancing customer service in various industries, Donut's applications are vast and transformative, underscoring its potential to revolutionize how organizations interact with and derive value from their document-centric data. As research in multimodal foundation models continues to advance, Donut exemplifies a powerful direction for future developments in AI-powered document intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## Donut: Belge Anlama Trafosu

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. Mimari Tasarım ve Çalışma Prensibi](#3-mimari-tasarım-ve-çalışma-prensibi)
  - [3.1. Görüntü Trafosu Kodlayıcı](#31-görüntü-trafosu-kodlayıcı)
  - [3.2. Trafo Kod Çözücü](#32-trafo-kod-çözücü)
  - [3.3. Kendi Kendine Denetimli Ön Eğitim](#33-kendi-kendine-denetimli-ön-eğitim)
- [4. Temel Avantajlar ve Uygulamalar](#4-temel-avantajlar-ve-uygulamalar)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Kurumsal operasyonların dijital dönüşümü, dijital belge hacminde üstel bir artışa yol açmıştır. Bu çeşitli, genellikle görsel olarak karmaşık belgelerden yapılandırılmış bilgi çıkarmak önemli bir zorluk olmaya devam etmektedir. Geleneksel yaklaşımlar genellikle **Optik Karakter Tanıma (OCR)** ve ardından **Doğal Dil İşleme (NLP)** tekniklerinden oluşan bir ardışık düzene dayanır. Bu çok aşamalı süreç bir dereceye kadar etkili olsa da, hata yayılımına eğilimlidir, görsel varyasyonlarla mücadele eder ve belge düzenlerindeki karmaşık uzaysal ve anlamsal ilişkileri yakalamada genellikle yetersiz kalır.

**Donut (Document Understanding Transformer)**, belge anlama alanında bir paradigma değişimi temsil ederek, **Belge Görsel Soru Cevaplama (DocVQA)** ve belge bilgi çıkarımı için uçtan uca, OCR'siz bir yaklaşım sunmaktadır. NAVER Cloud'daki bir ekip tarafından geliştirilen Donut, **Görsel Trafoların (ViT)** ve standart Trafo kod çözücülerin gücünden yararlanarak belge anlamayı bir görüntüden metne üretim görevi olarak yeniden ele alır. Ham belge görüntülerini doğrudan tüketerek ve yapılandırılmış JSON veya metin çıktıları üreterek, Donut açık OCR'nin sınırlamalarını aşar ve karmaşık belge anlama görevleri için daha sağlam, verimli ve görsel olarak duyarlı bir çözüm sunar. Bu belge, Donut'un mimarisine, temel prensiplerine, temel avantajlarına ve pratik uygulamalarına kapsamlı bir genel bakış sunmaktadır.

<a name="2-arka-plan-ve-motivasyon"></a>
## 2. Arka Plan ve Motivasyon
Tarihsel olarak, belge anlama sistemleri büyük ölçüde modüler bir ardışık düzeni takip etmiştir. İlk adım, OCR motorlarının pikselleri metinsel temsillerine dönüştürmesini içerir. Daha sonra, çıkarılan bu metne adlandırılmış varlık tanıma, ilişki çıkarımı veya sınıflandırma gibi görevler için NLP modelleri uygulanır. Ancak bu yaklaşım, çeşitli içsel dezavantajlardan muzdariptir:
*   **Hata Yayılımı:** Özellikle düşük kaliteli taramalar veya karmaşık düzenlere sahip belgelerde OCR hataları, sonraki NLP modellerinin performansını doğrudan etkiler.
*   **Görsel Bilgi Kaybı:** Görsel ve metinsel işleme aşamalarını ayırmak, genellikle önemli anlamsal anlam taşıyan (örneğin, bir faturadaki bir değere bir etiketin yakınlığı gibi) kritik düzen ve uzamsal bilgilerin ya kaybolmasına ya da karmaşık, sezgisel tabanlı yeniden entegrasyon gerektirmesine neden olur.
*   **Ardışık Düzen Karmaşıklığı:** Ayrı OCR ve NLP bileşenlerini yönetmek ve optimize etmek, geliştirme ve dağıtım karmaşıklığını artırır.
*   **Alan Özgüllüğü:** OCR motorları belirli yazı tipleri, diller veya belge türleri için ince ayar gerektirebilir ve NLP modelleri alana özgü metinler üzerinde kapsamlı eğitime ihtiyaç duyar.

**Çok modlu öğrenme** ve **Trafo mimarilerinin** ortaya çıkışı, bu zorlukların üstesinden gelmek için yeni yollar açmıştır. LayoutLM ve ardılları gibi modeller, metin ve görsel özelliklerin yanı sıra düzen bilgilerini de dahil etmeye başlamıştır. Ancak çoğu, hala temel bir OCR adımına güveniyordu.

Donut'un motivasyonu, insanların yaptığı gibi, belge görüntülerini ve metinsel ipuçlarını eş zamanlı olarak, açık bir OCR olmaksızın işleyerek belgeleri bütünsel olarak anlayabilen gerçekten uçtan uca bir sistem yaratma arzusundan kaynaklanmaktadır. Belge görüntülerini doğrudan yapılandırılmış metinsel çıktıya eşleyen bu üretken yaklaşım, geleneksel yöntemlerden önemli bir sapmayı işaret eder ve diğer alanlarda uçtan uca görüntü altyazılama ve görsel soru cevaplama sistemlerinin başarısıyla uyumludur.

<a name="3-mimari-tasarım-ve-çalışma-prensibi"></a>
## 3. Mimari Tasarım ve Çalışma Prensibi
Donut'un mimarisi temel olarak, bir Görsel Trafo kodlayıcı ve metin tabanlı bir Trafo kod çözücüden oluşan bir **diziden-diziye (seq2seq)** modelidir. Tasarımı, bir giriş belge görüntüsünü doğrudan işlemesine ve genellikle bir **JSON** formatında istenen yapılandırılmış çıktıyı temsil eden bir belirteç dizisi üretmesine olanak tanır.

### <a name="31-görüntü-trafosu-kodlayıcı"></a>3.1. Görüntü Trafosu Kodlayıcı
Donut'un kodlayıcı bileşeni genellikle önceden eğitilmiş bir **Görsel Trafo (ViT)**'dur. ViT modeli, bir belge görüntüsünü girdi olarak alır ve görüntüyü sabit boyutlu, çakışmayan yamalara bölerek işler. Her yama daha sonra doğrusal olarak gömülür ve uzaysal bilgiyi korumak için konumsal gömmeler eklenir. Bu yama gömmeleri, özel bir `[CLS]` belirteci ile birlikte standart bir Trafo kodlayıcı yığınına beslenir. ViT kodlayıcısının çıktısı, açıkça karakter tanıma yapmadan hem görsel özellikleri hem de örtük metinsel içeriği kapsayan, girdi görüntüsünün zengin, bağlamsallaştırılmış bir temsilidir. Bu, kodlayıcının metni ve düzenini doğrudan piksellerden "görmeyi" öğrendiği anlamına gelir.

### <a name="32-trafo-kod-çözücü"></a>3.2. Trafo Kod Çözücü
Kod çözücü, doğal dil üretimi görevlerinde kullanılanlara benzer standart bir **Trafo kod çözücü** mimarisidir. ViT kodlayıcısından (çapraz dikkat) gelen çıktı temsillerini alır ve ardışık olarak bir belirteç dizisi üretir. Eğitim sırasında kod çözücü, genellikle belgeden çıkarılacak bilgiyi temsil eden serileştirilmiş bir JSON dizgisi olan bir hedef diziye koşullandırılır. Örneğin, bir fatura için hedef `{"fatura_numarası": "INV-123", "toplam_tutar": "100.00"}` olabilir. Kod çözücü, JSON yapısı için özel belirteçler (örneğin, `<s>`, `<pad>`, `</s>`, `<unk>`), anahtar adları ve bunlara karşılık gelen değerler dahil olmak üzere bu belirteçleri üretmeyi öğrenir.

### <a name="33-kendi-kendine-denetimli-ön-eğitim"></a>3.3. Kendi Kendine Denetimli Ön Eğitim
Donut'un başarısının kritik bir yönü, **kendi kendine denetimli ön eğitim** stratejisidir. Donut, büyük bir belge görüntüsü ve ilgili metin veri kümesi üzerinde önceden eğitilir (örneğin, DocVQA gibi genel veri kümelerinden veya sentetik belge üretiminden). Ön eğitim görevi esasen bir görüntüden metne üretim görevidir: bir belge görüntüsü verildiğinde, model yapısal bilgiler de dahil olmak üzere tüm metinsel içeriğini üretmek üzere eğitilir. Bu süreç, modelin açık bir OCR denetimi olmaksızın görsel özellikler (karakterler, kelimeler, düzenler) ile anlamsal anlamları arasında güçlü bir korelasyon öğrenmesini sağlar. Kod çözücü, orijinal metin içeriğini yeniden yapılandırmak üzere eğitilir ve böylece "okuma" sürecini etkili bir şekilde içselleştirir.

Bu kapsamlı ön eğitimden sonra Donut, çeşitli aşağı akış görevleri için ince ayar yapılabilir:
*   **Belge Görsel Soru Cevaplama (DocVQA):** Bir belge görüntüsünün içeriği hakkında soruları yanıtlama.
*   **Form Anlama:** Faturalar, makbuzlar veya başvuru formları gibi yapılandırılmış formlardan belirli alanları çıkarma.
*   **Bilgi Çıkarımı:** Daha az yapılandırılmış belgelerden anahtar-değer çiftlerini veya varlıkları tanımlama ve çıkarma.

<a name="4-temel-avantajlar-ve-uygulamalar"></a>
## 4. Temel Avantajlar ve Uygulamalar
Donut, geleneksel OCR tabanlı belge anlama ardışık düzenlerine göre çeşitli önemli avantajlar sunar:

*   **Uçtan Uca, OCR'siz:** En belirgin avantajı, ara bir OCR adımı olmadan pikselleri doğrudan yapılandırılmış çıktıya eşleyebilme yeteneğidir. Bu, OCR'den kaynaklanan hata yayılımını ortadan kaldırır ve genel sistemi basitleştirir.
*   **Görsel-Metinsel Entegrasyon:** Görüntüleri doğrudan bir Görsel Trafo ile işleyerek, Donut görsel ipuçlarının yanı sıra uzamsal ve düzen bilgilerini de doğal olarak yakalar; bu, görsel bağlamın çok önemli olduğu karmaşık belgeleri anlamak için hayati önem taşır.
*   **Görsel Gürültüye Karşı Sağlamlık:** Mükemmel OCR'ye bağımlılık olmadan, Donut belge kalitesindeki, yazı tipi stillerindeki, renklerdeki ve küçük bozulmalardaki varyasyonlara karşı daha sağlam olabilir.
*   **Üretken Esneklik:** Donut'un üretken doğası, belirli çıkarma ihtiyaçlarına göre uyarlanmış yüksek düzeyde yapılandırılmış çıktılar (örneğin, JSON) üretmesine olanak tanır, bu da onu çeşitli bilgi çıkarma görevlerine uyarlanabilir kılar.
*   **Azaltılmış Ardışık Düzen Karmaşıklığı:** Tek bir model hem görsel yorumlamayı hem de metin üretimini ele alarak geliştirme, dağıtım ve bakım karmaşıklığını azaltır.

Donut, çok çeşitli sektörlerde ve kullanım durumlarında uygulamalar bulur:

*   **Otomatik Veri Girişi:** Faturalardan, makbuzlardan, siparişlerden ve diğer ticari belgelerden verileri otomatik olarak çıkararak finansal süreçleri kolaylaştırma.
*   **Müşteri Hizmetleri:** Sohbet botlarını ve sanal asistanları gönderilen belge görüntülerini (örneğin, faturalar, kimlik kartları) "okuma" ve anlama yeteneğiyle geliştirme.
*   **Sağlık Hizmetleri:** Tıbbi kayıtlardan, reçetelerden ve laboratuvar raporlarından hasta bilgilerini çıkarma.
*   **Hukuk ve Uyum:** Sözleşmelerden ve yasal belgelerden maddelerin ve varlıkların incelenmesini ve çıkarılmasını otomatikleştirme.
*   **Dijital Arşivleme:** Taranmış tarihi belgelerden yapılandırılmış, aranabilir meta veriler oluşturma.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği
Aşağıdaki Python kod parçacığı, bir faturayı ayrıştırmak ve yapılandırılmış bilgi çıkarmak için Hugging Face `transformers` kütüphanesinden önceden eğitilmiş bir Donut modelinin nasıl kullanılacağını göstermektedir.

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import os

# 1. Önceden eğitilmiş işlemciyi ve modeli yükle
# Fatura ayrıştırma için özel olarak ince ayar yapılmış bir model kullanılıyor (örn. Hugging Face Hub'dan)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# 2. Görüntü yolunu tanımla (kendi gerçek görüntü yolunuzla değiştirin)
# Pratik uygulama için, 'image_path' geçerli bir PIL Image nesnesini işaret ettiğinden emin olun.
image_path = "örnek_fatura.png" # Gerçek fatura görüntünüzün yoluyla değiştirin

# Görüntüyü yüklemeye çalışın veya bulunamazsa bir kukla görüntü oluşturun
try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görüntü {image_path} konumunda bulunamadı.")
    image = Image.open(image_path).convert("RGB")
    print(f"Görüntü {image_path} konumundan başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Örnek görüntü {image_path} konumunda bulunamadı. Gösterim için bir kukla görüntü oluşturuluyor.")
    from PIL import ImageDraw, ImageFont
    image = Image.new('RGB', (800, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(image)
    try: # Sistem yazı tipi kullanmaya çalış
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError: # arial.ttf bulunamazsa varsayılana geri dön
        font = ImageFont.load_default()
    d.text((100, 100), "Fatura No: INV-2023-001\nToplam: 123.45 TL\nTarih: 2023-10-27", fill=(0, 0, 0), font=font)
    print("Gösterim için oluşturulmuş kukla görüntü kullanılıyor.")


# 3. Görüntüyü model için hazırla
pixel_values = processor(image, return_tensors="pt").pixel_values

# 4. Çıktı dizisini oluştur
# Model, yapılandırılmış veriyi temsil eden belirteçleri görüntüye göre üretecektir.
# Maksimum uzunluk, beklenen çıktı uzunluğuna göre ayarlanabilir.
# Görüntüden metne olduğu için başlangıç üretme adımında kod çözücü giriş kimlikleri sağlanmaz.
output_ids = model.generate(
    pixel_values,
    max_length=processor.tokenizer.model_max_length,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    num_beams=1, # Daha basit çıktı için; daha yüksek kalite için artırılabilir
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# 5. Üretilen kimlikleri metne dönüştür
# İşlemci özel belirteçleri işleyebilir ve üretilen diziyi bir sözlüğe dönüştürebilir.
result_text = processor.batch_decode(output_ids.sequences, skip_special_tokens=True)[0]

# 6. Yapılandırılmış JSON elde etmek için çıktıyı ön işlemden geçir
# Donut modelleri genellikle ayrıştırılması gereken bir JSON dizgisi çıktısı verir.
try:
    parsed_output = json.loads(result_text)
    print("\nÇıkarılan Bilgiler (JSON):")
    print(json.dumps(parsed_output, indent=2, ensure_ascii=False)) # ensure_ascii=False Türkçe karakterler için
except json.JSONDecodeError:
    print("\nÇıktıdan JSON çözme başarısız oldu:", result_text)
    print("Ham model çıktısı:", result_text)


(Kod örneği bölümünün sonu)
```
<a name="6-sonuç"></a>
## 6. Sonuç
Donut, **belge yapay zekası** alanında önemli bir ileri adımı temsil etmektedir. Uçtan uca, OCR'siz, görüntüden metne üretim yaklaşımını benimseyerek, geleneksel ardışık düzen tabanlı belge anlama sistemlerinin doğal sınırlamalarının çoğunu etkili bir şekilde ele almaktadır. Görsel Trafoları standart kod çözücülerle birleştiren ve kapsamlı kendi kendine denetimli ön eğitimden yararlanan benzersiz mimarisi, belge görüntülerinden hem görsel hem de anlamsal bilgileri bütünsel olarak yakalamasını sağlar. Bu yetenek, Donut'u görsel gürültüye ve çeşitli belge düzenlerine karşı son derece sağlam kılar, daha doğru ve verimli bilgi çıkarımının önünü açar. Finansta otomatik veri girişinden çeşitli sektörlerde müşteri hizmetlerini geliştirmeye kadar, Donut'un uygulamaları geniştir ve dönüştürücüdür, kuruluşların belge merkezli verileriyle nasıl etkileşim kurduğunu ve bunlardan nasıl değer elde ettiğini devrim niteliğinde değiştirebilecek potansiyelini vurgulamaktadır. Çok modlu temel modellerdeki araştırmalar ilerlemeye devam ettikçe, Donut yapay zeka destekli belge zekasında gelecekteki gelişmeler için güçlü bir yönü örneklendirmektedir.
