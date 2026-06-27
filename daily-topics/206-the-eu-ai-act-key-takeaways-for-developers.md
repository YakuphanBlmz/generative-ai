# The EU AI Act: Key Takeaways for Developers

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Key Concepts and Definitions](#2-key-concepts-and-definitions)
    - [2.1. Defining AI Systems, Providers, and Deployers](#21-defining-ai-systems-providers-and-deployers)
    - [2.2. The Risk-Based Approach](#22-the-risk-based-approach)
        - [2.2.1. Unacceptable Risk AI Systems](#221-unacceptable-risk-ai-systems)
        - [2.2.2. High-Risk AI Systems](#222-high-risk-ai-systems)
        - [2.2.3. Limited Risk AI Systems](#223-limited-risk-ai-systems)
        - [2.2.4. Minimal Risk AI Systems](#224-minimal-risk-ai-systems)
- [3. Core Obligations and Implications for Developers of High-Risk AI Systems](#3-core-obligations-and-implications-for-developers-of-high-risk-ai-systems)
    - [3.1. Robustness, Accuracy, and Cybersecurity](#31-robustness-accuracy-and-cybersecurity)
    - [3.2. Data Governance and Quality](#32-data-governance-and-quality)
    - [3.3. Transparency, Explainability, and Human Oversight](#33-transparency-explainability-and-human-oversight)
    - [3.4. Technical Documentation and Logging](#34-technical-documentation-and-logging)
    - [3.5. Risk Management and Quality Management Systems](#35-risk-management-and-quality-management-systems)
    - [3.6. Conformity Assessment and Post-Market Monitoring](#36-conformity-assessment-and-post-market-monitoring)
- [4. Code Example: AI System Risk Classification Mockup](#4-code-example-ai-system-risk-classification-mockup)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The European Union's Artificial Intelligence Act (EU AI Act), formally adopted in March 2024, represents a landmark legislative effort to regulate **Artificial Intelligence (AI)**. As the world's first comprehensive legal framework for AI, it aims to foster trustworthy AI development and deployment by ensuring that AI systems placed on the EU market and used within the EU are safe, transparent, non-discriminatory, and respect fundamental rights. For developers, innovators, and organizations involved in the entire AI lifecycle, understanding the nuances of this regulation is not merely an exercise in compliance but a strategic imperative. This document provides a detailed overview of the EU AI Act's key provisions, focusing specifically on the practical implications and obligations for developers. It emphasizes the risk-based approach at the core of the Act and outlines the technical and procedural requirements that will shape the future of AI development within and for the European market.

## 2. Key Concepts and Definitions
To navigate the EU AI Act effectively, developers must first grasp its foundational terminology and the central concept of its risk-based regulatory framework.

### 2.1. Defining AI Systems, Providers, and Deployers
The Act introduces several critical definitions that delineate responsibilities:
*   **AI System:** Defined broadly as a machine-based system that operates with varying levels of autonomy and that can, for explicit or implicit objectives, infer from the input it receives how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments. This definition is intended to be technology-neutral and comprehensive, covering a wide range of AI paradigms, including machine learning, logic- and knowledge-based approaches, and statistical approaches.
*   **Provider:** Any natural or legal person, public authority, agency, or other body that develops an AI system or that has an AI system developed and places it on the market or puts it into service under its name or trademark. This typically refers to the organizations or individuals creating and releasing AI products.
*   **Deployer (User):** Any natural or legal person, public authority, agency, or other body using an AI system under its authority, except where the AI system is used in the course of a personal non-professional activity. Deployers are responsible for using AI systems in accordance with the provider's instructions and the Act's provisions.

Understanding these roles is crucial because the Act assigns specific obligations to each, although developers often act as providers or contribute directly to the provider's responsibilities.

### 2.2. The Risk-Based Approach
The EU AI Act employs a **risk-based approach**, imposing stricter obligations on AI systems that pose higher potential risks to fundamental rights, health, safety, and the rule of law. AI systems are categorized into four levels of risk: unacceptable, high, limited, and minimal.

#### 2.2.1. Unacceptable Risk AI Systems
These are AI systems considered a clear threat to fundamental rights and are therefore prohibited. Examples include:
*   Cognitive behavioural manipulation that substantially distorts a person’s behaviour in a manner that causes or is likely to cause that person or another person physical or psychological harm.
*   Social scoring by public authorities based on social behaviour, which leads to detrimental or unfavourable treatment.
*   Real-time remote biometric identification systems in publicly accessible spaces for law enforcement purposes, with limited exceptions.
Developers must ensure their AI systems do not fall into this category, as their development, placement on the market, or use is strictly forbidden.

#### 2.2.2. High-Risk AI Systems
This category is at the core of the Act's regulatory burden for developers. High-risk AI systems are those that pose significant harm to the health, safety, or fundamental rights of persons. The Act identifies high-risk systems in two main categories:
1.  AI systems intended to be used as a **safety component of a product** falling under existing EU harmonization legislation (e.g., medical devices, aviation, critical infrastructure).
2.  AI systems used in specific areas such as:
    *   **Biometric identification and categorization** of natural persons.
    *   **Management and operation of critical infrastructure** (e.g., energy, water, transport).
    *   **Education and vocational training** (e.g., determining access, evaluating learning outcomes).
    *   **Employment, worker management, and access to self-employment** (e.g., recruitment, promotion, task allocation).
    *   **Access to essential private services and public services and benefits** (e.g., creditworthiness assessment, dispatching emergency services).
    *   **Law enforcement** (e.g., lie detectors, predictive policing).
    *   **Migration, asylum, and border control management** (e.g., assessing eligibility).
    *   **Administration of justice and democratic processes** (e.g., assisting judicial authorities).

Developers of high-risk AI systems face the most stringent obligations, detailed in the following section.

#### 2.2.3. Limited Risk AI Systems
These systems are subject to specific transparency obligations, primarily to ensure that individuals are aware they are interacting with an AI. Examples include:
*   AI systems intended to interact with natural persons (e.g., chatbots).
*   AI systems used to generate or manipulate image, audio, or video content (e.g., deepfakes), requiring disclosure that the content is artificially generated or manipulated.
*   AI systems used for emotion recognition or biometric categorization (requiring transparency about their operation).
Developers for these systems must focus on clear communication and disclosure mechanisms.

#### 2.2.4. Minimal Risk AI Systems
The vast majority of AI systems fall into this category. These systems pose no specific risks under the Act and are largely unregulated beyond existing legislation. The Act encourages the development of codes of conduct for these systems on a voluntary basis. For developers, this means a lower regulatory burden, though best practices for ethical AI should always be considered.

## 3. Core Obligations and Implications for Developers of High-Risk AI Systems
For developers engaged in building **high-risk AI systems**, the EU AI Act introduces a comprehensive set of obligations that span the entire lifecycle of the AI system, from design and development to deployment and post-market monitoring. These requirements necessitate a shift towards a **"privacy and ethics by design"** paradigm for AI.

### 3.1. Robustness, Accuracy, and Cybersecurity
High-risk AI systems must be designed and developed to be sufficiently robust and accurate for their intended purpose.
*   **Robustness:** Systems must be resilient to errors, faults, and inconsistencies, and operate reliably throughout their lifecycle. This includes resilience against attempts to manipulate or circumvent the system. Developers must implement rigorous testing protocols, including stress testing, and ensure fail-safe mechanisms are in place.
*   **Accuracy:** The system's performance metrics must consistently meet defined levels of accuracy for its intended use, especially concerning protected groups, to prevent bias or discrimination. Performance testing and validation against relevant datasets are critical.
*   **Cybersecurity:** High-risk AI systems must be resilient against cybersecurity threats, protecting against data breaches, unauthorized access, and malicious attacks that could compromise the AI system's integrity or performance. This involves applying state-of-the-art cybersecurity measures throughout the development process.

### 3.2. Data Governance and Quality
The quality and integrity of data used for training, validation, and testing high-risk AI systems are paramount.
*   **Training, Validation, and Testing Data:** Datasets must be relevant, sufficiently representative, and free from errors and biases. Developers are required to implement robust **data governance** practices to ensure data quality, including data collection protocols, data cleaning, and dataset curation.
*   **Bias Mitigation:** Developers must proactively identify and mitigate potential biases in the data that could lead to discriminatory outcomes. This involves systematic bias assessment and fairness-aware AI development techniques.
*   **Data Management:** Implementing clear data management policies, including data lineage tracking, version control, and access management, is essential for demonstrating compliance.

### 3.3. Transparency, Explainability, and Human Oversight
High-risk AI systems must be designed to allow for **transparency** regarding their operation and to be **interpretable** by humans.
*   **Transparency:** Developers must ensure that high-risk AI systems provide sufficient information about their capabilities, limitations, and performance characteristics to both deployers and affected individuals. This includes clarity on the system's purpose, the data used, and how it performs.
*   **Explainability:** The outputs and decisions generated by high-risk AI systems must be interpretable and explainable to human users, enabling them to understand the basis of the AI's recommendations or actions. This often requires incorporating techniques for **model interpretability** (e.g., LIME, SHAP).
*   **Human Oversight:** High-risk AI systems must be designed with effective human oversight mechanisms. This means ensuring that humans can monitor the system's operation, understand its outputs, override or intervene in its decisions, and deactivate it if necessary. Developers must build interfaces and functionalities that facilitate meaningful human control.

### 3.4. Technical Documentation and Logging
Extensive **technical documentation** is a foundational requirement, encompassing the entire lifecycle of a high-risk AI system.
*   **Technical Documentation:** Providers must draw up and maintain comprehensive technical documentation before placing an AI system on the market or putting it into service. This documentation must include detailed information on the system's design, development, purpose, capabilities, performance, data sources, risk management system, and conformity assessment procedures.
*   **Logging Capabilities:** High-risk AI systems must be designed to automatically record events ("logs") throughout their operation. These logs should enable the traceability of the system's functioning and allow for monitoring, analysis, and auditing, especially concerning actions that might have an impact on fundamental rights or safety. This is critical for post-market monitoring and incident investigation.

### 3.5. Risk Management and Quality Management Systems
Developers, as providers, are required to establish and implement robust systems for managing risks and ensuring quality.
*   **Risk Management System:** This system must be established and continuously maintained throughout the entire lifecycle of the high-risk AI system. It involves identifying, analyzing, evaluating, and mitigating risks to fundamental rights, health, and safety, including residual risks. This is an iterative process.
*   **Quality Management System:** Providers must establish, implement, document, and maintain a quality management system to ensure compliance with the Act. This system should cover all aspects of an AI system's lifecycle, including design, data governance, testing, maintenance, and post-market monitoring. It often aligns with existing ISO standards.

### 3.6. Conformity Assessment and Post-Market Monitoring
Before a high-risk AI system can be placed on the EU market or put into service, it must undergo a **conformity assessment**.
*   **Conformity Assessment:** This is a set of procedures that verify whether an AI system complies with the requirements of the Act. Depending on the specific type of high-risk AI system, this may involve an internal assessment by the provider or a third-party assessment by a notified body. Upon successful assessment, the AI system can be affixed with the **CE marking**, indicating its compliance.
*   **Post-Market Monitoring:** Even after deployment, providers are obligated to implement a system for **post-market monitoring** to continuously collect and analyze data on the AI system's performance, incidents, and potential risks throughout its lifecycle. This includes proactive measures to identify and correct any unforeseen issues, biases, or harms. Serious incidents must be reported to relevant national authorities.

## 4. Code Example: AI System Risk Classification Mockup
This Python snippet illustrates a simplified, conceptual function that a developer might use to classify an AI system's risk based on its intended use and features, aligning with the EU AI Act's framework. This is a mockup and does not represent a legally binding classification.

```python
import enum

class AIRiskLevel(enum.Enum):
    UNACCEPTABLE = "Unacceptable"
    HIGH_RISK = "High-Risk"
    LIMITED_RISK = "Limited-Risk"
    MINIMAL_RISK = "Minimal-Risk"

def classify_ai_system_risk(intended_use_cases: list[str], features: dict) -> AIRiskLevel:
    """
    Classifies an AI system's risk level based on its intended use cases and specific features,
    mimicking the EU AI Act's risk-based approach.

    :param intended_use_cases: A list of strings describing the primary applications of the AI system.
    :param features: A dictionary of key features/capabilities of the AI system.
    :return: An AIRiskLevel enum indicating the classified risk.
    """

    # 1. Check for Unacceptable Risk
    prohibited_uses = [
        "cognitive behavioral manipulation",
        "social scoring by public authorities",
        "real-time remote biometric identification in public spaces"
    ]
    if any(p_use in uc.lower() for uc in intended_use_cases for p_use in prohibited_uses):
        return AIRiskLevel.UNACCEPTABLE
    
    # 2. Check for High Risk
    high_risk_triggers = [
        "biometric identification",
        "critical infrastructure management",
        "education access",
        "employment decisions",
        "access to essential services",
        "law enforcement",
        "migration control",
        "judicial assistance"
    ]
    is_safety_component = features.get("is_safety_component", False)
    
    if is_safety_component or any(hr_trigger in uc.lower() for uc in intended_use_cases for hr_trigger in high_risk_triggers):
        return AIRiskLevel.HIGH_RISK

    # 3. Check for Limited Risk (Transparency obligations)
    limited_risk_triggers = [
        "human interaction", # e.g., chatbots
        "generate deepfakes",
        "manipulate media",
        "emotion recognition",
        "biometric categorization" # if not high-risk context
    ]
    if any(lr_trigger in uc.lower() for uc in intended_use_cases for lr_trigger in limited_risk_triggers):
        return AIRiskLevel.LIMITED_RISK

    # 4. Default to Minimal Risk
    return AIRiskLevel.MINIMAL_RISK

# --- Example Usage ---
# Example 1: A high-risk system
system_a_uses = ["Employment decisions for job applicants", "Critical infrastructure monitoring"]
system_a_features = {"is_safety_component": False, "data_sources": ["applicant resumes", "sensor data"]}
print(f"System A Risk: {classify_ai_system_risk(system_a_uses, system_a_features).value}")

# Example 2: A limited-risk system
system_b_uses = ["Customer support chatbot for FAQs", "Generating marketing content (deepfakes)"]
system_b_features = {"interacts_with_humans": True, "generates_media": True}
print(f"System B Risk: {classify_ai_system_risk(system_b_uses, system_b_features).value}")

# Example 3: A minimal-risk system
system_c_uses = ["Spam filtering for email", "Personalized movie recommendations"]
system_c_features = {"impact_on_rights": "low"}
print(f"System C Risk: {classify_ai_system_risk(system_c_uses, system_c_features).value}")

# Example 4: An unacceptable risk system
system_d_uses = ["Real-time remote biometric identification in public spaces for general surveillance"]
system_d_features = {}
print(f"System D Risk: {classify_ai_system_risk(system_d_uses, system_d_features).value}")

(End of code example section)
```
## 5. Conclusion
The EU AI Act marks a significant turning point in the regulation of artificial intelligence, shifting the landscape for developers from an unregulated frontier to a structured environment focused on safety, fundamental rights, and ethical principles. For developers, particularly those working on **high-risk AI systems**, the Act demands a proactive and integrated approach to compliance. This involves embedding principles of **robustness, accuracy, data governance, transparency, human oversight, and cybersecurity** throughout the entire development lifecycle, rather than treating compliance as an afterthought.

The obligations outlined, from rigorous data quality management and comprehensive technical documentation to establishing robust risk and quality management systems and participating in conformity assessments, require substantial re-engineering of existing development practices. Developers must embrace explainable AI (XAI) techniques, prioritize bias mitigation, and design systems that allow for meaningful human control.

While challenging, the EU AI Act also presents an opportunity. Developers who proactively integrate these requirements into their development processes can build AI systems that are inherently more trustworthy, resilient, and ethically sound. This commitment to responsible AI development can lead to competitive advantages, fostering greater public trust and opening new markets for compliant and high-quality AI solutions. As the Act comes into full effect, staying informed and adapting development methodologies will be paramount for any developer aiming to innovate and deploy AI within the EU's evolving regulatory ecosystem.

---
<br>

<a name="türkçe-içerik"></a>
## AB Yapay Zeka Yasası: Geliştiriciler İçin Temel Çıkarımlar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Tanımlar](#2-temel-kavramlar-ve-tanımlar)
    - [2.1. Yapay Zeka Sistemleri, Sağlayıcılar ve Kullanıcıların Tanımlanması](#21-yapay-zeka-sistemleri-sağlayıcılar-ve-kullanıcıların-tanımlanması)
    - [2.2. Risk Tabanlı Yaklaşım](#22-risk-tabanlı-yaklaşım)
        - [2.2.1. Kabul Edilemez Riskli Yapay Zeka Sistemleri](#221-kabul-edilemez-riskli-yapay-zeka-sistemleri)
        - [2.2.2. Yüksek Riskli Yapay Zeka Sistemleri](#222-yüksek-riskli-yapay-zeka-sistemleri)
        - [2.2.3. Sınırlı Riskli Yapay Zeka Sistemleri](#223-sınırlı-riskli-yapay-zeka-sistemleri)
        - [2.2.4. Minimal Riskli Yapay Zeka Sistemleri](#224-minimal-riskli-yapay-zeka-sistemleri)
- [3. Yüksek Riskli Yapay Zeka Sistemleri Geliştiricileri İçin Temel Yükümlülükler ve Çıkarımlar](#3-yüksek-riskli-yapay-zeka-sistemleri-geliştiricileri-için-temel-yükümlülükler-ve-çıkarımlar)
    - [3.1. Sağlamlık, Doğruluk ve Siber Güvenlik](#31-sağlamlık-doğruluk-ve-siber-güvenlik)
    - [3.2. Veri Yönetişimi ve Kalitesi](#32-veri-yönetişimi-ve-kalitesi)
    - [3.3. Şeffaflık, Açıklanabilirlik ve İnsan Gözetimi](#33-şeffaflık-açıklanabilirlik-ve-insan-gözetimi)
    - [3.4. Teknik Dokümantasyon ve Kayıt Tutma](#34-teknik-dokümantasyon-ve-kayıt-tutma)
    - [3.5. Risk Yönetimi ve Kalite Yönetim Sistemleri](#35-risk-yönetimi-ve-kalite-yönetim-sistemleri)
    - [3.6. Uygunluk Değerlendirmesi ve Pazar Sonrası Gözetim](#36-uygunluk-değerlendirmesi-ve-pazar-sonrası-gözetim)
- [4. Kod Örneği: Yapay Zeka Sistemi Risk Sınıflandırma Taslağı](#4-kod-örneği-yapay-zeka-sistemi-risk-sınıflandırma-taslağı)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
Avrupa Birliği'nin Yapay Zeka Yasası (AB Yapay Zeka Yasası), Mart 2024'te resmen kabul edilmiş olup, **Yapay Zeka (YZ)** alanını düzenlemek için dönüm noktası niteliğinde bir yasal çabayı temsil etmektedir. YZ için dünyanın ilk kapsamlı yasal çerçevesi olarak, AB pazarına sunulan ve AB içinde kullanılan YZ sistemlerinin güvenli, şeffaf, ayrımcı olmayan ve temel haklara saygılı olmasını sağlayarak güvenilir YZ geliştirme ve dağıtımını teşvik etmeyi amaçlamaktadır. YZ yaşam döngüsünün tamamında yer alan geliştiriciler, yenilikçiler ve kuruluşlar için bu düzenlemenin inceliklerini anlamak sadece bir uyumluluk egzersizi değil, aynı zamanda stratejik bir zorunluluktur. Bu belge, AB Yapay Zeka Yasası'nın temel hükümlerine ilişkin ayrıntılı bir genel bakış sunmakta ve özellikle geliştiriciler için pratik çıkarımlara ve yükümlülüklere odaklanmaktadır. Yasa'nın temelindeki risk tabanlı yaklaşımı vurgulamakta ve Avrupa pazarı içinde ve için YZ geliştirmenin geleceğini şekillendirecek teknik ve prosedürel gereksinimleri özetlemektedir.

## 2. Temel Kavramlar ve Tanımlar
AB Yapay Zeka Yasası'nı etkili bir şekilde kullanabilmek için geliştiricilerin öncelikle temel terminolojisini ve risk tabanlı düzenleyici çerçevesinin merkezi kavramını kavraması gerekmektedir.

### 2.1. Yapay Zeka Sistemleri, Sağlayıcılar ve Kullanıcıların Tanımlanması
Yasa, sorumlulukları belirleyen birkaç kritik tanım sunmaktadır:
*   **Yapay Zeka Sistemi:** Değişen özerklik düzeylerinde çalışan ve açık veya zımni hedefler için, aldığı girdiden tahminler, içerik, öneriler veya fiziksel ya da sanal ortamları etkileyebilecek kararlar gibi çıktılar üretmeyi çıkarabilen makine tabanlı bir sistem olarak tanımlanır. Bu tanım, teknoloji nötr ve kapsamlı olmayı amaçlar; makine öğrenimi, mantık ve bilgi tabanlı yaklaşımlar ve istatistiksel yaklaşımlar dahil olmak üzere geniş bir YZ paradigması yelpazesini kapsar.
*   **Sağlayıcı:** Bir YZ sistemini geliştiren veya bir YZ sistemini geliştirilmesini sağlayan ve kendi adı veya ticari markası altında piyasaya süren veya hizmete sunan herhangi bir gerçek veya tüzel kişi, kamu kurumu, ajans veya diğer kurum. Bu genellikle YZ ürünlerini oluşturan ve yayınlayan kuruluşları veya bireyleri ifade eder.
*   **Kullanıcı (Dağıtıcı):** Yetkisi altında bir YZ sistemini kullanan herhangi bir gerçek veya tüzel kişi, kamu kurumu, ajans veya diğer kurum; YZ sisteminin kişisel, mesleki olmayan bir faaliyet kapsamında kullanıldığı durumlar hariç. Kullanıcılar, YZ sistemlerini sağlayıcının talimatlarına ve Yasa'nın hükümlerine uygun olarak kullanmaktan sorumludur.

Bu rolleri anlamak çok önemlidir, çünkü Yasa her birine belirli yükümlülükler atar; ancak geliştiriciler genellikle sağlayıcı olarak hareket eder veya sağlayıcının sorumluluklarına doğrudan katkıda bulunurlar.

### 2.2. Risk Tabanlı Yaklaşım
AB Yapay Zeka Yasası, temel haklara, sağlığa, güvenliğe ve hukukun üstünlüğüne potansiyel olarak daha yüksek riskler oluşturan YZ sistemlerine daha sıkı yükümlülükler getiren **risk tabanlı bir yaklaşım** benimsemektedir. YZ sistemleri dört risk seviyesine ayrılmıştır: kabul edilemez, yüksek, sınırlı ve minimal.

#### 2.2.1. Kabul Edilemez Riskli Yapay Zeka Sistemleri
Bunlar, temel haklara açık bir tehdit olarak kabul edilen ve bu nedenle yasaklanan YZ sistemleridir. Örnekler şunlardır:
*   Bir kişinin davranışını, o kişiye veya başka bir kişiye fiziksel veya psikolojik zarar veren veya vermesi muhtemel bir şekilde önemli ölçüde bozan bilişsel davranış manipülasyonu.
*   Kamu otoriteleri tarafından sosyal davranışa dayalı olarak yapılan ve zararlı veya olumsuz muameleye yol açan sosyal puanlama.
*   Sınırlı istisnalar dışında, kamuya açık alanlarda emniyet teşkilatı amaçları için gerçek zamanlı uzaktan biyometrik tanımlama sistemleri.
Geliştiriciler, YZ sistemlerinin bu kategoriye girmediğinden emin olmalıdır, çünkü bunların geliştirilmesi, piyasaya sürülmesi veya kullanılması kesinlikle yasaktır.

#### 2.2.2. Yüksek Riskli Yapay Zeka Sistemleri
Bu kategori, Yasa'nın geliştiriciler için düzenleyici yükünün çekirdeğini oluşturmaktadır. Yüksek riskli YZ sistemleri, kişilerin sağlığına, güvenliğine veya temel haklarına önemli zarar verme potansiyeli taşıyan sistemlerdir. Yasa, yüksek riskli sistemleri iki ana kategoride tanımlar:
1.  Mevcut AB uyum mevzuatı kapsamına giren bir **ürünün güvenlik bileşeni** olarak kullanılması amaçlanan YZ sistemleri (örneğin, tıbbi cihazlar, havacılık, kritik altyapı).
2.  Aşağıdaki gibi belirli alanlarda kullanılan YZ sistemleri:
    *   Doğal kişilerin **biyometrik tanımlanması ve kategorize edilmesi**.
    *   **Kritik altyapının yönetimi ve işletilmesi** (örneğin, enerji, su, ulaşım).
    *   **Eğitim ve mesleki eğitim** (örneğin, erişimi belirleme, öğrenme çıktılarını değerlendirme).
    *   **İstihdam, işçi yönetimi ve serbest mesleğe erişim** (örneğin, işe alım, terfi, görev dağılımı).
    *   **Temel özel hizmetlere ve kamu hizmetlerine ve faydalarına erişim** (örneğin, kredi değerliliği değerlendirmesi, acil servisleri yönlendirme).
    *   **Emniyet teşkilatı** (örneğin, yalan dedektörleri, tahmine dayalı polislik).
    *   **Göç, iltica ve sınır kontrol yönetimi** (örneğin, uygunluğun değerlendirilmesi).
    *   **Adaletin idaresi ve demokratik süreçler** (örneğin, adli makamlara yardımcı olma).

Yüksek riskli YZ sistemleri geliştiricileri, bir sonraki bölümde ayrıntıları verilen en katı yükümlülüklerle karşı karşıyadır.

#### 2.2.3. Sınırlı Riskli Yapay Zeka Sistemleri
Bu sistemler, esas olarak kişilerin bir YZ ile etkileşimde olduklarının farkında olmalarını sağlamak için belirli şeffaflık yükümlülüklerine tabidir. Örnekler şunlardır:
*   Doğal kişilerle etkileşim kurması amaçlanan YZ sistemleri (örneğin, sohbet robotları).
*   Görüntü, ses veya video içeriği (örneğin, deepfake'ler) oluşturmak veya manipüle etmek için kullanılan YZ sistemleri, içeriğin yapay olarak oluşturulduğunun veya manipüle edildiğinin açıklanmasını gerektirir.
*   Duygu tanıma veya biyometrik kategorizasyon için kullanılan YZ sistemleri (işleyişleri hakkında şeffaflık gerektirir).
Bu sistemlerin geliştiricileri, net iletişim ve açıklama mekanizmalarına odaklanmalıdır.

#### 2.2.4. Minimal Riskli Yapay Zeka Sistemleri
YZ sistemlerinin büyük çoğunluğu bu kategoriye girer. Bu sistemler, Yasa kapsamında özel riskler taşımaz ve mevcut mevzuatın ötesinde büyük ölçüde düzenlenmemiştir. Yasa, bu sistemler için gönüllülük esasına dayalı davranış kurallarının geliştirilmesini teşvik etmektedir. Geliştiriciler için bu, daha düşük bir düzenleyici yük anlamına gelir; ancak etik YZ için en iyi uygulamalar her zaman dikkate alınmalıdır.

## 3. Yüksek Riskli Yapay Zeka Sistemleri Geliştiricileri İçin Temel Yükümlülükler ve Çıkarımlar
**Yüksek riskli YZ sistemleri** geliştiren geliştiriciler için, AB Yapay Zeka Yasası, YZ sisteminin tasarım ve geliştirmeden dağıtım ve pazar sonrası gözetimine kadar tüm yaşam döngüsünü kapsayan kapsamlı bir dizi yükümlülük getirir. Bu gereksinimler, YZ için **"tasarımla gizlilik ve etik"** paradigmasına geçişi zorunlu kılmaktadır.

### 3.1. Sağlamlık, Doğruluk ve Siber Güvenlik
Yüksek riskli YZ sistemleri, amaçlanan amaçları için yeterince sağlam ve doğru olacak şekilde tasarlanmalı ve geliştirilmelidir.
*   **Sağlamlık:** Sistemler hatalara, kusurlara ve tutarsızlıklara karşı dayanıklı olmalı ve yaşam döngüleri boyunca güvenilir bir şekilde çalışmalıdır. Bu, sistemi manipüle etme veya atlatma girişimlerine karşı dayanıklılığı da içerir. Geliştiriciler, stres testi dahil olmak üzere titiz test protokolleri uygulamalı ve hataya karşı güvenli mekanizmaların mevcut olduğundan emin olmalıdır.
*   **Doğruluk:** Sistemin performans metrikleri, özellikle korunan gruplar söz konusu olduğunda, yanlılığı veya ayrımcılığı önlemek için amaçlanan kullanımı için tanımlanmış doğruluk seviyelerini sürekli olarak karşılamalıdır. İlgili veri kümelerine karşı performans testi ve doğrulama kritik öneme sahiptir.
*   **Siber Güvenlik:** Yüksek riskli YZ sistemleri, YZ sisteminin bütünlüğünü veya performansını tehlikeye atabilecek veri ihlalleri, yetkisiz erişim ve kötü niyetli saldırılara karşı korunmak için siber güvenlik tehditlerine karşı dayanıklı olmalıdır. Bu, geliştirme süreci boyunca son teknoloji siber güvenlik önlemlerinin uygulanmasını içerir.

### 3.2. Veri Yönetişimi ve Kalitesi
Yüksek riskli YZ sistemlerinin eğitimi, doğrulanması ve test edilmesi için kullanılan verilerin kalitesi ve bütünlüğü büyük önem taşımaktadır.
*   **Eğitim, Doğrulama ve Test Verileri:** Veri kümeleri, ilgili, yeterince temsil edici ve hatalardan ve önyargılardan arınmış olmalıdır. Geliştiriciler, veri toplama protokolleri, veri temizleme ve veri kümesi kürasyonu dahil olmak üzere veri kalitesini sağlamak için sağlam **veri yönetişimi** uygulamaları uygulamalıdır.
*   **Yanlılık Azaltma:** Geliştiriciler, ayrımcı sonuçlara yol açabilecek verilerdeki potansiyel yanlılıkları proaktif olarak tanımlamalı ve azaltmalıdır. Bu, sistematik yanlılık değerlendirmesi ve adil YZ geliştirme tekniklerini içerir.
*   **Veri Yönetimi:** Veri kökeni izleme, sürüm kontrolü ve erişim yönetimi dahil olmak üzere açık veri yönetimi politikalarının uygulanması, uyumluluğu göstermek için esastır.

### 3.3. Şeffaflık, Açıklanabilirlik ve İnsan Gözetimi
Yüksek riskli YZ sistemleri, çalışma şekilleri açısından **şeffaflık** sağlayacak ve insanlar tarafından **yorumlanabilir** olacak şekilde tasarlanmalıdır.
*   **Şeffaflık:** Geliştiriciler, yüksek riskli YZ sistemlerinin hem kullanıcılara hem de etkilenen bireylere yetenekleri, sınırlamaları ve performans özellikleri hakkında yeterli bilgi sağladığından emin olmalıdır. Bu, sistemin amacı, kullanılan veriler ve nasıl performans gösterdiği hakkında netliği içerir.
*   **Açıklanabilirlik:** Yüksek riskli YZ sistemleri tarafından üretilen çıktılar ve kararlar, insan kullanıcılar için yorumlanabilir ve açıklanabilir olmalı, YZ'nin önerilerinin veya eylemlerinin temelini anlamalarını sağlamalıdır. Bu genellikle **model yorumlanabilirliği** tekniklerini (örneğin, LIME, SHAP) dahil etmeyi gerektirir.
*   **İnsan Gözetimi:** Yüksek riskli YZ sistemleri, etkili insan gözetim mekanizmaları ile tasarlanmalıdır. Bu, insanların sistemin çalışmasını izleyebilmesini, çıktılarını anlayabilmesini, kararlarını geçersiz kılabileceğini veya müdahale edebileceğini ve gerekirse devre dışı bırakabileceğini sağlamak anlamına gelir. Geliştiriciler, anlamlı insan kontrolünü kolaylaştıran arayüzler ve işlevler oluşturmalıdır.

### 3.4. Teknik Dokümantasyon ve Kayıt Tutma
Kapsamlı **teknik dokümantasyon**, yüksek riskli YZ sisteminin tüm yaşam döngüsünü kapsayan temel bir gereksinimdir.
*   **Teknik Dokümantasyon:** Sağlayıcılar, bir YZ sistemini piyasaya sürmeden veya hizmete sokmadan önce kapsamlı teknik dokümantasyon hazırlamalı ve sürdürmelidir. Bu dokümantasyon, sistemin tasarımı, geliştirilmesi, amacı, yetenekleri, performansı, veri kaynakları, risk yönetim sistemi ve uygunluk değerlendirme prosedürleri hakkında ayrıntılı bilgileri içermelidir.
*   **Kayıt Tutma Yetenekleri:** Yüksek riskli YZ sistemleri, çalışmaları boyunca olayları ("günlükler") otomatik olarak kaydedecek şekilde tasarlanmalıdır. Bu günlükler, sistemin işleyişinin izlenebilirliğini sağlamalı ve özellikle temel haklar veya güvenlik üzerinde etkisi olabilecek eylemlerle ilgili olarak izleme, analiz ve denetime izin vermelidir. Bu, pazar sonrası izleme ve olay araştırması için kritik öneme sahiptir.

### 3.5. Risk Yönetimi ve Kalite Yönetim Sistemleri
Geliştiriciler, sağlayıcılar olarak, riskleri yönetmek ve kaliteyi sağlamak için sağlam sistemler kurmak ve uygulamak zorundadır.
*   **Risk Yönetim Sistemi:** Bu sistem, yüksek riskli YZ sisteminin tüm yaşam döngüsü boyunca kurulmalı ve sürekli olarak sürdürülmelidir. Temel haklara, sağlığa ve güvenliğe yönelik riskleri, kalan riskler dahil olmak üzere tanımlamayı, analiz etmeyi, değerlendirmeyi ve azaltmayı içerir. Bu, yinelemeli bir süreçtir.
*   **Kalite Yönetim Sistemi:** Sağlayıcılar, Yasa'ya uyumu sağlamak için bir kalite yönetim sistemi kurmalı, uygulamalı, belgelemeli ve sürdürmelidir. Bu sistem, bir YZ sisteminin tüm yaşam döngüsü yönlerini kapsamalıdır; bunlar arasında tasarım, veri yönetişimi, test, bakım ve pazar sonrası izleme bulunur. Genellikle mevcut ISO standartlarıyla uyumludur.

### 3.6. Uygunluk Değerlendirmesi ve Pazar Sonrası Gözetim
Yüksek riskli bir YZ sistemi AB pazarına sunulmadan veya hizmete sokulmadan önce bir **uygunluk değerlendirmesinden** geçmelidir.
*   **Uygunluk Değerlendirmesi:** Bu, bir YZ sisteminin Yasa'nın gereksinimlerine uygun olup olmadığını doğrulayan bir dizi prosedürdür. Yüksek riskli YZ sisteminin özel türüne bağlı olarak, bu, sağlayıcı tarafından dahili bir değerlendirme veya bildirilmiş bir kuruluş tarafından üçüncü taraf bir değerlendirme içerebilir. Başarılı değerlendirme üzerine, YZ sistemi, uyumluluğunu gösteren **CE işareti** ile etiketlenebilir.
*   **Pazar Sonrası Gözetim:** Dağıtımdan sonra bile, sağlayıcılar, YZ sisteminin performansı, olayları ve potansiyel riskleri hakkında yaşam döngüsü boyunca sürekli veri toplamak ve analiz etmek için bir **pazar sonrası gözetim** sistemi uygulamak zorundadır. Bu, öngörülemeyen sorunları, yanlılıkları veya zararları belirlemek ve düzeltmek için proaktif önlemleri içerir. Ciddi olaylar ilgili ulusal yetkililere bildirilmelidir.

## 4. Kod Örneği: Yapay Zeka Sistemi Risk Sınıflandırma Taslağı
Bu Python kodu, AB Yapay Zeka Yasası'nın risk tabanlı yaklaşımına uygun olarak, bir geliştiricinin bir YZ sisteminin riskini amaçlanan kullanımına ve özelliklerine göre sınıflandırmak için kullanabileceği basitleştirilmiş, kavramsal bir işlevi göstermektedir. Bu bir taslaktır ve yasal olarak bağlayıcı bir sınıflandırmayı temsil etmez.

```python
import enum

class AIRiskLevel(enum.Enum):
    UNACCEPTABLE = "Kabul Edilemez"
    HIGH_RISK = "Yüksek Riskli"
    LIMITED_RISK = "Sınırlı Riskli"
    MINIMAL_RISK = "Minimal Riskli"

def classify_ai_system_risk(intended_use_cases: list[str], features: dict) -> AIRiskLevel:
    """
    Bir YZ sisteminin risk seviyesini, amaçlanan kullanım durumlarına ve belirli özelliklerine göre sınıflandırır,
    AB Yapay Zeka Yasası'nın risk tabanlı yaklaşımını taklit eder.

    :param intended_use_cases: YZ sisteminin birincil uygulamalarını açıklayan dizelerin bir listesi.
    :param features: YZ sisteminin temel özelliklerini/yeteneklerini içeren bir sözlük.
    :return: Sınıflandırılmış riski gösteren bir AIRiskLevel enum değeri.
    """

    # 1. Kabul Edilemez Risk Kontrolü
    prohibited_uses = [
        "bilişsel davranışsal manipülasyon",
        "kamu otoriteleri tarafından sosyal puanlama",
        "kamuya açık alanlarda gerçek zamanlı uzaktan biyometrik tanımlama"
    ]
    if any(p_use in uc.lower() for uc in intended_use_cases for p_use in prohibited_uses):
        return AIRiskLevel.UNACCEPTABLE
    
    # 2. Yüksek Risk Kontrolü
    high_risk_triggers = [
        "biyometrik tanımlama",
        "kritik altyapı yönetimi",
        "eğitime erişim",
        "istihdam kararları",
        "temel hizmetlere erişim",
        "kolluk kuvvetleri",
        "göç kontrolü",
        "adli yardım"
    ]
    is_safety_component = features.get("is_safety_component", False)
    
    if is_safety_component or any(hr_trigger in uc.lower() for uc in intended_use_cases for hr_trigger in high_risk_triggers):
        return AIRiskLevel.HIGH_RISK

    # 3. Sınırlı Risk Kontrolü (Şeffaflık yükümlülükleri)
    limited_risk_triggers = [
        "insan etkileşimi", # örn., sohbet robotları
        "deepfake üretimi",
        "medya manipülasyonu",
        "duygu tanıma",
        "biyometrik kategorizasyon" # yüksek riskli bağlamda değilse
    ]
    if any(lr_trigger in uc.lower() for uc in intended_use_cases for lr_trigger in limited_risk_triggers):
        return AIRiskLevel.LIMITED_RISK

    # 4. Varsayılan olarak Minimal Risk
    return AIRiskLevel.MINIMAL_RISK

# --- Örnek Kullanım ---
# Örnek 1: Yüksek riskli bir sistem
system_a_uses = ["İş başvuruları için istihdam kararları", "Kritik altyapı izlemesi"]
system_a_features = {"is_safety_component": False, "data_sources": ["başvuru özgeçmişleri", "sensör verileri"]}
print(f"Sistem A Riski: {classify_ai_system_risk(system_a_uses, system_a_features).value}")

# Örnek 2: Sınırlı riskli bir sistem
system_b_uses = ["SSS için müşteri destek sohbet botu", "Pazarlama içeriği oluşturma (deepfake'ler)"]
system_b_features = {"interacts_with_humans": True, "generates_media": True}
print(f"Sistem B Riski: {classify_ai_system_risk(system_b_uses, system_b_features).value}")

# Örnek 3: Minimal riskli bir sistem
system_c_uses = ["E-posta için spam filtreleme", "Kişiselleştirilmiş film önerileri"]
system_c_features = {"impact_on_rights": "düşük"}
print(f"Sistem C Riski: {classify_ai_system_risk(system_c_uses, system_c_features).value}")

# Örnek 4: Kabul edilemez riskli bir sistem
system_d_uses = ["Genel gözetim için kamusal alanlarda gerçek zamanlı uzaktan biyometrik tanımlama"]
system_d_features = {}
print(f"Sistem D Riski: {classify_ai_system_risk(system_d_uses, system_d_features).value}")

(Kod örneği bölümünün sonu)
```
## 5. Sonuç
AB Yapay Zeka Yasası, yapay zeka düzenlemesinde önemli bir dönüm noktası oluşturarak, geliştiriciler için mevzuatsız bir alandan güvenlik, temel haklar ve etik ilkelere odaklanan yapılandırılmış bir ortama geçişi sağlamaktadır. Özellikle **yüksek riskli YZ sistemleri** üzerinde çalışan geliştiriciler için Yasa, uyumluluğa proaktif ve entegre bir yaklaşım gerektirmektedir. Bu, uyumluluğu sonradan akla gelen bir düşünce olarak ele almak yerine, **sağlamlık, doğruluk, veri yönetişimi, şeffaflık, insan gözetimi ve siber güvenlik** ilkelerini tüm geliştirme yaşam döngüsü boyunca yerleştirmeyi içerir.

Titiz veri kalite yönetiminden kapsamlı teknik dokümantasyona, sağlam risk ve kalite yönetim sistemleri kurmaktan uygunluk değerlendirmelerine katılmaya kadar özetlenen yükümlülükler, mevcut geliştirme uygulamalarının önemli ölçüde yeniden düzenlenmesini gerektirmektedir. Geliştiricilerin açıklanabilir YZ (XAI) tekniklerini benimsemesi, yanlılık azaltmayı önceliklendirmesi ve anlamlı insan kontrolüne izin veren sistemler tasarlaması gerekmektedir.

Zorlayıcı olmakla birlikte, AB Yapay Zeka Yasası aynı zamanda bir fırsat da sunmaktadır. Bu gereksinimleri geliştirme süreçlerine proaktif bir şekilde entegre eden geliştiriciler, doğası gereği daha güvenilir, dayanıklı ve etik açıdan sağlam YZ sistemleri oluşturabilirler. Sorumlu YZ geliştirmeye olan bu bağlılık, daha fazla kamu güveni oluşturarak ve uyumlu ve yüksek kaliteli YZ çözümleri için yeni pazarlar açarak rekabet avantajları sağlayabilir. Yasa tam olarak yürürlüğe girdiğinde, gelişen AB düzenleyici ekosistemi içinde YZ'yi yenilemeyi ve dağıtmayı amaçlayan her geliştirici için bilgili kalmak ve geliştirme metodolojilerini uyarlamak hayati önem taşıyacaktır.

