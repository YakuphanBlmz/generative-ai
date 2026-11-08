# Streamlit for Data Science Web Apps

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. What is Streamlit?](#2-what-is-streamlit)
- [3. Key Features and Advantages](#3-key-features-and-advantages)
- [4. Building a Simple Streamlit App](#4-building-a-simple-streamlit-app)
- [5. Streamlit in the Data Science Workflow](#5-streamlit-in-the-data-science-workflow)
- [6. Advanced Concepts and Best Practices](#6-advanced-concepts-and-best-practices)
- [7. Code Example: A Simple Data Explorer](#7-code-example-a-simple-data-explorer)
- [8. Conclusion](#8-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The rapidly evolving field of data science increasingly demands effective ways to present analytical insights and interactive models to a broader audience, often comprising non-technical stakeholders. While traditional analytical tools provide robust capabilities for data processing and model building, they often fall short in delivering user-friendly interfaces that enable real-time exploration and engagement. This challenge has spurred the development of various frameworks designed to bridge the gap between complex data science outputs and intuitive web applications. Among these, **Streamlit** has emerged as a particularly potent tool, revolutionizing the process of creating custom web applications for data science and machine learning projects with remarkable ease and speed. This document explores Streamlit's capabilities, its position within the data science ecosystem, and its practical application in developing interactive data-driven web solutions.

<a name="2-what-is-streamlit"></a>
## 2. What is Streamlit?
Streamlit is an open-source Python library that transforms data scripts into interactive web applications. Designed with data scientists and machine learning engineers in mind, it significantly reduces the overhead typically associated with web development, allowing users to focus primarily on their data and models rather than intricate front-end coding. Unlike traditional web frameworks that require knowledge of HTML, CSS, JavaScript, and complex routing, Streamlit abstracts away these complexities, enabling the creation of powerful data applications using only Python. Its philosophy centers on **simplicity and speed**, advocating for a "scripting" approach where the application's UI automatically updates as the underlying data or code changes. This paradigm enables rapid prototyping and iterative development, making it an invaluable asset for showcasing proofs of concept, building interactive dashboards, or deploying machine learning models.

<a name="3-key-features-and-advantages"></a>
## 3. Key Features and Advantages
Streamlit’s popularity stems from several distinctive features and inherent advantages:

*   **Python-Native Development:** The most significant advantage is the ability to build entire web applications using **pure Python**. This eliminates the need for multi-language development, making it accessible to data scientists who are primarily proficient in Python.
*   **Rapid Prototyping:** With its simple API and automatic UI updates, Streamlit facilitates **exceptionally fast development cycles**. A basic interactive application can often be created with just a few lines of code.
*   **Component-Based Architecture:** Streamlit provides a rich set of pre-built widgets and components (e.g., sliders, buttons, text inputs, dataframes, charts) that can be easily integrated into applications. This modularity simplifies UI construction and ensures a consistent user experience.
*   **Automatic Reruns:** Whenever a user interacts with a widget or the underlying script changes, Streamlit automatically reruns the entire script from top to bottom. While this might seem inefficient, Streamlit employs **intelligent caching mechanisms** (`st.cache_data`, `st.cache_resource`) to optimize performance by storing the results of expensive computations.
*   **Interactive Data Visualization:** Integration with popular Python plotting libraries like Matplotlib, Plotly, Altair, and Bokeh is seamless, allowing for the creation of **dynamic and interactive data visualizations** directly within the web app.
*   **Easy Deployment:** Streamlit applications can be deployed relatively easily on various platforms, including Streamlit Community Cloud, Heroku, Docker, and AWS, making them accessible to a global audience.
*   **Active Community and Ecosystem:** A vibrant and growing community supports Streamlit, contributing to its extensive documentation, numerous tutorials, and a rich ecosystem of third-party components and extensions.

<a name="4-building-a-simple-streamlit-app"></a>
## 4. Building a Simple Streamlit App
Developing a Streamlit application typically involves importing the `streamlit` library and using its functions to display data, collect user input, and render visualizations. The core principle revolves around writing a Python script that Streamlit then executes to build and update the web interface.

A fundamental Streamlit application structure often includes:
1.  **Importing `streamlit`**: `import streamlit as st`
2.  **Adding a Title and Text**: Using `st.title()`, `st.header()`, `st.subheader()`, `st.write()`, `st.markdown()` to convey information.
3.  **Displaying Data**: `st.dataframe()`, `st.table()` for tabular data, or `st.metric()` for key performance indicators.
4.  **Creating Interactive Widgets**: `st.slider()`, `st.selectbox()`, `st.text_input()`, `st.button()` to allow user interaction.
5.  **Rendering Charts**: `st.line_chart()`, `st.bar_chart()`, `st.pyplot()` for various plotting needs.

The execution flow is straightforward: save your Python script (e.g., `app.py`) and run it from your terminal using `streamlit run app.py`. Streamlit will then open the application in your web browser.

<a name="5-streamlit-in-the-data-science-workflow"></a>
## 5. Streamlit in the Data Science Workflow
Streamlit fits seamlessly into various stages of the data science workflow, enhancing collaboration and impact:

*   **Exploratory Data Analysis (EDA):** Data scientists can quickly build interactive dashboards to explore datasets, visualize distributions, identify correlations, and detect anomalies. This allows for **dynamic data exploration** that goes beyond static reports.
*   **Model Prototyping and Evaluation:** It provides an excellent platform for building interactive UIs around machine learning models. Users can input parameters, see real-time predictions, and evaluate model performance metrics, making it easier to **iterate and refine models**.
*   **Demonstrating Proof of Concept (POC):** Streamlit is ideal for rapidly creating POC applications to showcase the potential of a data product or analytical solution to stakeholders, garnering feedback and buy-in early in the development cycle.
*   **Production Deployment of ML Models:** While not a full-fledged production-grade web framework like Flask or Django, Streamlit can serve as a lightweight front-end for deploying machine learning models, enabling predictions via a user-friendly interface.
*   **Educational Tools:** Its ease of use makes it a popular choice for creating **interactive tutorials and educational tools** for teaching data science concepts.

<a name="6-advanced-concepts-and-best-practices"></a>
## 6. Advanced Concepts and Best Practices
Beyond basic functionality, Streamlit offers several advanced features and patterns that improve application performance, scalability, and user experience:

*   **Session State (`st.session_state`):** This powerful feature allows applications to **maintain state across reruns**, enabling more complex interactions where user inputs or computed values need to persist. It's crucial for building multi-page apps or applications with complex workflows.
*   **Caching Mechanisms (`st.cache_data`, `st.cache_resource`):** As mentioned, Streamlit automatically reruns the entire script. Caching decorators are essential for **optimizing performance** by preventing re-execution of expensive functions or data loading operations when their inputs haven't changed.
    *   `st.cache_data`: Caches data, typically for functions that return dataframes, lists, or dictionaries.
    *   `st.cache_resource`: Caches resources like machine learning models or database connections that are expensive to initialize.
*   **Layout and Containers:** Streamlit offers functions like `st.columns()`, `st.tabs()`, `st.expander()`, and `st.container()` to organize content effectively, create responsive layouts, and improve the visual appeal of applications.
*   **Custom Components:** For highly specialized UI elements not available out-of-the-box, Streamlit allows the integration of **custom components** built using React, enabling developers to extend its functionality.
*   **Theming:** Applications can be customized visually through theming options, allowing adjustment of colors, fonts, and overall aesthetics to match brand guidelines or personal preferences.
*   **Deployment Strategies:** For production, containerization with Docker is a common approach, providing environment consistency and easier scaling. Streamlit Community Cloud offers a direct, free deployment option for public applications.

<a name="7-code-example-a-simple-data-explorer"></a>
## 7. Code Example: A Simple Data Explorer
The following Python script demonstrates a basic Streamlit application that allows users to explore a dummy dataset, filter it by category, and visualize a subset of its values.

```python
import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration for a wider layout
st.set_page_config(layout="wide")

# Main title of the application
st.title("Simple Data Explorer App")
st.write("This application demonstrates basic Streamlit features for interactive data exploration. "
         "It generates synthetic data and allows users to filter and visualize it.")

# Generate some dummy data for demonstration
@st.cache_data
def load_data():
    """Generates and returns a DataFrame with dummy data."""
    np.random.seed(42) # for reproducibility
    data = pd.DataFrame({
        'Category': np.random.choice(['Alpha', 'Beta', 'Gamma'], 100),
        'Value': np.random.rand(100) * 100,
        'Timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    })
    return data

data = load_data()

# Section 1: Data Preview
st.header("1. Data Preview")
st.markdown("Here's a glimpse of the generated dataset:")
st.dataframe(data.head(10)) # Display the first 10 rows

# Section 2: Interactive Filtering
st.header("2. Interactive Filtering")
st.markdown("Select a category to filter the data and view specific subsets.")

# Create a selectbox in the sidebar for category selection
with st.sidebar:
    st.subheader("Filter Options")
    selected_category = st.selectbox(
        "Choose a Category",
        options=data['Category'].unique(),
        help="Select a category to filter the main dataset."
    )

filtered_data = data[data['Category'] == selected_category]

st.write(f"Displaying data for the selected category: **{selected_category}**")
st.dataframe(filtered_data)

# Section 3: Basic Plotting
st.header("3. Basic Plotting")
st.markdown(f"A simple line chart showing 'Value' over 'Timestamp' for **{selected_category}**:")
st.line_chart(filtered_data.set_index('Timestamp')['Value'])

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.info(
    "This application was built using Streamlit to demonstrate its capabilities for "
    "creating interactive web applications for data science. "
    "Run `streamlit run your_app_file.py` to launch this app locally."
)

(End of code example section)
```

<a name="8-conclusion"></a>
## 8. Conclusion
Streamlit represents a significant advancement in democratizing the creation of data-driven web applications. By enabling data scientists to build sophisticated, interactive UIs entirely in Python, it dramatically accelerates the process of sharing insights, deploying models, and fostering collaboration. Its focus on simplicity, rapid development, and rich component set makes it an indispensable tool for anyone looking to transform their data science projects from static analyses into dynamic, engaging, and accessible web experiences. As the demand for explainable AI and interactive data products continues to grow, Streamlit is poised to remain at the forefront of tools empowering data professionals to communicate their work effectively to a wider audience.

---
<br>

<a name="türkçe-içerik"></a>
## Veri Bilimi Web Uygulamaları İçin Streamlit

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Streamlit Nedir?](#2-streamlit-nedir)
- [3. Temel Özellikler ve Avantajlar](#3-temel-özellikler-ve-avantajlar)
- [4. Basit Bir Streamlit Uygulaması Oluşturma](#4-basit-bir-streamlit-uygulaması-oluşturma)
- [5. Veri Bilimi İş Akışında Streamlit](#5-veri-bilimi-iş-akışında-streamlit)
- [6. İleri Düzey Konular ve En İyi Uygulamalar](#6-ileri-düzey-konular-ve-en-iyi-uygulamalar)
- [7. Kod Örneği: Basit Bir Veri Gezgini](#7-kod-örneği-basit-bir-veri-gezgini)
- [8. Sonuç](#8-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Veri biliminin hızla gelişen alanı, analitik içgörüleri ve etkileşimli modelleri, genellikle teknik olmayan paydaşlardan oluşan daha geniş bir kitleye sunmak için etkili yollara giderek daha fazla ihtiyaç duymaktadır. Geleneksel analitik araçlar, veri işleme ve model oluşturma için güçlü yetenekler sunarken, gerçek zamanlı keşif ve etkileşimi sağlayan kullanıcı dostu arayüzler sunma konusunda genellikle yetersiz kalmaktadır. Bu zorluk, karmaşık veri bilimi çıktıları ile sezgisel web uygulamaları arasındaki boşluğu doldurmak için tasarlanmış çeşitli çerçevelerin geliştirilmesini teşvik etmiştir. Bunlar arasında **Streamlit**, veri bilimi ve makine öğrenimi projeleri için özel web uygulamaları oluşturma sürecini olağanüstü kolaylık ve hızla devrim yaratan özellikle güçlü bir araç olarak ortaya çıkmıştır. Bu belge, Streamlit'in yeteneklerini, veri bilimi ekosistemindeki konumunu ve etkileşimli veri odaklı web çözümleri geliştirmedeki pratik uygulamasını incelemektedir.

<a name="2-streamlit-nedir"></a>
## 2. Streamlit Nedir?
Streamlit, veri komut dosyalarını etkileşimli web uygulamalarına dönüştüren açık kaynaklı bir Python kütüphanesidir. Veri bilimcileri ve makine öğrenimi mühendisleri düşünülerek tasarlanmış olup, genellikle web geliştirme ile ilişkili genel gideri önemli ölçüde azaltır ve kullanıcıların karmaşık ön uç kodlamadan ziyade öncelikli olarak verilerine ve modellerine odaklanmasına olanak tanır. HTML, CSS, JavaScript ve karmaşık yönlendirme bilgisi gerektiren geleneksel web çerçevelerinden farklı olarak, Streamlit bu karmaşıklıkları soyutlayarak yalnızca Python kullanarak güçlü veri uygulamaları oluşturmayı mümkün kılar. Felsefesi, **basitlik ve hıza** odaklanarak, temel veriler veya kod değiştikçe uygulamanın kullanıcı arayüzünün otomatik olarak güncellendiği bir "komut dosyası oluşturma" yaklaşımını savunur. Bu paradigma, hızlı prototipleme ve yinelemeli geliştirmeyi sağlayarak, kavram kanıtlarını sergilemek, etkileşimli kontrol panelleri oluşturmak veya makine öğrenimi modellerini dağıtmak için paha biçilmez bir varlık haline getirir.

<a name="3-temel-özellikler-ve-avantajlar"></a>
## 3. Temel Özellikler ve Avantajlar
Streamlit'in popülaritesi, birkaç ayırt edici özelliğinden ve doğal avantajından kaynaklanmaktadır:

*   **Python Yerel Geliştirme:** En önemli avantaj, tüm web uygulamalarını **saf Python** kullanarak inşa etme yeteneğidir. Bu, birden fazla dil gerektiren geliştirme ihtiyacını ortadan kaldırarak, esas olarak Python'da yetkin olan veri bilimcileri için erişilebilir hale getirir.
*   **Hızlı Prototipleme:** Basit API'si ve otomatik kullanıcı arayüzü güncellemeleri sayesinde, Streamlit **olağanüstü hızlı geliştirme döngüleri** sağlar. Temel bir etkileşimli uygulama genellikle sadece birkaç satır kodla oluşturulabilir.
*   **Bileşen Tabanlı Mimari:** Streamlit, uygulamalara kolayca entegre edilebilen zengin bir önceden oluşturulmuş widget ve bileşen (örneğin, kaydırıcılar, düğmeler, metin girişleri, veri çerçeveleri, grafikler) seti sunar. Bu modülerlik, kullanıcı arayüzü oluşturmayı basitleştirir ve tutarlı bir kullanıcı deneyimi sağlar.
*   **Otomatik Tekrarlar:** Bir kullanıcı bir widget ile etkileşim kurduğunda veya temel betik değiştiğinde, Streamlit tüm betiği otomatik olarak baştan sona yeniden çalıştırır. Bu verimsiz görünse de, Streamlit pahalı hesaplamaların sonuçlarını depolayarak performansı optimize etmek için **akıllı önbellekleme mekanizmaları** (`st.cache_data`, `st.cache_resource`) kullanır.
*   **Etkileşimli Veri Görselleştirme:** Matplotlib, Plotly, Altair ve Bokeh gibi popüler Python çizim kütüphaneleriyle entegrasyonu sorunsuz olup, doğrudan web uygulaması içinde **dinamik ve etkileşimli veri görselleştirmeleri** oluşturulmasına olanak tanır.
*   **Kolay Dağıtım:** Streamlit uygulamaları, Streamlit Community Cloud, Heroku, Docker ve AWS dahil olmak üzere çeşitli platformlarda nispeten kolayca dağıtılabilir ve böylece küresel bir kitleye erişilebilir hale gelir.
*   **Aktif Topluluk ve Ekosistem:** Canlı ve büyüyen bir topluluk Streamlit'i desteklemekte, kapsamlı belgelerine, sayısız eğitimine ve zengin üçüncü taraf bileşenleri ve uzantıları ekosistemine katkıda bulunmaktadır.

<a name="4-basit-bir-streamlit-uygulaması-oluşturma"></a>
## 4. Basit Bir Streamlit Uygulaması Oluşturma
Bir Streamlit uygulaması geliştirmek genellikle `streamlit` kütüphanesini içe aktarmayı ve veri görüntülemek, kullanıcı girdisi toplamak ve görselleştirmeler oluşturmak için işlevlerini kullanmayı içerir. Temel prensip, Streamlit'in web arayüzünü oluşturmak ve güncellemek için yürüttüğü bir Python betiği yazmaya dayanır.

Temel bir Streamlit uygulama yapısı genellikle şunları içerir:
1.  **`streamlit`'i İçe Aktarma**: `import streamlit as st`
2.  **Başlık ve Metin Ekleme**: Bilgi iletmek için `st.title()`, `st.header()`, `st.subheader()`, `st.write()`, `st.markdown()` kullanma.
3.  **Veri Görüntüleme**: Tablosal veriler için `st.dataframe()`, `st.table()` veya ana performans göstergeleri için `st.metric()`.
4.  **Etkileşimli Widget'lar Oluşturma**: Kullanıcı etkileşimine izin vermek için `st.slider()`, `st.selectbox()`, `st.text_input()`, `st.button()`.
5.  **Grafikler Oluşturma**: Çeşitli çizim ihtiyaçları için `st.line_chart()`, `st.bar_chart()`, `st.pyplot()`.

Yürütme akışı basittir: Python betiğinizi kaydedin (örneğin, `app.py`) ve terminalinizden `streamlit run app.py` komutuyla çalıştırın. Streamlit daha sonra uygulamayı web tarayıcınızda açacaktır.

<a name="5-veri-bilimi-iş-akışında-streamlit"></a>
## 5. Veri Bilimi İş Akışında Streamlit
Streamlit, veri bilimi iş akışının çeşitli aşamalarına sorunsuz bir şekilde uyum sağlar, işbirliğini ve etkiyi artırır:

*   **Keşifsel Veri Analizi (EDA):** Veri bilimcileri, veri kümelerini keşfetmek, dağılımları görselleştirmek, korelasyonları tanımlamak ve anormallikleri tespit etmek için hızla etkileşimli kontrol panelleri oluşturabilir. Bu, statik raporların ötesine geçen **dinamik veri keşfine** olanak tanır.
*   **Model Prototipleme ve Değerlendirme:** Makine öğrenimi modelleri etrafında etkileşimli kullanıcı arayüzleri oluşturmak için mükemmel bir platform sağlar. Kullanıcılar parametreler girebilir, gerçek zamanlı tahminleri görebilir ve model performans metriklerini değerlendirebilir, böylece **modelleri yinelemek ve iyileştirmek** kolaylaşır.
*   **Kavram Kanıtı (POC) Gösterimi:** Streamlit, paydaşlara bir veri ürününün veya analitik çözümün potansiyelini göstermek, geliştirme döngüsünün erken aşamalarında geri bildirim ve onay almak için hızlı bir şekilde POC uygulamaları oluşturmak için idealdir.
*   **ML Modellerinin Üretim Dağıtımı:** Flask veya Django gibi tam teşekküllü bir üretim düzeyinde web çerçevesi olmasa da, Streamlit makine öğrenimi modellerini dağıtmak için hafif bir ön uç görevi görebilir ve kullanıcı dostu bir arayüz aracılığıyla tahminlere olanak tanır.
*   **Eğitim Araçları:** Kullanım kolaylığı, veri bilimi kavramlarını öğretmek için **etkileşimli eğitimler ve eğitim araçları** oluşturmak için popüler bir seçimdir.

<a name="6-ileri-düzey-konular-ve-en-iyi-uygulamalar"></a>
## 6. İleri Düzey Konular ve En İyi Uygulamalar
Temel işlevselliğin ötesinde, Streamlit uygulama performansını, ölçeklenebilirliğini ve kullanıcı deneyimini iyileştiren birkaç gelişmiş özellik ve desen sunar:

*   **Oturum Durumu (`st.session_state`):** Bu güçlü özellik, uygulamaların **yeniden çalıştırmalar arasında durumu korumasına** olanak tanır, bu da kullanıcı girdilerinin veya hesaplanan değerlerin kalıcı olması gereken daha karmaşık etkileşimleri mümkün kılar. Çok sayfalı uygulamalar veya karmaşık iş akışlarına sahip uygulamalar oluşturmak için kritik öneme sahiptir.
*   **Önbellekleme Mekanizmaları (`st.cache_data`, `st.cache_resource`):** Belirtildiği gibi, Streamlit tüm betiği otomatik olarak yeniden çalıştırır. Önbellekleme dekore edicileri, girdileri değişmediğinde pahalı işlevlerin veya veri yükleme işlemlerinin yeniden yürütülmesini önleyerek **performansı optimize etmek** için gereklidir.
    *   `st.cache_data`: Veri çerçeveleri, listeler veya sözlükler döndüren işlevler için tipik olarak verileri önbelleğe alır.
    *   `st.cache_resource`: Başlatılması pahalı olan makine öğrenimi modelleri veya veritabanı bağlantıları gibi kaynakları önbelleğe alır.
*   **Düzen ve Konteynerler:** Streamlit, içeriği etkili bir şekilde düzenlemek, duyarlı düzenler oluşturmak ve uygulamaların görsel çekiciliğini artırmak için `st.columns()`, `st.tabs()`, `st.expander()` ve `st.container()` gibi işlevler sunar.
*   **Özel Bileşenler:** Kutudan çıktığı gibi mevcut olmayan yüksek düzeyde özelleştirilmiş kullanıcı arayüzü öğeleri için Streamlit, React kullanılarak oluşturulmuş **özel bileşenlerin** entegrasyonuna izin vererek geliştiricilerin işlevselliğini genişletmelerini sağlar.
*   **Temalandırma:** Uygulamalar, markanın yönergelerine veya kişisel tercihlere uyması için renklerin, yazı tiplerinin ve genel estetiğin ayarlanmasına olanak tanıyan temalandırma seçenekleri aracılığıyla görsel olarak özelleştirilebilir.
*   **Dağıtım Stratejileri:** Üretim için, Docker ile kapsülleme yaygın bir yaklaşımdır, ortam tutarlılığı ve daha kolay ölçeklendirme sağlar. Streamlit Community Cloud, genel uygulamalar için doğrudan, ücretsiz bir dağıtım seçeneği sunar.

<a name="7-kod-örneği-basit-bir-veri-gezgini"></a>
## 7. Kod Örneği: Basit Bir Veri Gezgini
Aşağıdaki Python betiği, kullanıcıların sahte bir veri kümesini keşfetmesine, kategoriye göre filtrelemesine ve değerlerinin bir alt kümesini görselleştirmesine olanak tanıyan basit bir Streamlit uygulamasını göstermektedir.

```python
import streamlit as st
import pandas as pd
import numpy as np

# Daha geniş bir düzen için sayfa yapılandırmasını ayarla
st.set_page_config(layout="wide")

# Uygulamanın ana başlığı
st.title("Basit Veri Keşif Uygulaması")
st.write("Bu uygulama, etkileşimli veri keşfi için temel Streamlit özelliklerini gösterir. "
         "Sentetik veri üretir ve kullanıcıların filtrelemesine ve görselleştirmesine olanak tanır.")

# Gösterim için bazı sahte veriler oluştur
@st.cache_data
def load_data():
    """Sahte verilerle bir DataFrame oluşturur ve döndürür."""
    np.random.seed(42) # tekrar üretilebilirlik için
    data = pd.DataFrame({
        'Kategori': np.random.choice(['Alfa', 'Beta', 'Gama'], 100),
        'Değer': np.random.rand(100) * 100,
        'Zaman Damgası': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    })
    return data

data = load_data()

# Bölüm 1: Veri Önizleme
st.header("1. Veri Önizleme")
st.markdown("Oluşturulan veri kümesine bir göz atın:")
st.dataframe(data.head(10)) # İlk 10 satırı göster

# Bölüm 2: Etkileşimli Filtreleme
st.header("2. Etkileşimli Filtreleme")
st.markdown("Verileri filtrelemek ve belirli alt kümeleri görüntülemek için bir kategori seçin.")

# Kategori seçimi için kenar çubuğunda bir seçim kutusu oluştur
with st.sidebar:
    st.subheader("Filtreleme Seçenekleri")
    selected_category = st.selectbox(
        "Bir Kategori Seçin",
        options=data['Kategori'].unique(),
        help="Ana veri kümesini filtrelemek için bir kategori seçin."
    )

filtered_data = data[data['Kategori'] == selected_category]

st.write(f"Seçilen kategori için veriler gösteriliyor: **{selected_category}**")
st.dataframe(filtered_data)

# Bölüm 3: Temel Grafik Çizimi
st.header("3. Temel Grafik Çizimi")
st.markdown(f"**{selected_category}** için 'Zaman Damgası' üzerinde 'Değer'i gösteren basit bir çizgi grafik:")
st.line_chart(filtered_data.set_index('Zaman Damgası')['Değer'])

# Kenar çubuğu bilgisi
st.sidebar.markdown("---")
st.sidebar.info(
    "Bu uygulama, veri bilimi için etkileşimli web uygulamaları oluşturma yeteneklerini "
    "göstermek amacıyla Streamlit kullanılarak geliştirilmiştir. "
    "Uygulamayı yerel olarak başlatmak için `streamlit run your_app_file.py` komutunu çalıştırın."
)

(Kod örneği bölümünün sonu)
```

<a name="8-sonuç"></a>
## 8. Sonuç
Streamlit, veri odaklı web uygulamalarının oluşturulmasını demokratikleştirmede önemli bir ilerlemeyi temsil etmektedir. Veri bilimcilerinin tamamen Python'da sofistike, etkileşimli kullanıcı arayüzleri oluşturmasına olanak tanıyarak, içgörüleri paylaşma, modelleri dağıtma ve işbirliğini teşvik etme sürecini önemli ölçüde hızlandırır. Basitliğe, hızlı geliştirmeye ve zengin bileşen setine odaklanması, veri bilimi projelerini statik analizlerden dinamik, ilgi çekici ve erişilebilir web deneyimlerine dönüştürmek isteyen herkes için onu vazgeçilmez bir araç haline getirmektedir. Açıklanabilir yapay zeka ve etkileşimli veri ürünlerine olan talep artmaya devam ettikçe, Streamlit, veri profesyonellerini çalışmalarını daha geniş bir kitleye etkili bir şekilde iletmeleri için güçlendiren araçlar arasında ön saflarda yer almaya hazırlanmaktadır.



