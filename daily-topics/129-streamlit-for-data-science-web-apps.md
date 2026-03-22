# Streamlit for Data Science Web Apps

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts of Streamlit](#2-core-concepts-of-streamlit)
    - [2.1. Declarative API](#21-declarative-api)
    - [2.2. Script Re-execution Model](#22-script-re-execution-model)
    - [2.3. Widgets and Interactivity](#23-widgets-and-interactivity)
- [3. Key Features for Data Scientists](#3-key-features-for-data-scientists)
    - [3.1. Data Visualization and Display](#31-data-visualization-and-display)
    - [3.2. Advanced Layouts and Containers](#32-advanced-layouts-and-containers)
    - [3.3. Performance Optimization with Caching](#33-performance-optimization-with-caching)
    - [3.4. Component Ecosystem](#34-component-ecosystem)
    - [3.5. Deployment Options](#35-deployment-options)
- [4. Code Example: Simple Interactive Streamlit App](#4-code-example-simple-interactive-streamlit-app)
- [5. Conclusion](#5-conclusion)

<br>

<a name="1-introduction"></a>
## 1. Introduction

In the rapidly evolving landscape of data science and machine learning, the ability to transform complex analytical insights into accessible, interactive web applications is paramount. While traditional web development often requires extensive knowledge of front-end technologies like HTML, CSS, and JavaScript, data scientists frequently lack this specialized expertise, leading to a significant barrier in deploying their models and analyses. **Streamlit** emerges as a revolutionary open-source Python library specifically designed to bridge this gap. It empowers data scientists and machine learning engineers to create beautiful, custom web applications for data exploration, model demonstration, and interactive dashboards using pure Python, with minimal effort and no prior web development experience.

Streamlit operates on the philosophy of simplicity and speed, allowing users to turn data scripts into shareable web apps within minutes. Its intuitive API abstracts away the complexities of web development, enabling a singular focus on the underlying data logic and computational tasks. This ease of use fosters rapid prototyping and iteration, making it an invaluable tool for presenting findings, gathering feedback, and democratizing access to data-driven tools within organizations. By providing a streamlined pathway from script to app, Streamlit has fundamentally altered how data professionals can communicate and interact with their work, making sophisticated data products more attainable than ever before.

<a name="2-core-concepts-of-streamlit"></a>
## 2. Core Concepts of Streamlit

Understanding Streamlit's foundational principles is crucial for effectively developing interactive data applications. Its design is based on several innovative concepts that prioritize ease of use and performance for data scientists.

<a name="21-declarative-api"></a>
### 2.1. Declarative API

At the heart of Streamlit lies its **declarative API**. Instead of defining the sequence of operations for rendering a web page, users declare *what* they want to see, and Streamlit handles the *how*. This means that UI elements such as text, data tables, charts, and widgets are added to the application simply by calling Streamlit functions (e.g., `st.write()`, `st.dataframe()`, `st.slider()`). Each call adds an element to the application in the order it appears in the Python script. This approach allows developers to write straightforward, top-to-bottom Python scripts that automatically translate into interactive web pages, eliminating the need for separate frontend and backend logic. The simplicity of this model significantly reduces the learning curve for data professionals accustomed to writing sequential scripts.

<a name="22-script-re-execution-model"></a>
### 2.2. Script Re-execution Model

A cornerstone of Streamlit's interactivity is its **script re-execution model**. Unlike traditional web frameworks that maintain state across HTTP requests, Streamlit re-runs the entire Python script from top to bottom every time a user interacts with a widget or modifies the application's state. While this might initially seem inefficient, Streamlit is heavily optimized to make this model performant. When an input widget's value changes, Streamlit automatically detects this change, re-executes the script, and updates only the necessary parts of the web page.

This paradigm offers several advantages:
*   **Simplicity**: Developers don't need to manage explicit callbacks or complex state machines. The script itself defines the UI and its behavior.
*   **Predictability**: The application's state is always determined by the current values of its inputs and the execution of the script, making debugging easier.
*   **Pythonic**: It feels natural for Python developers who are used to writing scripts that run sequentially.

To mitigate performance issues associated with re-running potentially expensive data loading or model inference operations, Streamlit provides robust **caching mechanisms**, which will be discussed in a later section.

<a name="23-widgets-and-interactivity"></a>
### 2.3. Widgets and Interactivity

**Widgets** are the interactive building blocks of Streamlit applications, allowing users to input data, select options, and trigger actions. Streamlit offers a rich collection of built-in widgets, including:
*   `st.button()`: For triggering actions.
*   `st.slider()`: For selecting numerical ranges.
*   `st.text_input()`: For entering text.
*   `st.selectbox()`, `st.multiselect()`: For choosing from lists.
*   `st.checkbox()`: For boolean inputs.
*   `st.radio()`: For mutually exclusive options.
*   `st.file_uploader()`: For uploading files.

Each widget returns its current value, which can then be used in the subsequent logic of the Python script. When a user interacts with a widget, its value changes, prompting Streamlit's script re-execution model to refresh the application, incorporating the new input into the display or computation. This seamless integration of user input with script execution makes creating dynamic and responsive data applications remarkably straightforward.

<a name="3-key-features-for-data-scientists"></a>
## 3. Key Features for Data Scientists

Streamlit offers a plethora of features specifically tailored to the needs of data scientists, enabling them to build powerful and engaging applications with minimal effort.

<a name="31-data-visualization-and-display"></a>
### 3.1. Data Visualization and Display

Displaying data effectively is fundamental to any data science application. Streamlit provides first-class support for rendering various forms of data and visualizations:
*   **Tabular Data**: Functions like `st.dataframe()` and `st.table()` allow for displaying pandas DataFrames with interactive sorting and search capabilities, or static tables, respectively.
*   **Charts and Plots**: Streamlit integrates seamlessly with popular Python visualization libraries. Users can render plots from **Matplotlib**, **Seaborn**, **Altair**, **Plotly**, **Bokeh**, and **Vega-Lite** directly using functions like `st.pyplot()`, `st.altair_chart()`, `st.plotly_chart()`, etc. This enables data scientists to leverage their existing visualization codebases without modification.
*   **Media**: Beyond numerical and categorical data, Streamlit supports displaying images (`st.image()`), audio (`st.audio()`), and video (`st.video()`), which is crucial for applications involving multimedia data or model outputs like computer vision results.
*   **Text and Markdown**: `st.write()`, `st.markdown()`, `st.header()`, `st.subheader()`, and `st.caption()` provide flexible ways to add explanatory text, titles, and rich content using Markdown syntax, making it easy to create well-documented applications.

<a name="32-advanced Layouts and Containers"></a>
### 3.2. Advanced Layouts and Containers

While Streamlit's default layout is sequential, it offers powerful tools for structuring complex applications and enhancing user experience:
*   **Sidebar (`st.sidebar`)**: A dedicated area on the left side of the application, ideal for placing input widgets, navigation links, or configuration options that are less central to the main content. All Streamlit elements can be placed within the sidebar.
*   **Columns (`st.columns`)**: Allows for creating horizontal divisions within the main content area, enabling parallel display of text, data, or charts. This is invaluable for comparing different views of data or organizing dashboards.
*   **Containers (`st.container`)**: Provides a way to group Streamlit elements together. This is particularly useful when combined with conditional logic to display or hide entire sections of an application.
*   **Expanders (`st.expander`)**: Creates a collapsible section, perfect for hiding detailed explanations, optional configurations, or complex output that users might only want to view on demand, thereby decluttering the UI.
*   **Tabs (`st.tabs`)**: Organizes content into distinct, selectable tabs, allowing for multi-page-like navigation within a single Streamlit page, ideal for multi-faceted dashboards or distinct analysis stages.

<a name="33-performance-optimization-with-caching"></a>
### 3.3. Performance Optimization with Caching

Given Streamlit's script re-execution model, efficiently handling expensive computations (e.g., loading large datasets, training machine learning models, performing complex simulations) is critical. Streamlit addresses this through its intelligent **caching mechanisms**:
*   `st.cache_data()`: Decorator designed for functions that return data (e.g., loading CSVs, querying databases, running ETL pipelines). It stores the function's output in a local cache (in-memory or on disk) based on its input parameters and the function's source code. If the function is called again with the same parameters and the code hasn't changed, Streamlit retrieves the result from the cache instead of re-executing the function.
*   `st.cache_resource()`: Similar to `st.cache_data()` but optimized for resources like database connections, machine learning models, or network clients that should be loaded only once and reused across different sessions. This prevents expensive re-initialization of resource-intensive objects.

These caching decorators are indispensable for building performant Streamlit applications, drastically reducing loading times and improving responsiveness, especially in data-intensive scenarios.

<a name="34-component-ecosystem"></a>
### 3.4. Component Ecosystem

Beyond its core functionalities, Streamlit boasts a vibrant **component ecosystem**. This allows developers to extend Streamlit's capabilities by integrating custom components built with HTML, CSS, and JavaScript. The `st.components.v1.html()` function enables embedding raw HTML, while more complex interactive components can be created and shared, ranging from custom charts to specialized input widgets. This extensibility ensures that Streamlit can adapt to virtually any niche requirement, fostering a community-driven development approach.

<a name="35-deployment-options"></a>
### 3.5. Deployment Options

Once a Streamlit application is developed, several options are available for deployment, making it accessible to a wider audience:
*   **Streamlit Community Cloud**: The easiest way to deploy Streamlit apps. It provides free hosting for public repositories, continuous deployment from GitHub, and basic scaling.
*   **On-Premise/Cloud VMs**: Apps can be deployed on any server (e.g., AWS EC2, Google Cloud, Azure VM) by installing Python and Streamlit and running the app as a background process, often using tools like `screen` or `systemd`.
*   **Docker Containers**: For more robust and scalable deployments, Streamlit apps can be containerized using Docker. This ensures environment consistency and simplifies deployment to container orchestration platforms like Kubernetes or services like AWS Fargate.
*   **Heroku/Render/etc.**: Various PaaS (Platform as a Service) providers also offer straightforward deployment paths for Streamlit applications, often requiring just a `requirements.txt` and a `Procfile`.

These diverse deployment options provide flexibility, catering to different scales, budgets, and operational requirements.

<a name="4-code-example-simple-interactive-streamlit-app"></a>
## 4. Code Example: Simple Interactive Streamlit App

This example demonstrates a basic Streamlit application that takes a user's name and age, then displays a personalized greeting and calculates the year they will turn 100.

```python
import streamlit as st
import datetime

# Set the title of the Streamlit application
st.title("Simple Interactive Streamlit App")
st.write("Enter your details below to see a personalized message.")

# Input widget for user's name
user_name = st.text_input("What's your name?", "Guest")

# Input widget for user's age
# A slider is a good choice for numerical input within a range
user_age = st.slider("How old are you?", 0, 120, 25) # min, max, default value

# Display a personalized greeting
st.write(f"Hello, **{user_name}**!")

# Calculate the year the user will turn 100
current_year = datetime.datetime.now().year
year_turn_100 = current_year + (100 - user_age)

# Display the calculated year
st.success(f"You will turn 100 in the year **{year_turn_100}**.")

# Optional: Add a checkbox to show more details
if st.checkbox("Show more details"):
    st.info(f"Today's date: {datetime.date.today()}")
    st.info(f"Your entered age is: {user_age}")

st.markdown("---")
st.caption("This is a simple demonstration of Streamlit's interactive widgets and display capabilities.")

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Streamlit has rapidly emerged as a game-changer for data scientists and machine learning practitioners, fundamentally simplifying the process of transforming data scripts into dynamic, interactive web applications. By abstracting away the complexities of traditional web development and focusing on a **Python-native, declarative API**, it empowers users to build sophisticated data dashboards, machine learning model UIs, and data exploration tools with unprecedented speed and ease. The combination of its intuitive **script re-execution model**, rich collection of **widgets**, powerful **data visualization capabilities**, flexible **layout options**, and essential **caching mechanisms** makes it an indispensable tool in the modern data science workflow.

Furthermore, its vibrant **component ecosystem** and diverse **deployment options** ensure that Streamlit applications are not only easy to build but also highly extensible and readily shareable. As the demand for interactive data products continues to grow, Streamlit stands out as a leading solution that democratizes app development for the data community, enabling greater collaboration, faster insights, and broader accessibility to data-driven innovations. Its continued evolution promises to further solidify its position as a cornerstone technology for operationalizing data science.

---
<br>

<a name="türkçe-içerik"></a>
## Streamlit ile Veri Bilimi Web Uygulamaları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Streamlit'in Temel Kavramları](#2-streamlitin-temel-kavramları)
    - [2.1. Bildirimsel API](#21-bildirimsel-api)
    - [2.2. Komut Dosyası Yeniden Yürütme Modeli](#22-komut-dosyası-yeniden-yürütme-modeli)
    - [2.3. Widget'lar ve Etkileşim](#23-widgetlar-ve-etkileşim)
- [3. Veri Bilimcileri için Temel Özellikler](#3-veri-bilimcileri-için-temel-özellikler)
    - [3.1. Veri Görselleştirme ve Görüntüleme](#31-veri-görselleştirme-ve-görüntüleme)
    - [3.2. Gelişmiş Düzenler ve Kapsayıcılar](#32-gelişmiş-düzenler-ve-kapsayıcılar)
    - [3.3. Önbellekleme ile Performans Optimizasyonu](#33-önbellekleme-ile-performans-optimizasyonu)
    - [3.4. Bileşen Ekosistemi](#34-bileşen-ekosistemi)
    - [3.5. Dağıtım Seçenekleri](#35-dağıtım-seçenekleri)
- [4. Kod Örneği: Basit Etkileşimli Streamlit Uygulaması](#4-kod-örneği-basit-etkileşimli-streamlit-uygulaması)
- [5. Sonuç](#5-sonuç)

<br>

<a name="1-giriş"></a>
## 1. Giriş

Veri bilimi ve makine öğrenimi alanındaki hızlı gelişen ortamda, karmaşık analitik içgörüleri erişilebilir, etkileşimli web uygulamalarına dönüştürme yeteneği hayati öneme sahiptir. Geleneksel web geliştirme genellikle HTML, CSS ve JavaScript gibi ön uç teknolojileri hakkında kapsamlı bilgi gerektirirken, veri bilimcileri genellikle bu özel uzmanlıktan yoksundur. Bu durum, modellerini ve analizlerini dağıtmada önemli bir engel oluşturur. **Streamlit**, bu boşluğu doldurmak için özel olarak tasarlanmış devrim niteliğinde açık kaynaklı bir Python kütüphanesi olarak ortaya çıkmıştır. Veri bilimcileri ve makine öğrenimi mühendislerinin, saf Python kullanarak, minimum çabayla ve önceden web geliştirme deneyimi olmadan veri keşfi, model tanıtımı ve etkileşimli panolar için güzel, özel web uygulamaları oluşturmasını sağlar.

Streamlit, sadelik ve hız felsefesi üzerine kuruludur. Kullanıcıların veri komut dosyalarını dakikalar içinde paylaşılabilir web uygulamalarına dönüştürmesine olanak tanır. Sezgisel API'si, web geliştirmenin karmaşıklıklarını soyutlayarak, yalnızca temel veri mantığına ve hesaplama görevlerine odaklanmayı mümkün kılar. Bu kullanım kolaylığı, hızlı prototiplemeyi ve yinelemeyi teşvik ederek, bulguları sunmak, geri bildirim toplamak ve kuruluşlar içinde veri odaklı araçlara erişimi demokratikleştirmek için paha biçilmez bir araç haline getirir. Streamlit, komut dosyasından uygulamaya kadar uzanan akıcı bir yol sunarak, veri profesyonellerinin çalışmalarıyla iletişim kurma ve etkileşimde bulunma biçimini temelden değiştirmiş, sofistike veri ürünlerini her zamankinden daha ulaşılabilir kılmıştır.

<a name="2-streamlitin-temel-kavramları"></a>
## 2. Streamlit'in Temel Kavramları

Etkileşimli veri uygulamalarını etkili bir şekilde geliştirmek için Streamlit'in temel ilkelerini anlamak çok önemlidir. Tasarımı, veri bilimcileri için kullanım kolaylığına ve performansa öncelik veren çeşitli yenilikçi kavramlara dayanmaktadır.

<a name="21-bildirimsel-api"></a>
### 2.1. Bildirimsel API

Streamlit'in kalbinde **bildirimsel API**'si yatar. Bir web sayfasının işlenmesi için işlemler dizisini tanımlamak yerine, kullanıcılar *ne* görmek istediklerini beyan eder ve Streamlit *nasıl* yapılacağını halleder. Bu, metin, veri tabloları, grafikler ve widget'lar gibi kullanıcı arayüzü öğelerinin sadece Streamlit fonksiyonları çağrılarak uygulamaya eklendiği anlamına gelir (örn. `st.write()`, `st.dataframe()`, `st.slider()`). Her çağrı, Python betiğinde göründüğü sıraya göre uygulamaya bir öğe ekler. Bu yaklaşım, geliştiricilerin doğrudan, yukarıdan aşağıya doğru çalışan Python betikleri yazmasına olanak tanır ve bu betikler otomatik olarak etkileşimli web sayfalarına dönüşür, böylece ayrı bir ön uç ve arka uç mantığına gerek kalmaz. Bu modelin basitliği, sıralı betikler yazmaya alışkın veri profesyonelleri için öğrenme eğrisini önemli ölçüde azaltır.

<a name="22-komut-dosyası-yeniden-yürütme-modeli"></a>
### 2.2. Komut Dosyası Yeniden Yürütme Modeli

Streamlit'in etkileşimliliğinin temel taşlarından biri, **komut dosyası yeniden yürütme modelidir**. HTTP istekleri arasında durumu koruyan geleneksel web çerçevelerinin aksine, Streamlit, bir kullanıcı bir widget ile her etkileşimde bulunduğunda veya uygulamanın durumunu her değiştirdiğinde tüm Python komut dosyasını baştan sona yeniden çalıştırır. Başlangıçta verimsiz gibi görünse de, Streamlit bu modeli performanslı hale getirmek için yoğun bir şekilde optimize edilmiştir. Bir girdi widget'ının değeri değiştiğinde, Streamlit bu değişikliği otomatik olarak algılar, komut dosyasını yeniden yürütür ve web sayfasının yalnızca gerekli kısımlarını günceller.

Bu paradigma çeşitli avantajlar sunar:
*   **Sadelik**: Geliştiricilerin açık geri çağrıları veya karmaşık durum makinelerini yönetmesine gerek yoktur. Komut dosyası, kullanıcı arayüzünü ve davranışını kendisi tanımlar.
*   **Öngörülebilirlik**: Uygulamanın durumu her zaman girdilerinin mevcut değerleri ve komut dosyasının yürütülmesiyle belirlenir, bu da hata ayıklamayı kolaylaştırır.
*   **Pythonic**: Sıralı çalışan betikler yazmaya alışkın Python geliştiricileri için doğal hissettirir.

Potansiyel olarak maliyetli veri yükleme veya model çıkarım işlemlerinin yeniden çalıştırılmasıyla ilişkili performans sorunlarını azaltmak için Streamlit, daha sonraki bir bölümde ele alınacak olan sağlam **önbellekleme mekanizmaları** sunar.

<a name="23-widgetlar-ve-etkileşim"></a>
### 2.3. Widget'lar ve Etkileşim

**Widget'lar**, Streamlit uygulamalarının etkileşimli yapı taşlarıdır ve kullanıcıların veri girmesine, seçenekleri seçmesine ve eylemleri tetiklemesine olanak tanır. Streamlit, zengin bir yerleşik widget koleksiyonu sunar:
*   `st.button()`: Eylemleri tetiklemek için.
*   `st.slider()`: Sayısal aralıkları seçmek için.
*   `st.text_input()`: Metin girmek için.
*   `st.selectbox()`, `st.multiselect()`: Listelerden seçim yapmak için.
*   `st.checkbox()`: Boolean girdiler için.
*   `st.radio()`: Birbirini dışlayan seçenekler için.
*   `st.file_uploader()`: Dosya yüklemek için.

Her widget, geçerli değerini döndürür ve bu değer daha sonra Python betiğinin sonraki mantığında kullanılabilir. Bir kullanıcı bir widget ile etkileşime girdiğinde, değeri değişir ve Streamlit'in komut dosyası yeniden yürütme modelini uygulamayı yenilemeye yönlendirerek yeni girdiyi ekrana veya hesaplamaya dahil eder. Kullanıcı girdisinin komut dosyası yürütmeyle bu kesintisiz entegrasyonu, dinamik ve duyarlı veri uygulamaları oluşturmayı oldukça basit hale getirir.

<a name="3-veri-bilimcileri-için-temel-özellikler"></a>
## 3. Veri Bilimcileri için Temel Özellikler

Streamlit, veri bilimcilerinin ihtiyaçlarına özel olarak tasarlanmış sayısız özellik sunarak, güçlü ve ilgi çekici uygulamaları minimum çabayla oluşturmalarını sağlar.

<a name="31-veri-görselleştirme-ve-görüntüleme"></a>
### 3.1. Veri Görselleştirme ve Görüntüleme

Veriyi etkili bir şekilde görüntülemek, herhangi bir veri bilimi uygulamasının temelidir. Streamlit, çeşitli veri ve görselleştirme biçimlerini işlemek için birinci sınıf destek sağlar:
*   **Tablo Verileri**: `st.dataframe()` ve `st.table()` gibi fonksiyonlar, sırasıyla etkileşimli sıralama ve arama özelliklerine sahip pandas DataFrame'lerini veya statik tabloları görüntülemeye olanak tanır.
*   **Grafikler ve Çizimler**: Streamlit, popüler Python görselleştirme kütüphaneleriyle sorunsuz bir şekilde entegre olur. Kullanıcılar, **Matplotlib**, **Seaborn**, **Altair**, **Plotly**, **Bokeh** ve **Vega-Lite** gibi kütüphanelerden çizimleri doğrudan `st.pyplot()`, `st.altair_chart()`, `st.plotly_chart()` vb. fonksiyonları kullanarak render edebilirler. Bu, veri bilimcilerinin mevcut görselleştirme kod tabanlarını değiştirmeden kullanmalarını sağlar.
*   **Medya**: Sayısal ve kategorik verilerin ötesinde, Streamlit, bilgisayar görüşü sonuçları gibi multimedya verileri veya model çıktılarını içeren uygulamalar için çok önemli olan görüntüleri (`st.image()`), sesleri (`st.audio()`) ve videoları (`st.video()`) görüntülemeyi destekler.
*   **Metin ve Markdown**: `st.write()`, `st.markdown()`, `st.header()`, `st.subheader()` ve `st.caption()` fonksiyonları, Markdown sözdizimi kullanarak açıklayıcı metin, başlıklar ve zengin içerik eklemenin esnek yollarını sunarak iyi belgelenmiş uygulamalar oluşturmayı kolaylaştırır.

<a name="32-gelişmiş-düzenler-ve-kapsayıcılar"></a>
### 3.2. Gelişmiş Düzenler ve Kapsayıcılar

Streamlit'in varsayılan düzeni sıralı olsa da, karmaşık uygulamaları yapılandırmak ve kullanıcı deneyimini geliştirmek için güçlü araçlar sunar:
*   **Yan Çubuk (`st.sidebar`)**: Uygulamanın sol tarafında özel bir alan olup, ana içeriğe daha az merkezi olan girdi widget'ları, gezinme bağlantıları veya yapılandırma seçenekleri için idealdir. Tüm Streamlit öğeleri yan çubuğa yerleştirilebilir.
*   **Sütunlar (`st.columns`)**: Ana içerik alanında yatay bölümler oluşturmaya olanak tanır, metin, veri veya grafikleri paralel olarak görüntülemeyi sağlar. Bu, farklı veri görünümlerini karşılaştırmak veya panoları düzenlemek için paha biçilmezdir.
*   **Kapsayıcılar (`st.container`)**: Streamlit öğelerini bir arada gruplamanın bir yolunu sunar. Bu, özellikle bir uygulamanın tüm bölümlerini görüntülemek veya gizlemek için koşullu mantıkla birleştirildiğinde kullanışlıdır.
*   **Genişleticiler (`st.expander`)**: Daraltılabilir bir bölüm oluşturur, ayrıntılı açıklamaları, isteğe bağlı yapılandırmaları veya kullanıcıların yalnızca talep üzerine görüntülemek isteyebileceği karmaşık çıktıları gizlemek için mükemmeldir, böylece kullanıcı arayüzünü düzenler.
*   **Sekmeler (`st.tabs`)**: İçeriği ayrı, seçilebilir sekmelerde düzenleyerek, tek bir Streamlit sayfası içinde çok sayfalı benzeri gezinmeye olanak tanır, çok yönlü panolar veya ayrı analiz aşamaları için idealdir.

<a name="33-önbellekleme-ile-performans-optimizasyonu"></a>
### 3.3. Önbellekleme ile Performans Optimizasyonu

Streamlit'in komut dosyası yeniden yürütme modeli göz önüne alındığında, maliyetli hesaplamaların (örn. büyük veri kümelerini yükleme, makine öğrenimi modellerini eğitme, karmaşık simülasyonlar gerçekleştirme) verimli bir şekilde ele alınması kritik öneme sahiptir. Streamlit, bu durumu akıllı **önbellekleme mekanizmaları** aracılığıyla çözer:
*   `st.cache_data()`: Veri döndüren fonksiyonlar (örn. CSV'leri yükleme, veritabanlarını sorgulama, ETL işlem hatları çalıştırma) için tasarlanmış bir dekoratördür. Fonksiyonun çıktısını, girdi parametrelerine ve fonksiyonun kaynak koduna göre yerel bir önbellekte (bellekte veya diskte) saklar. Fonksiyon aynı parametrelerle tekrar çağrılırsa ve kod değişmediyse, Streamlit fonksiyonu yeniden yürütmek yerine sonucu önbellekten alır.
*   `st.cache_resource()`: `st.cache_data()`'ya benzer ancak veritabanı bağlantıları, makine öğrenimi modelleri veya yalnızca bir kez yüklenmesi ve farklı oturumlar arasında yeniden kullanılması gereken ağ istemcileri gibi kaynaklar için optimize edilmiştir. Bu, kaynak yoğun nesnelerin maliyetli yeniden başlatılmasını önler.

Bu önbellekleme dekoratörleri, özellikle veri yoğun senaryolarda yükleme sürelerini önemli ölçüde azaltarak ve yanıt verme hızını artırarak, performanslı Streamlit uygulamaları oluşturmak için vazgeçilmezdir.

<a name="34-bileşen-ekosistemi"></a>
### 3.4. Bileşen Ekosistemi

Temel işlevlerinin ötesinde, Streamlit canlı bir **bileşen ekosistemine** sahiptir. Bu, geliştiricilerin HTML, CSS ve JavaScript ile oluşturulmuş özel bileşenleri entegre ederek Streamlit'in yeteneklerini genişletmesine olanak tanır. `st.components.v1.html()` fonksiyonu, ham HTML'yi gömmeyi sağlarken, özel grafiklerden uzmanlaşmış girdi widget'larına kadar daha karmaşık etkileşimli bileşenler oluşturulabilir ve paylaşılabilir. Bu genişletilebilirlik, Streamlit'in neredeyse her niş gereksinime uyum sağlayabilmesini sağlayarak topluluk odaklı bir geliştirme yaklaşımını teşvik eder.

<a name="35-dağıtım-seçenekleri"></a>
### 3.5. Dağıtım Seçenekleri

Bir Streamlit uygulaması geliştirildikten sonra, daha geniş bir kitleye erişilebilir hale getirmek için çeşitli dağıtım seçenekleri mevcuttur:
*   **Streamlit Community Cloud**: Streamlit uygulamalarını dağıtmanın en kolay yolu. Genel depolar için ücretsiz barındırma, GitHub'dan sürekli dağıtım ve temel ölçeklendirme sağlar.
*   **Şirket İçi/Bulut Sanal Makineler**: Uygulamalar, herhangi bir sunucuya (örn. AWS EC2, Google Cloud, Azure VM) Python ve Streamlit yüklenerek ve uygulamayı `screen` veya `systemd` gibi araçlar kullanarak bir arka plan süreci olarak çalıştırarak dağıtılabilir.
*   **Docker Konteynerleri**: Daha sağlam ve ölçeklenebilir dağıtımlar için Streamlit uygulamaları Docker kullanılarak konteynerize edilebilir. Bu, ortam tutarlılığını sağlar ve Kubernetes veya AWS Fargate gibi konteyner orkestrasyon platformlarına dağıtımı basitleştirir.
*   **Heroku/Render/vb.**: Çeşitli PaaS (Hizmet Olarak Platform) sağlayıcıları da Streamlit uygulamaları için genellikle yalnızca bir `requirements.txt` ve bir `Procfile` gerektiren basit dağıtım yolları sunar.

Bu çeşitli dağıtım seçenekleri, farklı ölçeklere, bütçelere ve operasyonel gereksinimlere hitap eden esneklik sağlar.

<a name="4-kod-örneği-basit-etkileşimli-streamlit-uygulaması"></a>
## 4. Kod Örneği: Basit Etkileşimli Streamlit Uygulaması

Bu örnek, kullanıcının adını ve yaşını alan, ardından kişiselleştirilmiş bir selamlama görüntüleyen ve kullanıcının 100 yaşına gireceği yılı hesaplayan basit bir Streamlit uygulamasını göstermektedir.

```python
import streamlit as st
import datetime

# Streamlit uygulamasının başlığını ayarla
st.title("Basit Etkileşimli Streamlit Uygulaması")
st.write("Kişiselleştirilmiş bir mesaj görmek için bilgilerinizi aşağıya girin.")

# Kullanıcının adı için girdi widget'ı
user_name = st.text_input("Adınız nedir?", "Misafir")

# Kullanıcının yaşı için girdi widget'ı
# Bir sayısal girdi için kaydırıcı iyi bir seçimdir
user_age = st.slider("Kaç yaşındasınız?", 0, 120, 25) # min, max, varsayılan değer

# Kişiselleştirilmiş bir selamlama görüntüle
st.write(f"Merhaba, **{user_name}**!")

# Kullanıcının 100 yaşına gireceği yılı hesapla
current_year = datetime.datetime.now().year
year_turn_100 = current_year + (100 - user_age)

# Hesaplanan yılı görüntüle
st.success(f"**{year_turn_100}** yılında 100 yaşına gireceksiniz.")

# İsteğe bağlı: Daha fazla ayrıntı göstermek için bir onay kutusu ekle
if st.checkbox("Daha fazla ayrıntı göster"):
    st.info(f"Bugünün tarihi: {datetime.date.today()}")
    st.info(f"Girdiğiniz yaş: {user_age}")

st.markdown("---")
st.caption("Bu, Streamlit'in etkileşimli widget'larının ve görüntüleme yeteneklerinin basit bir gösterimidir.")

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Streamlit, veri bilimcileri ve makine öğrenimi uzmanları için hızla oyun değiştirici haline gelmiş, veri komut dosyalarını dinamik, etkileşimli web uygulamalarına dönüştürme sürecini temelden basitleştirmiştir. Geleneksel web geliştirmenin karmaşıklıklarını soyutlayarak ve **Python tabanlı, bildirimsel bir API**'ye odaklanarak, kullanıcıların sofistike veri panoları, makine öğrenimi modeli kullanıcı arayüzleri ve veri keşif araçlarını eşi benzeri görülmemiş bir hız ve kolaylıkla oluşturmalarını sağlar. Sezgisel **komut dosyası yeniden yürütme modeli**, zengin **widget** koleksiyonu, güçlü **veri görselleştirme yetenekleri**, esnek **düzen seçenekleri** ve temel **önbellekleme mekanizmalarının** birleşimi, onu modern veri bilimi iş akışında vazgeçilmez bir araç haline getirir.

Ayrıca, canlı **bileşen ekosistemi** ve çeşitli **dağıtım seçenekleri**, Streamlit uygulamalarının yalnızca inşa edilmesinin kolay olmasını değil, aynı zamanda yüksek düzeyde genişletilebilir ve kolayca paylaşılabilir olmasını da sağlar. Etkileşimli veri ürünlerine olan talep artmaya devam ederken, Streamlit, veri topluluğu için uygulama geliştirmeyi demokratikleştiren, daha fazla işbirliği, daha hızlı içgörüler ve veri odaklı yeniliklere daha geniş erişilebilirlik sağlayan önde gelen bir çözüm olarak öne çıkmaktadır. Sürekli evrimi, veri bilimini operasyonelleştirmek için temel bir teknoloji olarak konumunu daha da sağlamlaştırmayı vaat etmektedir.