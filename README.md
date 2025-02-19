# Data_Processing_and_Machine_Learning
Портфолио выполненных работ по ML и DS

## Data Science / Classic Machine Learning

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[00. Введение в NumPy](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/00_NumPy_intro.ipynb)|Знакомство с основными методами NumPy.|**Анализ данных**|*Python*, *NumPy*|
|[01. Введение в Pandas](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/01_Pandas_intro.ipynb)|Знакомство с основными методами Pandas.|**Анализ данных**|*Python*, *Pandas*|
|[03. Разведочный анализ данных](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/03_Exploratory_Data_Analysis.ipynb)|Проведение разведочного анализа данных и построение различных графиков.|**Анализ данных**|*Python*, *Pandas*, *Matplotlib*, *Seaborn*, *Plotly*|
|[1. Предсказание оттока клиентов](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/1_Prediction_of_Subscriber_Outflow.ipynb)|Для каждого абонента предсказать колонку Leave_Next_Month, означающую, уйдет ли клиент в следующем месяце. Значение должно быть либо 0 - абонент точно не уйдет в следующем месяце, либо 1 - абонент точно уйдет в следующем месяце. Сравнить LightGBM, XGBoost, CatBoost.|**Машинное обучение**, **Бинарная классификация**|*Python*, *Pandas*, *Matplotlib*, *sScikit-learn*, *LightGBM*, *XGBoost*, *CatBoost*|
|[2. Предсказание погоды](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/2_Weather_Classification.ipynb)|Предсказываем погоду различными методами. Тестируем классификаторы из Scikit-learn.|**Машинное обучение**, **Классификация**|*Python*, *Pandas*, *Numpy*, *Matplotlib*, *Scikit-learn*, *CatBoost*|
|[3. Классификация впечатлений от полета](https://github.com/goralex02/Flight_satisfaction)|По результатам послеполетной анкеты выясняем, понравился ли пассажиру полет, и с какой вероятностью. Используем веб-интерфейс на Streamlit (ссылка на приложение в репозитории).|**Машинное обучение**, **Бинарная классификация**, **Веб-интерфейс**|*Python*, *Pandas*, *Matplotlib*, *Scikit-learn*, *Pyplot*, *Seaborn*, *Streamlit*|
|[4. Предсказание года выхода песни](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/4_PyTorch_Regression_MLP.ipynb)|Предсказание года выхода песни с помощью полносвязной нейронной сети на PyTorch.|**Глубинное обучение**, **Классификация**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *PyTorch*|

## Computer Vision

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[5. Классификация изображений с дообучением](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/5_PyTorch_Image_Classification_DenseNet.ipynb)|Классификация изображений с помощью DenseNet121. Обучение модели с нуля, дообучение головы, fine-tuning всей модели.|**Глубинное обучение**, **Компьютерное зрение**, **Классификация**, **Дообучение**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *PyTorch*, *Сверточные нейронные сети*, *DenseNet*|
|[6. Детекция игральных карт с YOLO](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/6_YOLO_Cards_Detector.ipynb)|Детекция игральных карт от девятки до туза самописным и готовым решением YOLO.|**Глубинное обучение**, **Компьютерное зрение**, **Детекция**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *PyTorch*, *TorchVision*, *Ultralytics*, *YOLOv11*|

## Audio processing

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[7. Классификация звуков по спектрограмме](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/7_Sounds_Classification.ipynb)|Классификация звуковых файлов по их спектрограмме.|**Глубинное обучение**, **Компьютерное зрение**, **Классификация**, **Обработка звука**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *PyTorch*, *TorchVision*, *librosa*, *Сверточные нейронные сети*, *EfficientNet-B3*|
|[8. Оценка качества синтеза модели Parler-TTS-mini-jenny-30H](https://github.com/goralex02/parler-tts-mini-jenny-30h_evaluation/tree/main)|У датасета Jenny из коробки нет разделения на тренировочный и тестовый датасет, необходимо самостоятельно определить сетап для оценки модели: аудиозаписи, метрики, текстовое описание для генерации и тд.|**Машинное обучение**, **Обработка звука**, **ParlerTTS**|*Python*, *PyTorch*, *TorchAudio*, *Transformers*, *librosa*, *soundfile*, *datasets*, *Matplotlib*|

## Natural Language Processing

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[9. Кластеризация новостных статей](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/9_Text_Clustering.ipynb)|Кластеризация новостных статей различными методами.|**Машинное обучение**, **Обработка естественного языка**, **Кластеризация**, **Визуализация**, **Бинарная классификация**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *Scikit-learn*, *KMeans*, *DBSCAN*, *t-SNE*, *Spectral/Aglomerative Clustering*|
|[10. Классификация новостных статей](https://github.com/goralex02/Data_Processing_and_Machine_Learning/blob/main/10_Text_Classification.ipynb)|Сравнение алгоритмов классификации новостных статей.|**Машинное обучение**, **Обработка естественного языка**, **Классификация**, **Глубинное обучение**, **Transformers**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *Scikit-learn*, *PyTorch*, *Transformers*, *BERT*|

## Data Sciense + Computer Vision + Natutal Language Processing

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[11. Предсказание числа лайков в соцсети](https://github.com/goralex02/Likes_prediction)|Предсказание числа лайков в соцсети на основе четырех нейронных сетей - три для анализа разных типов данных (табличные данные о пользователе, текст публикации, изображение на публикации) и четвертая нейросеть объединяет предсказания трёх предыдущих в итоговый результат.|**Машинное обучение**, **Обработка естественного языка**, **Компьютерное зрение**, **Регрессия**, **Глубинное обучение**, **Transformers**|*Python*, *Pandas*, *NumPy*, *matplotlib*, *Scikit-learn*, *PyTorch*, *ResNet*, *Transformers*, *BERT*, *Transfer learning*|

## Time Series

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[12. Классификация наличия аномалий у временных рядов](https://github.com/goralex02/vk-ds-test-2024/blob/main/solution.ipynb)|Нужно провести бианрную классификацию временных рядов на предмет наличия аномалий.|**Машинное обучение**, **Временные ряды**, **Бинарная классификация**|*Python*, *Pandas*, *Matplotlib*, *CatBoost*|

## MLSecOps

|**Проект**|**Задача проекта**|**Направление**|**Требуемые навыки и инструменты**|
|:-----------------|:-------------------------------|:-----------|:-----------|
|[13. Атаки на ситемы машинного обучения](https://github.com/goralex02/attacks_on_models/tree/main)|Предлагается выполнить серию практических заданий, каждое из которых посвящено конкретной атаке. Задача — реализовать код для выполнения атаки, провести анализ последствий и предложить возможные способы защиты модели.|**Adversarial attack**, **Membership Inference attack**, **Model stealing**|*Python*, *PyTorch*, *Pandas*, *Matplotlib*, *Scikit-learn*, *YOLO*, *Transfer learning*, *ART*|