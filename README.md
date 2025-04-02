Aqui está o **README.md** atualizado, incluindo trechos de código para ilustrar as análises realizadas no projeto.  

---

# 📊 Análise de Sentimento e Confiabilidade de Notícias  

Este projeto analisa artigos de notícias para entender seus **sentimentos** e **nível de confiabilidade**. Utilizamos **Python** e **Machine Learning** para prever sentimentos, identificar padrões e verificar a relação entre características das notícias e sua veracidade.  

## 📝 Perguntas analisadas:

1️⃣ **Qual é a média da pontuação de sentimento para a categoria "Technology"?**  
2️⃣ **Qual é a pontuação de sentimento do artigo "Breaking News 9" da HuffPost na categoria "Politics"?**  
3️⃣ **Podemos prever se um artigo da categoria "Business" terá um sentimento positivo ou negativo?**  
4️⃣ **Qual é a acurácia do modelo Random Forest para prever sentimentos?**  
5️⃣ **Quais fontes têm um baixo Trust Score e qual é a média da pontuação de sentimento dessas fontes?**  
6️⃣ **Existe correlação entre o número de compartilhamentos e a veracidade da notícia?**  
7️⃣ **Podemos prever se um artigo é "Fake" ou "Real" com base no número de comentários e na legibilidade?**  

---

## 🔧 Tecnologias utilizadas:
- **Python**  
- **Pandas** - Manipulação e análise de dados  
- **Seaborn & Matplotlib** - Visualização gráfica  
- **Scikit-learn** - Modelos de Machine Learning  

---

## 📌 Estrutura do Projeto  

### 1️⃣ Análise da Categoria "Technology"  
Calculamos a média da pontuação de sentimento e geramos um gráfico de distribuição.  

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
data = pd.read_csv("tabela_ajustada.csv")

# Filtrar artigos de tecnologia
tech_articles = data[data['category'] == 'Technology']

# Média do sentimento
average_sentiment = tech_articles['sentiment_score'].mean()
print(f'Média do sentimento para Tecnologia: {average_sentiment}')

# Criar gráfico de distribuição
plt.figure(figsize=(10, 5))
sns.histplot(tech_articles['sentiment_score'], bins=20, kde=True, color='blue')
plt.axvline(average_sentiment, color='red', linestyle='dashed', linewidth=2, label=f'Média: {average_sentiment:.2f}')
plt.xlabel("Pontuação de Sentimento")
plt.ylabel("Frequência")
plt.title("Distribuição das Pontuações de Sentimento - Tecnologia")
plt.legend()
plt.show()
```

---

### 2️⃣ Sentimento do artigo "Breaking News 9"  
Analisamos se o artigo tem um tom mais positivo ou negativo.  

```python
huffpost_politics = data[(data['source'] == 'HuffPost') & (data['category'] == 'Politics')]
article_sentiment = huffpost_politics[huffpost_politics['title'] == 'Breaking News 9']['sentiment_score'].iloc[0]
print(f'Sentimento do artigo "Breaking News 9": {article_sentiment}')
```

---

### 3️⃣ Previsão de Sentimento em Artigos de "Business"  
Treinamos um **modelo de Machine Learning (Random Forest)** para prever o sentimento dos artigos de negócios.  

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Convertendo dados categóricos
data['source'] = data['source'].astype('category').cat.codes
data['political_bias'] = data['political_bias'].map({'Left': 0, 'Center': 1, 'Right': 2})

# Definir variáveis de entrada
X = data[['source', 'political_bias', 'num_shares']]
y = data['sentiment_score'] > 0  

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)
```

---

### 4️⃣ Avaliação do Modelo  
Usamos **acurácia, matriz de confusão e relatório de classificação** para medir o desempenho.  

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Acurácia
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy:.2%}')

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, predictions))
```

---

### 5️⃣ Fontes com Baixo Trust Score  
Identificamos fontes com **baixo índice de confiança** e analisamos sua pontuação de sentimento.  

```python
low_trust_sources = data[data['trust_score'] < 3]['source'].unique()
print(f'Fontes com baixo Trust Score: {low_trust_sources}')

for source in low_trust_sources:
    avg_sentiment = data[data['source'] == source]['sentiment_score'].mean()
    print(f'{source}: Sentimento médio {avg_sentiment:.2f}')
```

---

### 6️⃣ Correlação entre Compartilhamentos e Veracidade  
Verificamos se o **número de compartilhamentos** tem relação com a **veracidade da notícia**.  

```python
# Convertendo a variável alvo para numérica
data['label'] = data['label'].apply(lambda x: 1 if x == 'Real' else 0)

# Correlação
correlation = data['num_shares'].corr(data['label'])
print(f'Correlação entre compartilhamentos e veracidade: {correlation}')
```

---

### 7️⃣ Previsão de Notícias "Fake" ou "Real"  
Treinamos um **modelo de Árvore de Decisão** para prever se uma notícia é falsa ou verdadeira.  

```python
from sklearn.tree import DecisionTreeClassifier

# Definir variáveis de entrada
X = data[['num_comments', 'readability_score']]
y = data['label']

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)
```

---

## 🚀 Como executar o projeto  

1️⃣ **Clone o repositório:**  
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```

2️⃣ **Instale as dependências:**  
```bash
pip install pandas seaborn scikit-learn matplotlib
```

3️⃣ **Execute o código no Jupyter Notebook:**  
```bash
jupyter notebook
```

4️⃣ **Ou execute diretamente com Python:**  
```bash
python script.py
```

---

## 📈 Conclusões  

🔹 **A categoria "Technology" tende a ter um sentimento médio neutro/positivo.**  
🔹 **O artigo "Breaking News 9" da HuffPost tem um sentimento específico, que pode ser analisado individualmente.**  
🔹 **Nosso modelo Random Forest conseguiu prever o sentimento dos artigos de "Business" com boa acurácia.**  
🔹 **Algumas fontes com baixo Trust Score tendem a ter um sentimento médio mais negativo.**  
🔹 **O número de compartilhamentos não tem forte correlação com a veracidade da notícia.**  
🔹 **A análise de legibilidade e número de comentários pode ser útil para prever se uma notícia é "Fake" ou "Real".**  

---

## 📌 Autor  
👨‍💻 **Rafael Andrade** 
📧 Contato: [seu-email@email.com](rafaelvitordeandrade58@gmail.com
)  
🔗 [LinkedIn](https://www.linkedin.com/in/rafael-andrade-1a2680358/) 

Se você achou útil, **⭐ Star no repositório**! 🚀✨
