# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Загрузка датасета
df = pd.read_csv('tested.csv')

print("Первые 5 строк датасета:")
print(df.head())

print("\nОбщая информация о датасете:")
df.info()

print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())

print("\nСтатистическое описание числовых признаков:")
print(df.describe())

# Заполняем пропущенные значения в столбце 'Age' медианой
df['Age'].fillna(df['Age'].median(), inplace=True)
print("\nКоличество пропущенных значений в 'Age' после заполнения:")
print(df['Age'].isnull().sum())

# Заполняем пропущенные значения в столбце 'Embarked' самым частым значением (модой)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("\nКоличество пропущенных значений в 'Embarked' после заполнения:")
print(df['Embarked'].isnull().sum())

# Создаем новый признак 'HasCabin', который указывает на наличие каюты у пассажира
df['HasCabin'] = df['Cabin'].notnull().astype(int)
# Удаляем столбец 'Cabin' за ненадобностью
df.drop('Cabin', axis=1, inplace=True)
print("\nПервые 5 строк с новым признаком 'HasCabin':")
print(df[['HasCabin']].head())

# Преобразуем признак 'Sex' в числовой формат: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("\nУникальные значения в 'Sex' после преобразования:")
print(df['Sex'].unique())

# Преобразуем признак 'Embarked' с помощью метода one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print("\nПервые 5 строк после преобразования 'Embarked':")
print(df[['Embarked_Q', 'Embarked_S']].head())

# Создаем признак 'FamilySize' как сумму 'SibSp' и 'Parch' плюс 1 (сам пассажир)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("\nПервые 5 строк с новым признаком 'FamilySize':")
print(df[['SibSp', 'Parch', 'FamilySize']].head())

# Создаем признак 'IsAlone': 1, если пассажир путешествует один, иначе 0
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
print("\nПервые 5 строк с новым признаком 'IsAlone':")
print(df[['FamilySize', 'IsAlone']].head())

# Удаляем столбцы, которые не несут полезной информации для модели
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
print("\nОставшиеся признаки после удаления ненужных столбцов:")
print(df.columns)

# Вычисляем корреляционную матрицу
corr_matrix = df.corr()
print("\nКорреляция признаков с 'Survived':")
print(corr_matrix['Survived'].sort_values(ascending=False))

print("\nПервые 5 строк обработанного датасета:")
print(df.head())

# Построим несколько графиков для лучшего понимания данных

# График распределения выживших и не выживших пассажиров
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Распределение выживших и не выживших пассажиров')
plt.xlabel('Выжил (1) или не выжил (0)')
plt.ylabel('Количество пассажиров')
plt.show()

# Выживаемость в зависимости от пола
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Выживаемость в зависимости от пола')
plt.xlabel('Пол (0 = Мужчина, 1 = Женщина)')
plt.ylabel('Количество пассажиров')
plt.legend(title='Выжил', loc='upper right', labels=['Нет', 'Да'])
plt.show()

# Распределение возраста пассажиров
plt.figure(figsize=(8,6))
plt.hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Распределение возраста пассажиров')
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')
plt.show()

# Выживаемость в зависимости от класса обслуживания
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Выживаемость в зависимости от класса обслуживания')
plt.xlabel('Класс обслуживания (1 = Первый, 2 = Второй, 3 = Третий)')
plt.ylabel('Количество пассажиров')
plt.legend(title='Выжил', loc='upper right', labels=['Нет', 'Да'])
plt.show()

# Влияние платы за проезд на выживаемость
plt.figure(figsize=(8,6))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Влияние платы за проезд на выживаемость')
plt.xlabel('Выжил (0 = Нет, 1 = Да)')
plt.ylabel('Плата за проезд')
plt.show()

# Матрица корреляции признаков
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции признаков')
plt.show()
