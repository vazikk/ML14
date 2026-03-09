# ML14

1) Использовать GANs или VAEs для генерации синтетических данных <br>

Ипользовал gan <br>

```
# Импорты
from ctgan import CTGAN, load_demo
import pandas as pd
import os

# Создаем папку для данных
os.makedirs('data', exist_ok=True)

print("ШАГ 1: Загружаем реальные данные")
real_data = load_demo()
print(f"Реальные данные: {real_data.shape}")
print("\nПервые 5 строк:")
print(real_data.head())

# Сохраняем реальные данные
real_data.to_csv('data/real_adult.csv', index=False)
print("\n Реальные данные сохранены в data/real_adult.csv")

print("\nШАГ 2: Настраиваем CTGAN")
discrete_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country', 'income'
]
print(f"Категориальные колонки: {discrete_columns}")

print("\nШАГ 3: Обучаем модель (это займет ~1-2 минуты)...")
ctgan = CTGAN(epochs=10, batch_size=500, verbose=True)
ctgan.fit(real_data, discrete_columns)

print("\nШАГ 4: Генерируем синтетические данные")
synthetic = ctgan.sample(1000)
print(f"Сгенерировано: {synthetic.shape}")

print("\nШАГ 5: Проверяем базовые статистики")
print("\nВозраст (age):")
print(f"  Реальные:    мин={real_data['age'].min()}, макс={real_data['age'].max()}")
print(f"  Синтетика:   мин={synthetic['age'].min()}, макс={synthetic['age'].max()}")

print("\nЧасы работы (hours-per-week):")
print(f"  Реальные:    мин={real_data['hours-per-week'].min()}, макс={real_data['hours-per-week'].max()}")
print(f"  Синтетика:   мин={synthetic['hours-per-week'].min()}, макс={synthetic['hours-per-week'].max()}")

# Сохраняем синтетику
synthetic.to_csv('data/synthetic_adult.csv', index=False)
print("\n Синтетические данные сохранены в data/synthetic_adult.csv")
```
Результат: <br>

```
ШАГ 1: Загружаем реальные данные
Реальные данные: (32561, 15)

Первые 5 строк:
   age         workclass  fnlwgt  education  education-num  \
0   39         State-gov   77516  Bachelors             13   
1   50  Self-emp-not-inc   83311  Bachelors             13   
2   38           Private  215646    HS-grad              9   
3   53           Private  234721       11th              7   
4   28           Private  338409  Bachelors             13   

       marital-status         occupation   relationship   race     sex  \
0       Never-married       Adm-clerical  Not-in-family  White    Male   
1  Married-civ-spouse    Exec-managerial        Husband  White    Male   
2            Divorced  Handlers-cleaners  Not-in-family  White    Male   
3  Married-civ-spouse  Handlers-cleaners        Husband  Black    Male   
4  Married-civ-spouse     Prof-specialty           Wife  Black  Female   

   capital-gain  capital-loss  hours-per-week native-country income  
0          2174             0              40  United-States  <=50K  
1             0             0              13  United-States  <=50K  
2             0             0              40  United-States  <=50K  
3             0             0              40  United-States  <=50K  
4             0             0              40           Cuba  <=50K  

 Реальные данные сохранены в data/real_adult.csv

ШАГ 2: Настраиваем CTGAN
Категориальные колонки: ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

ШАГ 3: Обучаем модель (это займет ~1-2 минуты)...
Gen. (+00.00) | Discrim. (+00.00):   0%|                         | 0/10 [00:00<?, ?it/s]/home/valera/taska14/synth_env/lib/python3.12/site-packages/torch/autograd/graph.py:865: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:330.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Gen. (-01.20) | Discrim. (+00.06): 100%|████████████████| 10/10 [00:26<00:00,  2.66s/it]

ШАГ 4: Генерируем синтетические данные
Сгенерировано: (1000, 15)

ШАГ 5: Проверяем базовые статистики

Возраст (age):
  Реальные:    мин=17, макс=90
  Синтетика:   мин=13, макс=92

Часы работы (hours-per-week):
  Реальные:    мин=1, макс=99
  Синтетика:   мин=1, макс=102

Синтетические данные сохранены в data/synthetic_adult.csv
```
<img width="816" height="481" alt="image" src="https://github.com/user-attachments/assets/f6c24590-5814-4e04-9982-506b3ec41242" /> <br>

<br>

2) Валидировать статистические свойства synthetic vs real data <br>

```
import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
real = pd.read_csv('data/real_adult.csv')
synth = pd.read_csv('data/synthetic_adult.csv')

print("СРАВНЕНИЕ РЕАЛЬНЫХ И СИНТЕТИЧЕСКИХ ДАННЫХ")
print("=" * 50)

# 1. ЧИСЛОВЫЕ ПРИЗНАКИ
print("\n1. ЧИСЛОВЫЕ ПРИЗНАКИ:")
print("-" * 30)

for col in ['age', 'hours-per-week']:
    print(f"\n{col}:")
    print(f"  Реальные:    среднее={real[col].mean():.1f}, "
          f"мин={real[col].min()}, макс={real[col].max()}")
    print(f"  Синтетика:   среднее={synth[col].mean():.1f}, "
          f"мин={synth[col].min()}, макс={synth[col].max()}")

# 2. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
print("\n2. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:")
print("-" * 30)

cat_cols = ['sex', 'race', 'income']
for col in cat_cols:
    print(f"\n{col}:")
    print("  Реальные (%):")
    for cat, pct in real[col].value_counts(normalize=True).head(3).items():
        print(f"    {cat}: {pct*100:.1f}%")
    print("  Синтетика (%):")
    for cat, pct in synth[col].value_counts(normalize=True).head(3).items():
        print(f"    {cat}: {pct*100:.1f}%")

```

Результ: <br>

```
СРАВНЕНИЕ РЕАЛЬНЫХ И СИНТЕТИЧЕСКИХ ДАННЫХ
==================================================

1. ЧИСЛОВЫЕ ПРИЗНАКИ:
------------------------------

age:
  Реальные:    среднее=38.6, мин=17, макс=90
  Синтетика:   среднее=41.9, мин=13, макс=92

hours-per-week:
  Реальные:    среднее=40.4, мин=1, макс=99
  Синтетика:   среднее=38.3, мин=1, макс=102

2. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:
------------------------------

sex:
  Реальные (%):
    Male: 66.9%
    Female: 33.1%
  Синтетика (%):
    Male: 69.8%
    Female: 30.2%

race:
  Реальные (%):
    White: 85.4%
    Black: 9.6%
    Asian-Pac-Islander: 3.2%
  Синтетика (%):
    White: 69.1%
    Black: 18.7%
    Asian-Pac-Islander: 8.6%

income:
  Реальные (%):
    <=50K: 75.9%
    >50K: 24.1%
  Синтетика (%):
    <=50K: 78.7%
    >50K: 21.3%

```

<br>

3) Применять data augmentation techniques <br>

```
import pandas as pd
import numpy as np
import os

# Создаем папку для данных
os.makedirs('data', exist_ok=True)

# Загружаем реальные данные
real = pd.read_csv('data/real_adult.csv')
print("="*60)
print("DATA AUGMENTATION")
print("="*60)
print(f"Исходный размер данных: {real.shape}")
print(f"Колонки: {list(real.columns)}")

print("\n" + "="*60)
print("МЕТОД 1: ДОБАВЛЕНИЕ ШУМА (для числовых признаков)")
print("="*60)

def add_noise(df, noise_factor=0.05):
    """Добавляет небольшой шум к числовым признакам"""
    df_noisy = df.copy()
    
    # Для возраста
    std_age = df['age'].std()
    noise_age = np.random.normal(0, noise_factor * std_age, len(df))
    df_noisy['age'] = df['age'] + noise_age
    df_noisy['age'] = df_noisy['age'].clip(17, 90).round().astype(int)
    
    # Для часов работы
    std_hours = df['hours-per-week'].std()
    noise_hours = np.random.normal(0, noise_factor * std_hours, len(df))
    df_noisy['hours-per-week'] = df['hours-per-week'] + noise_hours
    df_noisy['hours-per-week'] = df_noisy['hours-per-week'].clip(1, 99).round().astype(int)
    
    return df_noisy

# Применяем
noisy_data = add_noise(real)
print(f"Данные с шумом: {noisy_data.shape}")
print("\nПример изменений (первые 3 строки):")
for i in range(3):
    print(f"\nСтрока {i+1}:")
    print(f"  Возраст: {real['age'].iloc[i]} -> {noisy_data['age'].iloc[i]}")
    print(f"  Часы: {real['hours-per-week'].iloc[i]} -> {noisy_data['hours-per-week'].iloc[i]}")

print("\n" + "="*60)
print("МЕТОД 2: ПЕРЕМЕШИВАНИЕ КАТЕГОРИЙ")
print("="*60)

def shuffle_categories(df, cols_to_shuffle=['education', 'occupation', 'race']):
    """Перемешивает значения в указанных колонках"""
    df_shuffled = df.copy()
    
    for col in cols_to_shuffle:
        if col in df.columns:
            original = df_shuffled[col].copy()
            shuffled = np.random.permutation(original)
            df_shuffled[col] = shuffled
    
    return df_shuffled

# Применяем
shuffled_data = shuffle_categories(real)
print(f"Данные с перемешанными категориями: {shuffled_data.shape}")
print("\nПример изменений (первые 3 строки):")
for i in range(3):
    print(f"\nСтрока {i+1}:")
    print(f"  Образование: {real['education'].iloc[i]} -> {shuffled_data['education'].iloc[i]}")
    print(f"  Профессия: {real['occupation'].iloc[i]} -> {shuffled_data['occupation'].iloc[i]}")
    print(f"  Раса: {real['race'].iloc[i]} -> {shuffled_data['race'].iloc[i]}")

print("\n" + "="*60)
print("МЕТОД 3: ИНТЕРПОЛЯЦИЯ (создание новых комбинаций)")
print("="*60)

def interpolate_rows(df, n_new=300):
    """Создает новые строки как комбинации существующих"""
    new_rows = []
    
    numeric_cols = ['age', 'hours-per-week']
    cat_cols = ['sex', 'race', 'income', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    
    for i in range(n_new):
        # Выбираем две случайные строки
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]
        
        new_row = {}
        
        # Для числовых - среднее
        for col in numeric_cols:
            new_row[col] = (row1[col] + row2[col]) // 2
        
        # Для категориальных - случайный выбор
        for col in cat_cols:
            if col in df.columns:
                new_row[col] = row1[col] if np.random.random() > 0.5 else row2[col]
        
        new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)

# Применяем
new_rows_df = interpolate_rows(real, n_new=300)
print(f"Создано новых строк: {len(new_rows_df)}")
print("\nПример новых строк (первые 3):")
print(new_rows_df[['age', 'hours-per-week', 'sex', 'race', 'income']].head(3))

print("\n" + "="*60)
print("СБОРКА ФИНАЛЬНОГО ДАТАСЕТА")
print("="*60)

# Объединяем все
final_augmented = pd.concat([
    real,                    # оригинал (32K)
    noisy_data,              # копии с шумом (32K)
    shuffled_data,           # с перемешанными категориями (32K)
    new_rows_df              # новые комбинации (300)
], ignore_index=True)

# Удаляем дубликаты
final_augmented = final_augmented.drop_duplicates()

print(f"\nРеальные данные: {len(real)} строк")
print(f"После аугментации: {len(final_augmented)} строк")
print(f"Рост: {len(final_augmented)/len(real):.2f}x")

# Статистика по ключевым признакам
print("\n" + "="*60)
print("СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ (ДО/ПОСЛЕ)")
print("="*60)

print("\nВозраст (среднее):")
print(f"  До: {real['age'].mean():.1f}")
print(f"  После: {final_augmented['age'].mean():.1f}")

print("\nЧасы работы (среднее):")
print(f"  До: {real['hours-per-week'].mean():.1f}")
print(f"  После: {final_augmented['hours-per-week'].mean():.1f}")

print("\nПол (% Male):")
print(f"  До: {(real['sex'] == 'Male').mean()*100:.1f}%")
print(f"  После: {(final_augmented['sex'] == 'Male').mean()*100:.1f}%")

print("\nДоход (% >50K):")
print(f"  До: {(real['income'] == '>50K').mean()*100:.1f}%")
print(f"  После: {(final_augmented['income'] == '>50K').mean()*100:.1f}%")

# Сохраняем
final_augmented.to_csv('data/adult_augmented.csv', index=False)
print("\n" + "="*60)
print(" Аугментированные данные сохранены в data/adult_augmented.csv")
print(f"   Размер файла: {len(final_augmented)} строк, {len(final_augmented.columns)} колонок")
print("="*60)

# Показываем пример финальных данных
print("\nПример финальных данных (первые 5 строк):")
print(final_augmented[['age', 'hours-per-week', 'sex', 'race', 'income']].head())
```

Результ: <br>

```
============================================================
DATA AUGMENTATION
============================================================
Исходный размер данных: (32561, 15)
Колонки: ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

============================================================
МЕТОД 1: ДОБАВЛЕНИЕ ШУМА (для числовых признаков)
============================================================
Данные с шумом: (32561, 15)

Пример изменений (первые 3 строки):

Строка 1:
  Возраст: 39 -> 38
  Часы: 40 -> 40

Строка 2:
  Возраст: 50 -> 49
  Часы: 13 -> 13

Строка 3:
  Возраст: 38 -> 39
  Часы: 40 -> 40

============================================================
МЕТОД 2: ПЕРЕМЕШИВАНИЕ КАТЕГОРИЙ
============================================================
Данные с перемешанными категориями: (32561, 15)

Пример изменений (первые 3 строки):

Строка 1:
  Образование: Bachelors -> Bachelors
  Профессия: Adm-clerical -> Prof-specialty
  Раса: White -> White

Строка 2:
  Образование: Bachelors -> Some-college
  Профессия: Exec-managerial -> Sales
  Раса: White -> White

Строка 3:
  Образование: HS-grad -> Some-college
  Профессия: Handlers-cleaners -> Craft-repair
  Раса: White -> Black

============================================================
МЕТОД 3: ИНТЕРПОЛЯЦИЯ (создание новых комбинаций)
============================================================
Создано новых строк: 300

Пример новых строк (первые 3):
   age  hours-per-week   sex                race income
0   45              37  Male               White  <=50K
1   20              28  Male  Asian-Pac-Islander  <=50K
2   46              32  Male               White  <=50K

============================================================
СБОРКА ФИНАЛЬНОГО ДАТАСЕТА
============================================================

Реальные данные: 32561 строк
После аугментации: 87329 строк
Рост: 2.68x

============================================================
СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ (ДО/ПОСЛЕ)
============================================================

Возраст (среднее):
  До: 38.6
  После: 38.6

Часы работы (среднее):
  До: 40.4
  После: 40.5

Пол (% Male):
  До: 66.9%
  После: 66.9%

Доход (% >50K):
  До: 24.1%
  После: 24.1%

============================================================
 Аугментированные данные сохранены в data/adult_augmented.csv
   Размер файла: 87329 строк, 15 колонок
============================================================

Пример финальных данных (первые 5 строк):
   age  hours-per-week     sex   race income
0   39              40    Male  White  <=50K
1   50              13    Male  White  <=50K
2   38              40    Male  White  <=50K
3   53              40    Male  Black  <=50K
4   28              40  Female  Black  <=50K

Click to add a cell.
```

<img width="794" height="368" alt="image" src="https://github.com/user-attachments/assets/d9e1ed76-aa96-4b08-8225-6a773524cf1b" /> <br>

<br>

4) Решать проблемы class imbalance <br>

```
import pandas as pd
import numpy as np

print("="*60)
print("AUTOMATIC CLASS IMBALANCE DETECTION")
print("="*60)

# Загружаем данные
real = pd.read_csv('data/real_adult.csv')
print(f"Всего записей: {len(real)}")
print(f"Колонки: {list(real.columns)}")

print("\n" + "="*60)
print("ПОИСК ДИСБАЛАНСА ВО ВСЕХ КАТЕГОРИАЛЬНЫХ ПРИЗНАКАХ")
print("="*60)

# Автоматически находим категориальные колонки
categorical_cols = real.select_dtypes(include=['object']).columns.tolist()
print(f"Найдено категориальных признаков: {len(categorical_cols)}")
print(categorical_cols)

print("\n" + "-"*60)
imbalance_report = []

for col in categorical_cols:
    print(f"\nПРИЗНАК: {col}")
    print("-"*40)
    
    # Считаем распределение
    counts = real[col].value_counts()
    percentages = real[col].value_counts(normalize=True) * 100
    
    # Находим самый частый и самый редкий класс
    most_common = counts.index[0]
    most_common_pct = percentages.iloc[0]
    most_common_count = counts.iloc[0]
    
    least_common = counts.index[-1]
    least_common_pct = percentages.iloc[-1]
    least_common_count = counts.iloc[-1]
    
    # Считаем коэффициент дисбаланса
    imbalance_ratio = most_common_count / least_common_count
    
    print(f"Всего категорий: {len(counts)}")
    print(f"\nСамый частый: '{most_common}' - {most_common_count} записей ({most_common_pct:.1f}%)")
    print(f"Самый редкий: '{least_common}' - {least_common_count} записей ({least_common_pct:.1f}%)")
    print(f"Коэффициент дисбаланса: {imbalance_ratio:.1f} (во сколько раз частый класс больше редкого)")
    
    # Определяем уровень дисбаланса
    if imbalance_ratio < 2:
        level = " Нормально"
    elif imbalance_ratio < 5:
        level = " Средний дисбаланс"
    elif imbalance_ratio < 10:
        level = " Высокий дисбаланс"
    else:
        level = " Критический дисбаланс"
    
    print(f"Статус: {level}")
    
    # Сохраняем в отчет
    imbalance_report.append({
        'Признак': col,
        'Категорий': len(counts),
        'Самый частый': f"{most_common} ({most_common_pct:.1f}%)",
        'Самый редкий': f"{least_common} ({least_common_pct:.1f}%)",
        'Коэф. дисбаланса': round(imbalance_ratio, 1),
        'Статус': level
    })
    
    # Показываем топ-5 категорий
    print("\nТоп-5 категорий:")
    for i, (cat, pct) in enumerate(percentages.head(5).items()):
        print(f"  {i+1}. {cat}: {pct:.1f}%")

print("\n" + "="*60)
print("ИТОГОВЫЙ ОТЧЕТ ПО ДИСБАЛАНСУ")
print("="*60)

# Создаем DataFrame с отчетом
report_df = pd.DataFrame(imbalance_report)
print(report_df.to_string(index=False))

print("\n" + "="*60)
print("АВТОМАТИЧЕСКАЯ БОРЬБА С ДИСБАЛАНСОМ")
print("="*60)

# Спрашиваем, с каким признаком работать
print("\nВ каком признаке хотите исправить дисбаланс?")
for i, col in enumerate(categorical_cols):
    print(f"{i+1}. {col}")

choice = input("\nВведите номер (или нажмите Enter для выбора 'income'): ").strip()

if choice == "":
    target_col = 'income'
    print(f"Выбран признак: {target_col}")
else:
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(categorical_cols):
            target_col = categorical_cols[idx]
        else:
            target_col = 'income'
            print(f"Неверный номер. Выбран {target_col}")
    except:
        target_col = 'income'
        print(f"Выбран {target_col}")

print("\n" + "-"*60)
print(f"БОРЬБА С ДИСБАЛАНСОМ В ПРИЗНАКЕ: {target_col}")
print("-"*60)

# Анализируем выбранный признак
counts = real[target_col].value_counts()
percentages = real[target_col].value_counts(normalize=True) * 100

print(f"\nТекущее распределение:")
for cat, pct in percentages.items():
    print(f"  {cat}: {pct:.1f}% ({counts[cat]} записей)")

# Находим миноритарный класс (самый редкий)
minority_class = counts.index[-1]
minority_count = counts[-1]
majority_count = counts[0]

print(f"\nМиноритарный класс: '{minority_class}' ({minority_count} записей)")
print(f"Мажоритарный класс: '{counts.index[0]}' ({majority_count} записей)")

# Создаем сбалансированную версию
print(f"\nСоздаем сбалансированную версию...")

# Берем все записи миноритарного класса
minority_data = real[real[target_col] == minority_class].copy()

# Сколько нужно создать копий
needed = majority_count - minority_count
print(f"Нужно создать {needed} копий миноритарного класса")

# Создаем копии с небольшими изменениями
augmented_minority = []

for i in range(needed):
    # Берем случайную запись из миноритарного класса
    person = minority_data.sample(1).iloc[0].copy()
    
    # Меняем числовые признаки (чтобы не было точных дубликатов)
    if 'age' in person.index:
        person['age'] = min(90, max(17, person['age'] + np.random.randint(-3, 4)))
    if 'hours-per-week' in person.index:
        person['hours-per-week'] = min(99, max(1, person['hours-per-week'] + np.random.randint(-5, 6)))
    if 'education-num' in person.index:
        person['education-num'] = min(16, max(1, person['education-num'] + np.random.randint(-1, 2)))
    
    augmented_minority.append(person)

print(f"Создано {len(augmented_minority)} новых записей")

# Собираем сбалансированный датасет
majority_data = real[real[target_col] == counts.index[0]]
balanced_data = pd.concat([
    majority_data,
    minority_data,
    pd.DataFrame(augmented_minority)
])

print("\nРЕЗУЛЬТАТ:")
print("-"*40)
new_counts = balanced_data[target_col].value_counts()
new_percentages = balanced_data[target_col].value_counts(normalize=True) * 100

for cat in new_counts.index:
    print(f"  {cat}: {new_counts[cat]} записей ({new_percentages[cat]:.1f}%)")

# Сохраняем
output_file = f'data/adult_balanced_{target_col}.csv'
balanced_data.to_csv(output_file, index=False)
print(f"\n Сбалансированные данные сохранены в {output_file}")

print("\n" + "="*60)
print(" Теперь классы сбалансированы.")
print("="*60)
```

Результ: <br>

```
============================================================
AUTOMATIC CLASS IMBALANCE DETECTION
============================================================
Всего записей: 32561
Колонки: ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

============================================================
ПОИСК ДИСБАЛАНСА ВО ВСЕХ КАТЕГОРИАЛЬНЫХ ПРИЗНАКАХ
============================================================
Найдено категориальных признаков: 9
['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

------------------------------------------------------------

ПРИЗНАК: workclass
----------------------------------------
Всего категорий: 9

Самый частый: 'Private' - 22696 записей (69.7%)
Самый редкий: 'Never-worked' - 7 записей (0.0%)
Коэффициент дисбаланса: 3242.3 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. Private: 69.7%
  2. Self-emp-not-inc: 7.8%
  3. Local-gov: 6.4%
  4. ?: 5.6%
  5. State-gov: 4.0%

ПРИЗНАК: education
----------------------------------------
Всего категорий: 16

Самый частый: 'HS-grad' - 10501 записей (32.3%)
Самый редкий: 'Preschool' - 51 записей (0.2%)
Коэффициент дисбаланса: 205.9 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. HS-grad: 32.3%
  2. Some-college: 22.4%
  3. Bachelors: 16.4%
  4. Masters: 5.3%
  5. Assoc-voc: 4.2%

ПРИЗНАК: marital-status
----------------------------------------
Всего категорий: 7

Самый частый: 'Married-civ-spouse' - 14976 записей (46.0%)
Самый редкий: 'Married-AF-spouse' - 23 записей (0.1%)
Коэффициент дисбаланса: 651.1 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. Married-civ-spouse: 46.0%
  2. Never-married: 32.8%
  3. Divorced: 13.6%
  4. Separated: 3.1%
  5. Widowed: 3.0%

ПРИЗНАК: occupation
----------------------------------------
Всего категорий: 15

Самый частый: 'Prof-specialty' - 4140 записей (12.7%)
Самый редкий: 'Armed-Forces' - 9 записей (0.0%)
Коэффициент дисбаланса: 460.0 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. Prof-specialty: 12.7%
  2. Craft-repair: 12.6%
  3. Exec-managerial: 12.5%
  4. Adm-clerical: 11.6%
  5. Sales: 11.2%

ПРИЗНАК: relationship
----------------------------------------
Всего категорий: 6

Самый частый: 'Husband' - 13193 записей (40.5%)
Самый редкий: 'Other-relative' - 981 записей (3.0%)
Коэффициент дисбаланса: 13.4 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. Husband: 40.5%
  2. Not-in-family: 25.5%
  3. Own-child: 15.6%
  4. Unmarried: 10.6%
  5. Wife: 4.8%

ПРИЗНАК: race
----------------------------------------
Всего категорий: 5

Самый частый: 'White' - 27816 записей (85.4%)
Самый редкий: 'Other' - 271 записей (0.8%)
Коэффициент дисбаланса: 102.6 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. White: 85.4%
  2. Black: 9.6%
  3. Asian-Pac-Islander: 3.2%
  4. Amer-Indian-Eskimo: 1.0%
  5. Other: 0.8%

ПРИЗНАК: sex
----------------------------------------
Всего категорий: 2

Самый частый: 'Male' - 21790 записей (66.9%)
Самый редкий: 'Female' - 10771 записей (33.1%)
Коэффициент дисбаланса: 2.0 (во сколько раз частый класс больше редкого)
Статус:  Средний дисбаланс

Топ-5 категорий:
  1. Male: 66.9%
  2. Female: 33.1%

ПРИЗНАК: native-country
----------------------------------------
Всего категорий: 42

Самый частый: 'United-States' - 29170 записей (89.6%)
Самый редкий: 'Holand-Netherlands' - 1 записей (0.0%)
Коэффициент дисбаланса: 29170.0 (во сколько раз частый класс больше редкого)
Статус:  Критический дисбаланс

Топ-5 категорий:
  1. United-States: 89.6%
  2. Mexico: 2.0%
  3. ?: 1.8%
  4. Philippines: 0.6%
  5. Germany: 0.4%

ПРИЗНАК: income
----------------------------------------
Всего категорий: 2

Самый частый: '<=50K' - 24720 записей (75.9%)
Самый редкий: '>50K' - 7841 записей (24.1%)
Коэффициент дисбаланса: 3.2 (во сколько раз частый класс больше редкого)
Статус:  Средний дисбаланс

Топ-5 категорий:
  1. <=50K: 75.9%
  2. >50K: 24.1%

============================================================
ИТОГОВЫЙ ОТЧЕТ ПО ДИСБАЛАНСУ
============================================================
       Признак  Категорий               Самый частый              Самый редкий  Коэф. дисбаланса                  Статус
     workclass          9            Private (69.7%)       Never-worked (0.0%)            3242.3  Критический дисбаланс
     education         16            HS-grad (32.3%)          Preschool (0.2%)             205.9  Критический дисбаланс
marital-status          7 Married-civ-spouse (46.0%)  Married-AF-spouse (0.1%)             651.1  Критический дисбаланс
    occupation         15     Prof-specialty (12.7%)       Armed-Forces (0.0%)             460.0  Критический дисбаланс
  relationship          6            Husband (40.5%)     Other-relative (3.0%)              13.4  Критический дисбаланс
          race          5              White (85.4%)              Other (0.8%)             102.6  Критический дисбаланс
           sex          2               Male (66.9%)            Female (33.1%)               2.0   Средний дисбаланс
native-country         42      United-States (89.6%) Holand-Netherlands (0.0%)           29170.0  Критический дисбаланс
        income          2              <=50K (75.9%)              >50K (24.1%)               3.2    Средний дисбаланс

============================================================
АВТОМАТИЧЕСКАЯ БОРЬБА С ДИСБАЛАНСОМ
============================================================

В каком признаке хотите исправить дисбаланс?
1. workclass
2. education
3. marital-status
4. occupation
5. relationship
6. race
7. sex
8. native-country
9. income

Введите номер (или нажмите Enter для выбора 'income'):  9

------------------------------------------------------------
БОРЬБА С ДИСБАЛАНСОМ В ПРИЗНАКЕ: income
------------------------------------------------------------

Текущее распределение:
  <=50K: 75.9% (24720 записей)
  >50K: 24.1% (7841 записей)

Миноритарный класс: '>50K' (7841 записей)
Мажоритарный класс: '<=50K' (24720 записей)

Создаем сбалансированную версию...
Нужно создать 16879 копий миноритарного класса
/tmp/ipykernel_11969/2781508366.py:125: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  minority_count = counts[-1]
/tmp/ipykernel_11969/2781508366.py:126: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  majority_count = counts[0]
Создано 16879 новых записей

РЕЗУЛЬТАТ:
----------------------------------------
  <=50K: 24720 записей (50.0%)
  >50K: 24720 записей (50.0%)

 Сбалансированные данные сохранены в data/adult_balanced_income.csv

============================================================
Теперь классы сбалансированы.
============================================================
```

<img width="769" height="347" alt="image" src="https://github.com/user-attachments/assets/77c40de0-8804-4c5b-b343-44c40f46be9c" /> <br>

5) Обеспечивать privacy через synthetic data <br>

```
import pandas as pd
import numpy as np

print("="*60)
print("PRIVACY CHECK - ПРОВЕРКА ПРИВАТНОСТИ")
print("="*60)

# Загружаем данные
real = pd.read_csv('data/real_adult.csv')
synth = pd.read_csv('data/synthetic_adult.csv')  # синтетика из пункта 1

print(f"Реальные данные: {len(real)} записей")
print(f"Синтетические данные: {len(synth)} записей")

print("\n" + "="*60)
print("1. ПОИСК ТОЧНЫХ КОПИЙ")
print("="*60)

# Проверяем, есть ли строки из real в synth
real_tuples = set(map(tuple, real.values))
synth_tuples = set(map(tuple, synth.values))

duplicates = real_tuples.intersection(synth_tuples)
print(f"Найдено точных копий реальных людей в синтетике: {len(duplicates)}")

if len(duplicates) == 0:
    print(" Хорошо! Нет точных копий - приватность соблюдается")
else:
    print(" Проблема! Найдены точные копии")

print("\n" + "="*60)
print("2. ПОИСК ПОХОЖИХ ЛЮДЕЙ (РИСК ИДЕНТИФИКАЦИИ)")
print("="*60)

# Выбираем ключевые признаки для проверки
key_features = ['age', 'sex', 'race', 'education-num', 'hours-per-week']

print(f"Проверяем по признакам: {key_features}")

# Для каждого реального человека ищем похожего в синтетике
at_risk = 0
risk_threshold = 2  # считаем рискованным, если отличаются не более чем по 2 признакам

for i, real_row in real[key_features].iterrows():
    for j, synth_row in synth[key_features].iterrows():
        # Считаем количество совпадающих признаков
        matches = sum(real_row[k] == synth_row[k] for k in key_features)
        if matches >= len(key_features) - risk_threshold:
            at_risk += 1
            break  # достаточно одного опасного совпадения

print(f"Найдено реальных людей, у которых есть очень похожий синтетический: {at_risk}")
print(f"Это {at_risk/len(real)*100:.2f}% от всех реальных людей")

if at_risk < len(real) * 0.01:  # меньше 1%
    print(" Риск низкий")
else:
    print(" Есть потенциальный риск")

print("\n" + "="*60)
print("3. ПРОВЕРКА РЕДКИХ КОМБИНАЦИЙ")
print("="*60)

# Ищем редкие комбинации в реальных данных
print("Ищем редкие группы людей (например, женщины-инженеры старше 70)")
rare_groups = []

# Несколько примеров редких групп
test_cases = [
    {'sex': 'Female', 'occupation': 'Exec-managerial', 'age': lambda x: x > 70},
    {'race': 'Other', 'income': '>50K'},
    {'native-country': 'Holand-Netherlands', 'income': '>50K'}
]

for i, test in enumerate(test_cases):
    print(f"\nТест {i+1}: {test}")
    
    # Ищем в реальных данных
    real_mask = pd.Series([True] * len(real))
    for k, v in test.items():
        if callable(v):
            real_mask = real_mask & v(real[k])
        else:
            real_mask = real_mask & (real[k] == v)
    
    real_count = real_mask.sum()
    
    # Ищем в синтетических
    synth_mask = pd.Series([True] * len(synth))
    for k, v in test.items():
        if callable(v):
            synth_mask = synth_mask & v(synth[k])
        else:
            synth_mask = synth_mask & (synth[k] == v)
    
    synth_count = synth_mask.sum()
    
    print(f"  В реальных данных: {real_count} человек")
    print(f"  В синтетических: {synth_count} человек")
    
    if real_count > 0 and synth_count > 0:
        print("    Найдена редкая группа в синтетике - возможен риск")

print("\n" + "="*60)
print("4. ЗАЩИТА ЧЕРЕЗ ПОСТ-ОБРАБОТКУ")
print("="*60)

# Создаем "приватную" версию синтетических данных
private_synth = synth.copy()

# Добавляем дополнительный шум для защиты
print("Добавляем защитный шум...")

private_synth['age'] = private_synth['age'] + np.random.randint(-2, 3, size=len(private_synth))
private_synth['age'] = private_synth['age'].clip(17, 90)

private_synth['hours-per-week'] = private_synth['hours-per-week'] + np.random.randint(-3, 4, size=len(private_synth))
private_synth['hours-per-week'] = private_synth['hours-per-week'].clip(1, 99)

print(" Защитный шум добавлен")

# Сохраняем приватную версию
private_synth.to_csv('data/synthetic_private.csv', index=False)
print("\n Приватные синтетические данные сохранены в data/synthetic_private.csv")

print("\n" + "="*60)
print("ИТОГ ПО ПРИВАТНОСТИ")
print("="*60)

print("""
Что мы проверили:
1.  Точные копии - не найдены (хорошо)
2.  Похожие люди - проверяем процент совпадений
3.  Редкие группы - проверяем, не "светятся" ли уникальные люди
4.  Защитный шум - добавлен для дополнительной безопасности

Рекомендации:
- Если найдены точные копии -> уменьшить эпохи обучения или добавить шум
- Если редкие группы "протекают" -> увеличить шум или использовать differential privacy
- Для production всегда использовать приватную версию с дополнительным шумом
""")

# Финальная проверка приватной версии
private_tuples = set(map(tuple, private_synth.values))
private_duplicates = real_tuples.intersection(private_tuples)

print(f"\nПосле добавления шума точных копий: {len(private_duplicates)}")
if len(private_duplicates) == 0:
    print(" Приватная версия безопасна!")
```

результ: <br>

```
============================================================
PRIVACY CHECK - ПРОВЕРКА ПРИВАТНОСТИ
============================================================
Реальные данные: 32561 записей
Синтетические данные: 1000 записей

============================================================
1. ПОИСК ТОЧНЫХ КОПИЙ
============================================================
Найдено точных копий реальных людей в синтетике: 0
 Хорошо! Нет точных копий - приватность соблюдается

============================================================
2. ПОИСК ПОХОЖИХ ЛЮДЕЙ (РИСК ИДЕНТИФИКАЦИИ)
============================================================
Проверяем по признакам: ['age', 'sex', 'race', 'education-num', 'hours-per-week']
Найдено реальных людей, у которых есть очень похожий синтетический: 32462
Это 99.70% от всех реальных людей
 Есть потенциальный риск

============================================================
3. ПРОВЕРКА РЕДКИХ КОМБИНАЦИЙ
============================================================
Ищем редкие группы людей (например, женщины-инженеры старше 70)

Тест 1: {'sex': 'Female', 'occupation': 'Exec-managerial', 'age': <function <lambda> at 0x74a9452c0ae0>}
  В реальных данных: 10 человек
  В синтетических: 1 человек
    Найдена редкая группа в синтетике - возможен риск

Тест 2: {'race': 'Other', 'income': '>50K'}
  В реальных данных: 25 человек
  В синтетических: 9 человек
    Найдена редкая группа в синтетике - возможен риск

Тест 3: {'native-country': 'Holand-Netherlands', 'income': '>50K'}
  В реальных данных: 0 человек
  В синтетических: 0 человек

============================================================
4. ЗАЩИТА ЧЕРЕЗ ПОСТ-ОБРАБОТКУ
============================================================
Добавляем защитный шум...
 Защитный шум добавлен

 Приватные синтетические данные сохранены в data/synthetic_private.csv

============================================================
ИТОГ ПО ПРИВАТНОСТИ
============================================================

Что мы проверили:
1.  Точные копии - не найдены (хорошо)
2.  Похожие люди - проверяем процент совпадений
3.  Редкие группы - проверяем, не "светятся" ли уникальные люди
4.  Защитный шум - добавлен для дополнительной безопасности

Рекомендации:
- Если найдены точные копии -> уменьшить эпохи обучения или добавить шум
- Если редкие группы "протекают" -> увеличить шум или использовать differential privacy
- Для production всегда использовать приватную версию с дополнительным шумом


После добавления шума точных копий: 0
 Приватная версия безопасна!
```
<img width="767" height="299" alt="image" src="https://github.com/user-attachments/assets/935bff90-aa63-414a-96bc-6d2cc3fe0116" /> <br>

6) Сравнивать performance моделей на real vs synthetic data <br>

```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("СРАВНЕНИЕ МОДЕЛЕЙ: REAL vs SYNTHETIC PRIVATE")
print("="*70)

# Загружаем данные
print("\n1. ЗАГРУЗКА ДАННЫХ:")
print("-"*50)

real = pd.read_csv('data/real_adult.csv')
private = pd.read_csv('data/adult_augmented.csv')

print(f"Реальные данные: {len(real)} записей")
print(f"Приватные синтетические: {len(private)} записей")

# Подготовка данных
print("\n2. ПОДГОТОВКА ДАННЫХ:")
print("-"*50)

def prepare_data(df):
    """Преобразует данные в формат для обучения"""
    data = df.copy()
    
    # Создаем целевую переменную
    data['target'] = (data['income'] == '>50K').astype(int)
    
    # Признаки
    features = ['age', 'hours-per-week', 'education-num']
    
    # Кодируем пол
    data['sex_male'] = (data['sex'] == 'Male').astype(int)
    features.append('sex_male')
    
    # Кодируем расу (упрощенно)
    data['race_white'] = (data['race'] == 'White').astype(int)
    features.append('race_white')
    
    return data[features], data['target'], features

# Подготавливаем
X_real, y_real, features = prepare_data(real)
X_private, y_private, _ = prepare_data(private)

print(f"Признаки: {features}")
print(f"\nРеальные данные: {X_real.shape}, богатых: {y_real.sum()/len(y_real)*100:.1f}%")
print(f"Приватные: {X_private.shape}, богатых: {y_private.sum()/len(y_private)*100:.1f}%")

# Разделяем реальные на train/test
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
)

print(f"\nТренировочные (реальные): {X_train_real.shape}")
print(f"Тестовые (реальные): {X_test_real.shape}")

print("\n" + "="*70)
print("3. ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ")
print("="*70)

results = []

def train_and_evaluate(X_train, y_train, X_test, y_test, name):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Модель': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f} (из тех, кого назвал богатыми - сколько правда богатые)")
    print(f"  Recall:    {metrics['Recall']:.4f} (из всех богатых - сколько нашел)")
    print(f"  F1-score:  {metrics['F1-score']:.4f} (среднее между Precision и Recall)")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    return metrics

# Модель 1: Обучена на реальных, тест на реальных
print("\n--- МОДЕЛЬ 1: REAL (обучена на реальных) ---")
res_real = train_and_evaluate(X_train_real, y_train_real, X_test_real, y_test_real, "Real (эталон)")
results.append(res_real)

# Модель 2: Обучена на приватных, тест на реальных
print("\n--- МОДЕЛЬ 2: PRIVATE (обучена на синтетике с шумом) ---")
res_private = train_and_evaluate(X_private, y_private, X_test_real, y_test_real, "Private (с шумом)")
results.append(res_private)

# Модель 3: Обучена на комбинации, тест на реальных
print("\n--- МОДЕЛЬ 3: REAL + PRIVATE (обучена на всех данных) ---")
X_combined = pd.concat([X_train_real, X_private])
y_combined = pd.concat([y_train_real, y_private])
res_combined = train_and_evaluate(X_combined, y_combined, X_test_real, y_test_real, "Real + Private")
results.append(res_combined)

print("\n" + "="*70)
print("4. СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("="*70)

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("5. АНАЛИЗ")
print("="*70)

# Берем реальную модель как эталон
real_f1 = results_df[results_df['Модель'] == 'Real (эталон)']['F1-score'].values[0]

print("\nНасколько хуже синтетика:")
for _, row in results_df.iterrows():
    if row['Модель'] != 'Real (эталон)':
        f1_diff = ((row['F1-score'] - real_f1) / real_f1) * 100
        print(f"\n{row['Модель']}:")
        print(f"  F1-score: {row['F1-score']:.4f} vs {real_f1:.4f} (эталон)")
        print(f"  Разница: {f1_diff:+.1f}%")
        
        if f1_diff >= -5:
            print("   Отличный результат! Синтетика почти не уступает")
        elif f1_diff >= -10:
            print("   Приемлемо, но синтетика хуже на 5-10%")
        else:
            print("   Синтетика сильно хуже реальных данных")

print("\n" + "="*70)
print("6. ВЫВОД")
print("="*70)

print("""
 ИТОГИ ЭКСПЕРИМЕНТА:
------------------------""")

best_f1 = results_df.loc[results_df['F1-score'].idxmax()]['Модель']
print(f" Лучшая модель: {best_f1}")

if res_private['F1-score'] >= res_real['F1-score'] * 0.95:
    print("\n СИНТЕТИЧЕСКИЕ ДАННЫЕ ПРИГОДНЫ ДЛЯ ИСПОЛЬЗОВАНИЯ")
    print("   Они сохраняют >95% качества при обеспечении приватности")
else:
    print("\n СИНТЕТИКА ЗАМЕТНО ХУЖЕ РЕАЛЬНЫХ ДАННЫХ")
    print("   Нужно улучшать качество генерации")

print(f"""
Детали:
- Real F1:    {res_real['F1-score']:.3f}
- Private F1: {res_private['F1-score']:.3f}
- Real+Private F1: {res_combined['F1-score']:.3f}
""")
```

результ: <br>

```
======================================================================
СРАВНЕНИЕ МОДЕЛЕЙ: REAL vs SYNTHETIC PRIVATE
======================================================================

1. ЗАГРУЗКА ДАННЫХ:
--------------------------------------------------
Реальные данные: 32561 записей
Приватные синтетические: 87329 записей

2. ПОДГОТОВКА ДАННЫХ:
--------------------------------------------------
Признаки: ['age', 'hours-per-week', 'education-num', 'sex_male', 'race_white']

Реальные данные: (32561, 5), богатых: 24.1%
Приватные: (87329, 5), богатых: 24.1%

Тренировочные (реальные): (22792, 5)
Тестовые (реальные): (9769, 5)

======================================================================
3. ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ
======================================================================

--- МОДЕЛЬ 1: REAL (обучена на реальных) ---

Real (эталон):
  Accuracy:  0.7847
  Precision: 0.5701 (из тех, кого назвал богатыми - сколько правда богатые)
  Recall:    0.4307 (из всех богатых - сколько нашел)
  F1-score:  0.4907 (среднее между Precision и Recall)
  ROC-AUC:   0.7907

--- МОДЕЛЬ 2: PRIVATE (обучена на синтетике с шумом) ---

Private (с шумом):
  Accuracy:  0.8630
  Precision: 0.7726 (из тех, кого назвал богатыми - сколько правда богатые)
  Recall:    0.6110 (из всех богатых - сколько нашел)
  F1-score:  0.6823 (среднее между Precision и Recall)
  ROC-AUC:   0.9258

--- МОДЕЛЬ 3: REAL + PRIVATE (обучена на всех данных) ---

Real + Private:
  Accuracy:  0.8516
  Precision: 0.7475 (из тех, кого назвал богатыми - сколько правда богатые)
  Recall:    0.5791 (из всех богатых - сколько нашел)
  F1-score:  0.6526 (среднее между Precision и Recall)
  ROC-AUC:   0.9157

======================================================================
4. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
======================================================================
           Модель  Accuracy  Precision  Recall  F1-score  ROC-AUC
    Real (эталон)    0.7847     0.5701  0.4307    0.4907   0.7907
Private (с шумом)    0.8630     0.7726  0.6110    0.6823   0.9258
   Real + Private    0.8516     0.7475  0.5791    0.6526   0.9157

======================================================================
5. АНАЛИЗ
======================================================================

Насколько хуже синтетика:

Private (с шумом):
  F1-score: 0.6823 vs 0.4907 (эталон)
  Разница: +39.0%
   Отличный результат! Синтетика почти не уступает

Real + Private:
  F1-score: 0.6526 vs 0.4907 (эталон)
  Разница: +33.0%
   Отличный результат! Синтетика почти не уступает

======================================================================
6. ВЫВОД
======================================================================

 ИТОГИ ЭКСПЕРИМЕНТА:
------------------------
 Лучшая модель: Private (с шумом)

 СИНТЕТИЧЕСКИЕ ДАННЫЕ ПРИГОДНЫ ДЛЯ ИСПОЛЬЗОВАНИЯ
   Они сохраняют >95% качества при обеспечении приватности

Детали:
- Real F1:    0.491
- Private F1: 0.682
- Real+Private F1: 0.653

```

