import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="FB-31 Gek VHI Data Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'init_done' not in st.session_state:
    st.session_state.init_done = True
    st.session_state.selected_param = 'VHI'
    st.session_state.selected_area = 'Вінницька'
    st.session_state.week_range = (1, 52)
    st.session_state.year_range = (1982, 2024)
    st.session_state.ascending_sort = False
    st.session_state.descending_sort = False

region_names = {
    1: "Вінницька",
    2: "Волинська",
    3: "Дніпропетровська",
    4: "Донецька",
    5: "Житомирська",
    6: "Закарпатська",
    7: "Запорізька",
    8: "Івано-Франківська",
    9: "Київська",
    10: "Кіровоградська",
    11: "Луганська",
    12: "Львівська",
    13: "Миколаївська",
    14: "Одеська",
    15: "Полтавська",
    16: "Рівненська",
    17: "Сумська",
    18: "Тернопільська",
    19: "Харківська",
    20: "Херсонська",
    21: "Хмельницька",
    22: "Черкаська",
    23: "Чернівецька",
    24: "Чернігівська",
    25: "Республіка Крим"
}

@st.cache_data
def read_vhi_files(data_dir="vhi_data"):
    if not os.path.exists(data_dir):
        st.error(f"Директорія {data_dir} не існує!")
        return None

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        st.error(f"У директорії {data_dir} не знайдено CSV файлів!")
        return None

    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    all_data = []

    for file in csv_files:
        try:
            province_id = int(file.split('_id_')[1].split('_')[0])
            df = pd.read_csv(file, header=1, names=headers)
            
            if isinstance(df.at[0, 'Year'], str) and '<' in df.at[0, 'Year']:
                df.at[0, 'Year'] = df.at[0, 'Year'].split('<')[0].strip()
            
            df = df.drop(df.loc[df['VHI'] == -1].index)
            
            df = df.drop('empty', axis=1)
            
            df['region_id'] = province_id
            
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
            
            all_data.append(df)
            
        except Exception as e:
            st.warning(f"Помилка при зчитуванні файлу {file}: {e}")
    
    if not all_data:
        st.error("Не вдалося зчитати жодного файлу з даними!")
        return None
    
    result_df = pd.concat(all_data, ignore_index=True)
    return result_df

def change_region_indices(df):
    if df is None or df.empty:
        st.error("Отримано порожній DataFrame!")
        return df
    
    region_mapping = {
        1: 22, 2: 24, 3: 23, 4: 25, 5: 3, 6: 4, 7: 8, 8: 19, 9: 20, 10: 21,
        11: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16,
        21: 17, 22: 18, 23: 6, 24: 1, 25: 2, 26: 6, 27: 5
    }
    
    result_df = df.copy()
    
    result_df['region_id'] = result_df['region_id'].map(region_mapping)
    
    result_df['region_name'] = result_df['region_id'].map(region_names)
    
    return result_df

def filter_data(df, param, area_name, week_range, year_range, asc, desc):
    if df is None or df.empty:
        return None

    area_id = None
    for key, value in region_names.items():
        if value == area_name:
            area_id = key
            break

    if area_id is None:
        st.error(f"Не вдалося знайти ID для області {area_name}")
        return None

    filtered_df = df[
        (df['region_id'] == area_id) &
        (df['Week'] >= week_range[0]) &
        (df['Week'] <= week_range[1]) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ].copy()

    if asc and not desc:
        filtered_df = filtered_df.sort_values(by=param, ascending=True)
    elif desc and not asc:
        filtered_df = filtered_df.sort_values(by=param, ascending=False)
    elif asc and desc:
        st.warning("Обрано обидва напрямки сортування. Дані відображаються без сортування.")
    
    return filtered_df

def plot_vhi_data(filtered_df, param):
    if filtered_df is None or filtered_df.empty:
        st.warning("Немає даних для візуалізації.")
        return None

    plt.figure(figsize=(12, 7))
    
    years = sorted(filtered_df['Year'].unique())
    
    if len(years) > 10:
        legend_cols = 2
    else:
        legend_cols = 1
    
    for year in years:
        year_data = filtered_df[filtered_df['Year'] == year]
        plt.plot(year_data['Week'], year_data[param], marker='o', linestyle='-', label=str(int(year)))
    
    area_name = filtered_df['region_name'].iloc[0] if not filtered_df.empty else "обрана область"
    
    plt.title(f"{param} для області {area_name}", fontsize=14)
    plt.xlabel("Тиждень", fontsize=12)
    plt.ylabel(param, fontsize=12)
    plt.grid(True)
    
    plt.legend(
        title="Рік", 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5), 
        fontsize=10, 
        ncol=legend_cols,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1
    )
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_comparison_data(df, param, area_name, week_range, year_range):
    if df is None or df.empty:
        st.warning("Немає даних для візуалізації.")
        return None

    area_id = None
    for key, value in region_names.items():
        if value == area_name:
            area_id = key
            break

    if area_id is None:
        st.error(f"Не вдалося знайти ID для області {area_name}")
        return None

    filtered_all = df[
        (df['Week'] >= week_range[0]) &
        (df['Week'] <= week_range[1]) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ].copy()

    if filtered_all.empty:
        st.warning("Немає даних для візуалізації після фільтрації.")
        return None

    avg_data = filtered_all.groupby(['region_id', 'region_name', 'Week'])[param].mean().reset_index()
    
    selected_area_data = avg_data[avg_data['region_id'] == area_id]
    other_areas_data = avg_data[avg_data['region_id'] != area_id]
    
    plt.figure(figsize=(12, 7))
    
    for region_id, region_group in other_areas_data.groupby('region_id'):
        region_name = region_group['region_name'].iloc[0]
        plt.plot(region_group['Week'], region_group[param], alpha=0.3, linestyle='-', linewidth=1, label=region_name)
    
    if not selected_area_data.empty:
        plt.plot(selected_area_data['Week'], selected_area_data[param], color='red', linewidth=3, marker='o', label=f"{area_name} (обрана)")
    
    plt.title(f"Порівняння {param} для області {area_name} з іншими областями", fontsize=14)
    plt.xlabel("Тиждень", fontsize=12)
    plt.ylabel(f"Середній {param}", fontsize=12)
    plt.grid(True)
    
    num_regions = len(other_areas_data['region_id'].unique()) + 1
    
    if num_regions > 12:
        legend_cols = 2
    else:
        legend_cols = 1
    
    plt.legend(
        title="Області", 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5), 
        fontsize=9,
        ncol=legend_cols,
        title_fontsize=12,
        frameon=True, 
        fancybox=True,
        shadow=True, 
        borderpad=1
    )
    
    plt.tight_layout()
    
    return plt.gcf()

st.title("FB-31 Gek VHI Data Analysis")
st.write("Додаток для аналізу даних VHI (Vegetation Health Index) для областей нашої країни")

data_path = st.text_input("Шлях до директорії з даними", value="vhi_data")
load_button = st.button("Завантажити дані")

if load_button or st.session_state.data_loaded:
    with st.spinner('Завантаження даних...'):
        df = read_vhi_files(data_path)
        if df is not None:
            df = change_region_indices(df)
            st.session_state.data_loaded = True
            st.success(f"Дані успішно завантажено. Загальна кількість записів: {len(df)}")
        else:
            st.error("Упс. Не вдалося завантажити дані. Перевірте шлях до директорії.")

if st.session_state.data_loaded:
    if 'selected_param' not in st.session_state:
        st.session_state.selected_param = 'VHI'
    if 'selected_area' not in st.session_state:
        st.session_state.selected_area = 'Вінницька'
    if 'week_range' not in st.session_state:
        st.session_state.week_range = (1, 52)
    if 'year_range' not in st.session_state:
        st.session_state.year_range = (1982, 2024)
    if 'ascending_sort' not in st.session_state:
        st.session_state.ascending_sort = False
    if 'descending_sort' not in st.session_state:
        st.session_state.descending_sort = False
    
    if st.button("Скинути фільтри"):
        st.session_state.selected_param = 'VHI'
        st.session_state.selected_area = 'Вінницька'
        st.session_state.week_range = (1, 52)
        st.session_state.year_range = (1982, 2024)
        st.session_state.ascending_sort = False
        st.session_state.descending_sort = False
        st.success("Фільтри скинуто до початкових значень")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Параметри аналізу")
        
        st.session_state.selected_param = st.selectbox(
            "Оберіть часовий ряд",
            options=['VCI', 'TCI', 'VHI'],
            index=['VCI', 'TCI', 'VHI'].index(st.session_state.selected_param)
        )
        
        st.session_state.selected_area = st.selectbox(
            "Оберіть область",
            options=list(region_names.values()),
            index=list(region_names.values()).index(st.session_state.selected_area)
        )
        
        st.session_state.week_range = st.slider(
            "Виберіть інтервал тижнів",
            1, 52, st.session_state.week_range
        )
        
        st.session_state.year_range = st.slider(
            "Виберіть інтервал років",
            1982, 2024, st.session_state.year_range
        )
        
        st.session_state.ascending_sort = st.checkbox(
            "Сортувати за зростанням",
            value=st.session_state.ascending_sort
        )
        
        st.session_state.descending_sort = st.checkbox(
            "Сортувати за спаданням",
            value=st.session_state.descending_sort
        )
    
    with col2:
        filtered_data = filter_data(
            df,
            st.session_state.selected_param,
            st.session_state.selected_area,
            st.session_state.week_range,
            st.session_state.year_range,
            st.session_state.ascending_sort,
            st.session_state.descending_sort
        )
        
        tab1, tab2, tab3 = st.tabs(["Таблиця", "Графік", "Порівняння з іншими областями"])
        
        with tab1:
            if filtered_data is not None and not filtered_data.empty:
                st.write(f"Кількість записів після фільтрації: {len(filtered_data)}")
                st.dataframe(filtered_data[['Year', 'Week', 'VCI', 'TCI', 'VHI', 'region_name']])
            else:
                st.warning("Немає даних для відображення. Змініть параметри фільтрації.")
        
        with tab2:
            if filtered_data is not None and not filtered_data.empty:
                fig = plot_vhi_data(filtered_data, st.session_state.selected_param)
                if fig:
                    st.pyplot(fig)
            else:
                st.warning("Немає даних для візуалізації. Змініть параметри фільтрації.")
        
        with tab3:
            if df is not None and not df.empty:
                fig = plot_comparison_data(
                    df,
                    st.session_state.selected_param,
                    st.session_state.selected_area,
                    st.session_state.week_range,
                    st.session_state.year_range
                )
                if fig:
                    st.pyplot(fig)
            else:
                st.warning("Немає даних для візуалізації. Завантажте дані спочатку.")
else:
    st.info("Завантажте дані, щоб почати аналіз.")

with st.expander("Довідка"):
    st.markdown("""
    ### Про цей веб-додаток та трохи про ці абревіатури
    Цей додаток дозволяє аналізувати та візуалізувати дані VHI (Vegetation Health Index) для областей нашої країни.
    
    ### Параметри аналізу
    - **VCI (Vegetation Condition Index)** - характеризує стан рослинного покриву
    - **TCI (Temperature Condition Index)** - характеризує температурний режим
    - **VHI (Vegetation Health Index)** - комбінований індекс, який враховує VCI та TCI
    
    ### Інтерпретація VHI
    - VHI < 15 - екстремальна посуха
    - VHI < 35 - посуха різного ступеня інтенсивності
    - VHI > 60 - сприятливі умови для рослинності
    
    ### Відображення даних
    - На вкладці "Таблиця" відображаються відфільтровані дані у табличному вигляді
    - На вкладці "Графік" відображається динаміка обраного індексу для вибраної області
    - На вкладці "Порівняння з іншими областями" можна порівняти значення для обраної області з іншими областями
    """)