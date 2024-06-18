
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Configuración de la página de Streamlit
st.set_page_config(page_title='Titanic', page_icon=':ship:', layout='wide', initial_sidebar_state='expanded')

# Título y texto de la aplicación
st.markdown("<h1 style='text-align: center;'>App para los análisis de datos del Titanic</h1>", unsafe_allow_html=True)
st.text('El Titanic fue el famoso transatlántico de personas más grande y lujoso de su época.')
st.text('Se hundió en su viaje inaugural en 1912 por el choque contra un iceberg en el Atlántico Norte.')
st.image('https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg', use_column_width=True)
st.sidebar.image('https://getwallpapers.com/wallpaper/full/8/3/2/998650-titanic-sinking-wallpaper-2506x1853-meizu.jpg', use_column_width=True)

# Definición de la barra lateral y el menú de opciones
with st.sidebar:
    selected = option_menu('Selecciona una opción:', ['Análisis de datos', 'Gráficos', 'Conclusiones'])

# Función para cargar y limpiar datos
def load_data():
    file_path = os.path.join(os.getcwd(), 'titanic.csv')
    if not os.path.exists(file_path):
        st.error('El archivo titanic.csv no se encuentra en el directorio actual.')
        return None
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()  # Eliminar duplicados
    data['Age'] = data['Age'].fillna(0)  # Rellenar valores nulos en Age con 0
    data['Age'] = data['Age'].astype(int)  # Convertir Age a entero
    data['Fare'] = data['Fare'] * 95.42  # Ajustar el precio del billete a valores actuales
    return data

# Sección de análisis de datos
if selected == 'Análisis de datos':
    st.title('Análisis de datos')
    st.header('Análisis de los datos del Titanic')
    titanic = load_data()
    if titanic is not None:
        st.write(((titanic.isnull().sum() / len(titanic)) * 100).apply(lambda x: f'{x:.2f}%').to_frame().T)
        titanic['Cabin'] = titanic['Cabin'].fillna('No hay datos')
        le = LabelEncoder()
        titanic['Cabin'] = le.fit_transform(titanic['Cabin'])
        titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
        titanic['Embarked'] = le.fit_transform(titanic['Embarked'])
        st.dataframe(titanic)

# Sección de gráficos
if selected == 'Gráficos':
    st.title('Gráficos')
    st.header('Gráficos del Titanic')
    titanic = load_data()
    if titanic is not None:
        # Filtro por clase de pasajero
        st.subheader('Filtro por Clase de Pasajero')
        clase = st.selectbox('Selecciona la clase del pasajero', ['Todas', 1, 2, 3])
        if clase != 'Todas':
            titanic = titanic[titanic['Pclass'] == clase]
        st.write(titanic)

        # Filtro por supervivencia
        st.subheader('Filtro por Supervivencia') 
        supervivencia = st.selectbox('¿Sobrevivió?', ['Todos', 'Sí', 'No']) 
        if supervivencia == 'Sí':
            titanic = titanic[titanic['Survived'] == 1]
        elif supervivencia == 'No':
            titanic = titanic[titanic['Survived'] == 0]
        st.write(titanic)

        # Gráfico de barras de supervivencia por clase
        st.subheader('Supervivencia por Clase')
        st.write('Este gráfico muestra la tasa de supervivencia por clase de pasajero en el Titanic.')
        survival_by_class = titanic.groupby('Pclass')['Survived'].mean()
        st.bar_chart(survival_by_class)

        # Gráfico de barras de supervivencia por género
        st.subheader('Supervivencia por Género')
        st.write('Este gráfico muestra la tasa de supervivencia por género en el Titanic.')
        survival_by_gender = titanic.groupby('Sex')['Survived'].mean()
        st.bar_chart(survival_by_gender)

        # Gráfico de barras de supervivencia por edad
        st.subheader('Supervivencia por Edad')
        st.write('Este gráfico muestra la tasa de supervivencia por grupo de edad en el Titanic.')
        # Crear las categorías de edad
        edades = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        titanic['AgeGroup'] = pd.cut(titanic['Age'], bins=edades, right=False)
        # Convertir los grupos de edad a cadenas
        titanic['AgeGroup'] = titanic['AgeGroup'].astype(str)

        # Calcular la tasa de supervivencia por grupo de edad
        survival_by_age = titanic.groupby('AgeGroup')['Survived'].mean().reset_index()

        fig_bar = px.bar(survival_by_age, x='AgeGroup', y='Survived', title='Tasa de Supervivencia por Grupo de Edad', labels={'AgeGroup': 'Grupo de Edad', 'Survived': 'Tasa de Supervivencia'})
        st.plotly_chart(fig_bar)

        # Gráfico de violín de la Edad por Género y Clase
        st.subheader('Gráfico de violín de la Edad por Género y Clase')
        fig_violin = px.violin(titanic, x='Pclass', y='Age', color='Sex', title='Distribución de la Edad por Género y Clase', labels={'Pclass': 'Clase', 'Age': 'Edad', 'Sex': 'Género'})
        st.plotly_chart(fig_violin)

        # Gráfico de dispersión de la Edad vs Tarifa por Supervivencia
        st.subheader('Gráfico de dispersión de la Edad vs Tarifa por Supervivencia')
        fig_scatter = px.scatter(titanic, x='Age', y='Fare', color='Survived', labels={'Fare': 'Tarifa', 'Age': 'Edad', 'Survived': 'Supervivencia'}, title='Edad vs Tarifa por Supervivencia')
        st.plotly_chart(fig_scatter)

        # Gráfico de pastel
        st.subheader('Supervivencia General')
        fig, ax = plt.subplots(figsize=(2, 2))  # Tamaño más pequeño
        titanic['Survived'].value_counts().plot(
            kind='pie', 
            colors=['lightcoral', 'lightblue'],
            labels=['No sobrevivió', 'Sobrevivió'],
            autopct='%1.1f%%', 
            shadow=True, 
            ax=ax 
        )
        st.pyplot(fig)

# Sección de conclusiones
if selected == 'Conclusiones':
   
    st.title('Conclusiones')
    st.header('Conclusiones finales del análisis del Titanic')
    st.text("Aunque ya lo he ido comentando anteriormente con los gráficos, podemos concluir que:")
    st.text("- La tasa de supervivencia fue mayor en primera clase que en segunda y tercera clase.")
    st.text("- La distribución de edad varía entre las diferentes clases y géneros.")
    st.text("- Los pasajeros que pagaron más por sus boletos tenían una mayor probabilidad de sobrevivir.")
    st.text("- La tasa de supervivencia fue mayor en las mujeres que en los hombres, 'las mujeres primero'.")
    st.text("Finalmente podemos afirmar que la supervivencia del Titanic estuvo fuertemente influenciada por la clase del pasajero o el precio de su boleto, pero sobre todo por su sexo.")
    st.text('Su trágico final tuvo profundas repercusiones en seguridad marítima y en la percepción de la tecnología y la ingeniería,')
    st.text('pero sobre todo una huella indeleble en la cultura popular.')
    st.text('(Ha sido objeto de numerosas investigaciones, libros, películas y documentales.)')
    st.title("Vídeo de la película Titanic") 
    st.text("A continuación mostramos un pequeño vídeo con un fragmento corto de la famosa película 'Titanic'")
    st.text("La película Titanic es una de las más famosas y taquilleras de la historia del cine.")
    video_file = open('video.mp4', 'rb')  # Cargar el archivo de video.
    st.video(video_file)  # Video de la película Titanic.
