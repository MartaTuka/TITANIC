
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

#Configuración de la página de Streamlit.
st.set_page_config(page_title='Titanic', page_icon=':ship:', layout='wide', initial_sidebar_state='expanded') 
st.markdown("<h1 style='text-align: center;'>App para los análisis de datos del Titanic</h1>", unsafe_allow_html=True)
st.text('El titanic fue el famoso transatlántico de personas más grade y lujoso de su época.')
st.text('se hundió en su viaje inaugural en 1912 por el choque contra un Iceberg en el Atlántico Norte.') 
st.image('https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg', use_column_width=True)

#Configuración de la barra lateral y carga de datos.
st.sidebar.write('### Cargar datos')
st.sidebar.title('Menú') 
st.sidebar.image('https://getwallpapers.com/wallpaper/full/8/3/2/998650-titanic-sinking-wallpaper-2506x1853-meizu.jpg', use_column_width=True)
st.sidebar.write('Selecciona la opción deseada para cargar los datos del Titanic')
# uploaded_file = st.sidebar.file_uploader('Cargar archivo CSV', type='csv')


st.title("Análisis del Titanic")
#Lectura y limpieza de datos:
titanic = pd.read_csv(os.path.join(os.getcwd(), 'titanic.csv'))  #Cargar datos
titanic = titanic.dropna() #Eliminar valores nulos
titanic = titanic.drop_duplicates() #Eliminar duplicados
titanic['Age'] = titanic['Age'].fillna(0) #Rellenamos los valores nulos con 0.
titanic['Age'] = titanic ['Age'].astype(int) #Convertimos la edad a entero.
st.write("Este gráfico muestra la distribución de la edad por género y clase de los pasajeros del Titanic.")
st.dataframe(titanic) #Mostrar datos

#st.cache 

# Ajustar el precio del billete a valores actuales.
st.title("Ajustamos el precio del billete a valores actuales.")
st.markdown('Información extraída de: https://www.bankofengland.co.uk/monetary-policy/inflation/inflation-calculator')
titanic['Fare'] = titanic['Fare'] * 95.42
titanic.round(2)
st.write(titanic)
    
# Normalizar Fare
#scaler = StandardScaler()
#titanic['Fare'] = scaler.fit_transform(titanic[['Fare']])


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

#Creo diferentes gráficos.

# Crear el gráfico de barras:
fig_bar = px.bar(survival_by_age, x='AgeGroup', y='Survived',
                 title='Tasa de Supervivencia por Grupo de Edad',
                 labels={'AgeGroup': 'Grupo de Edad', 'Survived': 'Tasa de Supervivencia'})
fig_bar.write_html("bar.html")
    
# # Mostrar el gráfico de barras en la aplicación
st.plotly_chart(fig_bar)

st.title('Gráficos interactivos para una mayor comprensión.')
#Crear el gráfico de violín:
st.subheader('Gráfico de violín de la Edad por Género y Clase')
st.text('Creo un gráfico de violín para mostrar cómo se distribuyen las edades de los pasajeros en cada clase por género.')
st.text('Cada violín representa la densidad de probabilidad de edad de los pasajeros en una clase y género.')
st.text('Cuanto más ancho es el violín mayor es la densidad de pasajeros para esa edad.')
st.text('Además añado una caja dentro del violín para representar los valores estadísticos como mediana, rango intercuartílico')
st.text('y valores atípicos.')
st.text('La distribución de edad varía entras las diferentes clases.')
st.text('En primera clase la distribución por edad es más amplia y desplazada hacia edades mayores en comparación con tercera clase,')
st.text('donde hay pasajeros más jóvenes.')
st.text('1ª clase: En hombres los rangos de edad están en torno a 30-50 mientras que las mujeres tienen un violín más estrecho y')
st.text('desplazado hacia edades más jóvenes (20-40).')
st.text('También obeservamos que las mujeres de primera clase son más jóvenes que los hombres.')
st.text('2ª clase: La distribución de los hombres es más estrecha (20-40 años) y la de las mujeres es similar pero con tendencia') 
st.text('a edades menores.')
st.text('3ª clase: Los hombres tienen una amplia distribución en rangos de 10 a 50 años, mientras que en las mujeres está ')
st.text('desplazado a edades comprendidas entre 10 y 30.')
fig = px.violin(titanic, x='Pclass', y='Age', color='Sex', 
                title='Distribución de la Edad por Género y Clase',
                labels={'Pclass': 'Clase', 'Age': 'Edad', 'Sex': 'Género'})
#fig.write_html("violin.html") Guarda el gráfico en un archivo HTML
# fig.show()
# Mostrar el gráfico en la aplicación
#st.plotly_chart(fig)


#Leemos el archivo
with open('violin.html', 'r', encoding='utf-8') as f:
    html_string1 = f.read()
# Mostrar el contenido HTML en Streamlit
    st.components.v1.html(html_string1, height=600)


# Crear el gráfico de dispersión:
st.subheader('Gráfico de dispersión de la Edad vs Tarifa por Supervivencia')
st.text('Observamos si existe relación entre la tarifa y la edad de los pasajeros y su supervivencia.')
st.text('Los puntos indican si sobrevivieron o no.')
st.text('En el gráfico podemos observar una gran concentración de puntos entre los 20 y 40 años y las tarifas en el primer rango.')
st.text('La mayoría de puntos que representan a los supervivientes están en un grado de tarifas más altas.')
st.text('Podemos afirmar que aquellos que pagaron más por sus boletos tenían una mayor probabilidad de sobrevivir.')

fig1 = px.scatter(titanic, x='Age', y='Fare', color='Survived', 
                 labels={'Fare': 'Tarifa', 'Age': 'Edad', 'Survived': 'Supervivencia'},
                 title='Edad vs Tarifa por Supervivencia')
fig1.write_html("scatter.html")

#Leemos el archivo
with open('scatter.html', 'r', encoding='utf-8') as f:
    html_string2 = f.read()
# Mostrar el contenido HTML en Streamlit
    st.components.v1.html(html_string2, height=600)

#Filtro por apellidos:
st.subheader('Filro por Apellidos')
#Creo una columna para los apellidos de los pasajeros
titanic['LastName'] = titanic['Name'].apply(lambda x: x.split(',')[0])
# Crear un filtro interactivo para seleccionar un apellido.
selected_lastname = st.selectbox('Selecciona un apellido:', titanic['LastName'].unique())
# Filtrar los pasajeros por el apellido seleccionado.
filtered_passengers = titanic[titanic['LastName'] == selected_lastname]
# Mostrar los pasajeros con el apellido seleccionado y su estado de supervivencia
if not filtered_passengers.empty:
    st.write(f"Personas con el apellido '{selected_lastname}':")
    st.write(filtered_passengers[['Name', 'Survived']])
else:
    st.write(f"No se encontraron pasajeros con el apellido '{selected_lastname}'.")
    
st.title("Vídeo de la película Titanic") 
st.text("A continuación mostramos un pequeño vídeo con un fragmento corto de la famosa película 'Titanic'")
st.text("La película Titanic es una de las más famosas y taquilleras de la historia del cine.")
video_file = open('video.mp4', 'rb') #Cargar el archivo de video.
st.video(video_file) #Video de la película Titanic.

#conlusiones
st.title("Conclusiones finales")
st.text("Aunque ya lo he ido comentando anteriormente con los gráficos, podemos concluir que:")
st.text("- La tasa de supervivencia fue mayor en primera clase que en segunda y tercera clase.")
st.text("- La distribución de edad varía entre las diferentes clases y géneros.")
st.text("- Los pasajeros que pagaron más por sus boletos tenían una mayor probabilidad de sobrevivir.")
st.text("- Si nos centramos en la cabina, aquellos que tenían una cabina asignada (1ª y 2ª clase), tenían mayor supervivencia.")
st.text("- La tasa de supervivencia fue mayor en las mujeres que en los hombres, 'las mujeres primero'. ")
st.text("Finalmente podemos afirmar que la supervivencia del Titanic estuvo fuertemente influenciada por la clase del pasajero")
st.text("o el precio de su boleto, pero sobre todo por su sexo.")
st.text('Su trágico final tuvo profundas repercusiones en seguridad marítima y en la percepción de la tecnología y la ingeniería,')
st.text('pero sobre todo una huella indeleble en la cultura popular.')
st.text('(Ha sido objeto de numerosas investigaciones, libros, películas y documentales.)')
