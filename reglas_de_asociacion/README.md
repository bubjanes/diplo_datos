# Reglas de Asociación

Brandon Janes
22 agosto 2019

Reglas de Asociación Práctico: Diplomatura de Ciencia de Datos
Valeria Rulloni, Georgina Flesia, Laura Alonso Alemany
***FaMAF|UNC***

Realicé el experimento de reglas de asociación usando Python 3.7 con datos de MovieLens, una empresa sin fines de lucro que hace recomendaciones de películas, usando el algoritmo de market basket research ```efficient apriori```. Por las restricciones de procesamiento que tiene mi computadora, usé la dataset ml_latest_sm (small) que contiene 100836 ratings hecho por 610 usuarios y 9724 películas. 

A pesar de los desafíos de procesamiento, pude observar mucho de este set de datos con respecto al comportamiento de su suporte, confianza, lift y convicción, las métricas principales de reglas de asociación. Este método ignora el *rating* que da el usuario y está basada en nada más que la lista de películas que han visto cada usuario. Busqué recomendaciones con lift alto, soporte baja y confianza alta. Como dice el notebook de RPubs, el parámetro que más nos interesa en la búsqueda de recomendaciones para películas es un lift alto. 

Viendo estas reglas que tienen lift relativamente alto, podemos ver patrones--causas latentes--muy interesantes que existen entre películas. Similar de los hallazgos de RPubs, las reglas entre películas y sus secuelas son las reglas más fuertes, e.g. las reglas entre películas de las series de the Matrix, Harry Potter, Back to the Future y Bourne tienen lift más grandes de todos. La primera regla fuerte que no sea parte de una serie es la regla entre WALL-E y Up, dos películas, secuenciales, de dibujo animado hecho por el mismo director Pete Doctor. También veo un patrón que hay lifts muy altos entre películas de superhéroes y ciencia ficción o viceversa. En particular la relación entre RoboCop y The Terminator, dos películas de robots/ inteligencia artificial, está muy fuerte. Por eso fans de estos géneros tienen alta probabilidad de descubrir recomendaciones adentro ese género que no han visto y que les va a gustar. 

Reglas que tienen más alto lift:
Harry Potter and the Goblet of Fire (2004) -> Harry Potter and the Prisoner of Azkaban (2004) 
(conf: 0.859, supp: 0.10, lift: 5.635, conv: 6.018)
WALL-E (2008) ―-> Up (2009)
(conf: 0.702, supp: 0.12, lift: 4.078, conv: 2.777)
Spiderman (2002) ―-> X2: X-Men United (2003), 
(conf: 0.5, supp: 0.1, lift: 4.013, conv: 1.751)
Blade Runner (1982) ―-> 2001: Space Odyssey (1968) 
(conf: 0.5, supp: 0.103, lift: 4.013, conv: 1.751)

Cuanto más bajo podamos mantener el soporte, más “interesantes” o útiles las reglas que podremos generar. Ya que vemos métricas de lifts altas, porque soporte alto significa que casi todos ven la película y el algoritmo no nos brinda mucha información nueva recomendar películas que han visto todos. Desafortunadamente mi máquina no me dejó meter el parámetro min_support debajo 0.1. Hubiera sido muy interesante ver el mínimo soporte posible, como 0.001, que nos puede dar información sobre las películas vistas por menos personas.

Hay muchos patrones que uno puede ver y claramente efficient apriori es un algoritmo muy poderoso. 



