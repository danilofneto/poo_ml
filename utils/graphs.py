import pandas as pd
import matplotlib.pyplot as plt

class Grafico:

    def __init__(self, x, y, df):
        self.x = x
        self.y = y
        self.df = df

    def bar(self):
        plt.bar(self.x, self.y, data=self.df)
        print('Vai imprimir um gráfico de barras')
        plt.show()

    def scatter(self):
        plt.scatter(self.x, self.y, data=self.df)
        plt.show()

    def pie(self):

        labels = self.x
        sizes = self.y
        explode = (0, 0.1, 0, 0)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        plt.show()





