from appJar import gui
import numpy as np
from data_processing import *
from trainer import *

class App:
    def __init__(self):
        self.app = gui("Classificador Tabajara", "640x480")
        self.currentFile = None

        self.app.addStatusbar()
        self.app.setStatusbar("Nenhum arquivo aberto.", 0)

        fileMenus = ["Abrir", "Fechar", "Sair"]
        fileMenusFuncs = [self.openFile, self.closeFile, self.exit]
        self.app.addMenuList("Arquivo", fileMenus, fileMenusFuncs)

        trainMenu = ["KNN"]
        trainMenuFuncs = [self.knn]
        self.app.addMenuList("Treinar", trainMenu, trainMenuFuncs)

        processMenu = ["Undersample"]
        processMenuFuncs = [self.undersample]

        self.app.addMenuList("Processar", processMenu, processMenuFuncs)


    def run(self):
        self.app.go()
    
    def exit(self):
        self.app.stop()
        exit()

    def openFile(self):
            
        self.currentFilepath = self.app.openBox("Selecione um dataset", fileTypes=[("datasets", "*.csv")])

        self.app.startSubWindow("openFileOptions", modal=True)

        self.app.addLabelEntry("Nome da coluna alvo: ")
        
        def ok():
            self.targetColumn = self.app.getEntry("Nome da coluna alvo: ")
            self.app.destroySubWindow("openFileOptions")
            self.importDataset()

        self.app.addButton("Ok", ok)
        
        self.app.go(startWindow="openFileOptions")
        
    def importDataset(self):
        self.dataset = getDataset(self.currentFilepath)
        self.app.setStatusbar("Arquivo aberto: " + self.currentFilepath, 0)
        self.X, self.y = getXAndY(self.dataset, self.targetColumn)
        self.X_train, self.X_test, self.y_train, self.y_test = getTrainingAndTesting(self.X, self.y, 0)

    def undersample(self):
        self.app.startSubWindow("undersampleOptions", modal=True)

        self.app.addLabelEntry("Quantidade de samples: ")
        
        def ok():
            samples = self.app.getEntry("Quantidade de samples: ")
            values = getAllColumnValues(self.dataset, self.targetColumn)
            self.dataset = undersample(self.dataset, int(samples), self.targetColumn, values)
            self.app.destroySubWindow("undersampleOptions")
            self.importDataset()

        self.app.addButton("Ok", ok)
        
        self.app.go(startWindow="undersampleOptions")

    def closeFile(self):
        self.currentFile.close()
        self.currentFilepath = ""
        self.app.setStatusbar("Nenhum arquivo aberto.", 0)

    def knn(self):
        self.app.startSubWindow("knnOptions", modal=True)

        self.app.addLabelEntry("N: ")
        self.app.setEntry("N: ", "3")
        self.app.addLabelScale("Porcentagem usada para testes: ")
        self.app.setScaleRange("Porcentagem usada para testes: ", 0, 100, curr=20)
        self.app.showScaleValue("Porcentagem usada para testes: ", show=True)
        
        def ok():
            n = int(self.app.getEntry("N: "))
            self.test_size = float(self.app.getScale("Porcentagem usada para testes: "))/100.0
            
            knn = KNNTrainer()
            configs = {'test_size':self.test_size, 'n':n}
            knn.setConfigs(configs)

            self.X_train, self.X_test, self.y_train, self.y_test = getTrainingAndTesting(self.X, self.y, self.test_size)

            knn.fit(self.X_train, self.X_test, self.y_train, self.y_test)

            print knn.accuracy

            self.app.destroySubWindow("knnOptions")
            self.importDataset()

        self.app.addButton("Ok", ok)
        
        self.app.go(startWindow="knnOptions")

    

app = App()

app.run()

