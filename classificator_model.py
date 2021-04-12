# CARREGAR OS DESCRITORES MOLECULARES E FAZER A LIMPEZA DA MATRIZ



#Carregado do output dos descritores moleculares
df_m_pIC50_AlvaDescriptors_output=pd.read_csv("df_m_pIC50_AlvaDescriptors_output.txt",sep="\t")
#limpeza dos dados gerados por AlvaDesc
df_m_pIC50_AlvaDescriptors_op_clean=df_m_pIC50_AlvaDescriptors_output.replace("na",0)

for i in range(df_m_pIC50_AlvaDescriptors_op_clean.shape[0]):
    for j in range(df_m_pIC50_AlvaDescriptors_op_clean.shape[1]):
        numero_pontos=str(df_m_pIC50_AlvaDescriptors_op_clean.iloc[i,j]).count(".")
        if numero_pontos >=2:
            df_m_pIC50_AlvaDescriptors_op_clean.iloc[i,j]=float(df_m_pIC50_AlvaDescriptors_op_clean.iloc[i,j].replace(".",""))


# NORMALIZAÇÃO DA COLUNA pIC50

#normalização dos valores de pIC50 em uma escala de base 1
def norm_pIC50_col(data):
    norm=[]
    for elem in data["pIC50"]:
        if elem<1:
            elem=1
        norm.append(elem)
    data["pIC50_norm"]=norm
    x=data.drop("pIC50",1)
    return x

df_m_pIC50_norm=norm_pIC50_col(df_moleculas_pIC50)


#Descriçaõ da coluna nova de pIC50
print(df_m_pIC50_norm["pIC50_norm"].describe())


# CLASSIFICAÇÃO DA BIOATIVIDADE DAS MOLECULAS


#Classificação da bioatividade
Bioactivity_class=[]
ativas=0
inativas=0
for i in df_m_pIC50_norm.loc[:,"pIC50_norm"]:
    if i>=7.2:
        Bioactivity_class.append("Active")
        ativas+=1
    elif  i<=4.5:
        Bioactivity_class.append("Inactive")
        inativas+=1
    else:
        Bioactivity_class.append("Intermediary")
#Adição da coluna ao DF principal
df_m_pIC50_norm["Bioactivity_class"]= Bioactivity_class
#Salvado do arquivo xlsx
df_m_pIC50_norm.to_excel("df_m_pIC50_norm.xlsx")
print("O numero de moleculas ativas é:  {}".format(ativas))
print("O numero de moleculas inativas é:  {}".format(inativas))


# GERAÇÃO DO DATA FRAME 2 CLASSES PARA O APRENDIZADO DE MAQUINA

#Preparação dos dados para Machine Learning e graficos
#juntar os descritores com o DF principal
df_AlvaDescriptors=df_m_pIC50_AlvaDescriptors_op_clean.drop("No.",1).drop("NAME",1)
df_ML=df_m_pIC50_norm.merge(df_AlvaDescriptors,left_index=True,right_index=True)


#Geraçaõ da faixa de valores entre especies ativas e inativas para o nosso modelo
df_ML_2class=df_ML[df_ML.Bioactivity_class != "Intermediary"]
df_ML_2class


# GRAFICO DAS DUAS ESPECIES


#Grafico # de especies ativas e # de especies inativas

import plotly.graph_objects as go
colors = ['black','crimson']
fig = go.Figure([go.Bar(x=["Inactive","Active"], y=[inativas,ativas],
                marker_color=colors,
                    width=[0.4,0.4])])

fig.update_layout(
    xaxis=dict(
        title="Bioactivity class",
        titlefont_size=20,
        tickfont_size=20
    ),
    yaxis=dict(
        title='No. Moleculas',
        titlefont_size=20,
        tickfont_size=20
    ))
fig.show()


# GRAFICO ESPECIES ATIVAS E A FAIXA REPRESENTADA


#Grafico # de especies ativas e # de especies inativas com a faixa representada e a atividade

colors = ['black',"red"]
fig = go.Figure()
fig.add_trace(go.Bar(x=["Inactive"], y=[3.5],
                base=[1],
                marker_color='black',
                name='Inactive',
                    width=[0.4]))
fig.add_trace(go.Bar( x=["Active"],y=[1.8],
                base=[7.2],
                marker_color='crimson',
                name='Active',
                width=[0.4],
                     ))

fig.update_layout(
    xaxis=dict(
        title="Bioactivity class",
        titlefont_size=20,
        tickfont_size=20
    ),
    yaxis=dict(
        title='pIC50',
        titlefont_size=25,
        tickfont_size=20
    ))
fig.show()


# DISTRIBUIÇÃO DOS DADOS SEGUNDO O PCA

# Distribuição dos dados segundo o PCA (Machine Learning Não Supervisionado)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#pegar os dados dos dos descritores do DF principal
df_ML_2class_NS=df_ML_2class.iloc[:,10:]
# Pre-procesamento "centrar na media para
#colocar os dados no origem para pegar a maxima variança dos dados
#y  dividir pelo desvio padrão para dar o mesmo peso para os dados
#de cada varaivel para colocar na mesma faixa de valores" a AUTO ESCALAR OS DADOS
Xa=StandardScaler().fit_transform(df_ML_2class_NS)
Xa.shape


# APLICAÇÃO DO PCA
# variança dos dados e variança acumulada


#foi feito o uso de 475 componentes para garantir a representação do 90% dos dados
pca=PCA(n_components=475) #vou representar as 3977 colunas"descritores" em 475
pca.fit(Xa) #fit é para fazer o ajuste dos dados
print((pca.explained_variance_ratio_))#varianza explicada, cada nova coluna explica uma percentagem da varianza original

print(np.cumsum(pca.explained_variance_ratio_))# np. cumsum () suma acumulada
print(pca.singular_values_)#valores singulares que dão uma ideia da varianza que esta sendo explicada



L=pca.components_.T# cada linha do array representa os pesos de cada variavel original nas novas variaveis
T=pca.transform(Xa)


# GRAFICO EM 3D DISTRIBUIÇÃO DAS MOLECULAS E AGRUPAÇÃO ESPECIES ATIVAS E INATIVAS
#GRAFICO PCA


import plotly.express as px
import plotly.graph_objects as go

dados=pd.DataFrame(T)
dados.columns=["PC"+str(i+1) for i in range(dados.shape[1])]

fig=px.scatter_3d(dados,x="PC1",y="PC2",z="PC3",
                  color=df_ML_2class.iloc[:,9],
                  size=df_ML_2class.iloc[:,8]*3.5,
                  size_max=31,
                  opacity=1,
                  hover_data= [df_ML_2class.iloc[:,4],df_ML_2class.iloc[:,8],df_ML_2class.iloc[:,10]]
                 )
fig.show()
#grafico em 3D tendo enconta a cor como a classificação inicial das especies entre ativas e inativas para tentar escolher
#o modelo de classificação segundo a distribuição dos dados


# AGRUPAÇÃO DOS DESCRITORES MOLECULARES


#Grafico da matriz dos pesos para olhar a contribuição dos descritores e
#tentar relacionar os descritores que mais representan a atividade biologica
dados1=pd.DataFrame(L)
dados1.columns=["PC"+str(i+1) for i in range(dados1.shape[1])]
size=[10 for i in range(len(dados1))]
fig1=px.scatter_3d(dados1,x="PC1",y="PC2",z="PC3",
                   color=df_ML_2class_NS.columns,
                   size=size,
                   size_max=10,
                   opacity=1,
                   hover_data= [df_ML_2class_NS.columns])
fig1.show()


# PREPARAÇÃO PARA O APRENDIZADO DE MAQUINA SUPERVISIONADO


#Criação da coluna Binary_bioactivity para representar a atividade (0 para especie inativa
# e 1 para especie ativa)
Binary_bioactivity=[]
for i in df_ML_2class.loc[:,"Bioactivity_class"]:
    if i=="Active":
        Binary_bioactivity.append(1)
    else:
        Binary_bioactivity.append(0)
#Adição da coluna ao DF de duas calses

df_Xa=pd.DataFrame(Xa)
df_Xa["Binary_bioactivity"]=Binary_bioactivity


# FUNÇÕES KENNARD-STONE PARA ESCOLHER O TREINO E O TESTE


#seleção dos conjuntos treino e test metodo Kennard-Stone######################################################


from scipy.spatial.distance import cdist
def max_min_distance_split(distance, train_size):
    """sample set split method based on maximun minimun distance, which is the core of Kennard Stone
    method
    Parameters
    ----------
    distance : distance matrix
        semi-positive real symmetric matrix of a certain distance metric
    train_size : train data sample size
        should be greater than 2
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero-based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero-based
    """

    select_pts = []
    remaining_pts = [x for x in range(distance.shape[0])]

    # first select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    select_pts.append(first_2pts[0])
    select_pts.append(first_2pts[1])

    # remove the first 2 points from the remaining list
    remaining_pts.remove(first_2pts[0])
    remaining_pts.remove(first_2pts[1])

    for i in range(train_size - 2):
        # find the maximum minimum distance
        select_distance = distance[select_pts, :]
        min_distance = select_distance[:, remaining_pts]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts




def kennardstone(spectra, test_size=0.25, metric='euclidean', *args, **kwargs):
    """Kennard Stone Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    --------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """

    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        distance = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts
Xa=np.array(df_Xa)
Set_Train_Test=kennardstone(Xa)


#separação dos dados em train e test


Set_Train=Set_Train_Test[0]
Set_Test=Set_Train_Test[1]

#Separação dos conjuntos Train e Test  e suas respetivas variaveis X e y
X_train=[]
y_train=[]
X_test=[]
y_test=[]
No_mol_train=0
No_mol_test=0
for i in range (len(df_Xa)):
    if i in Set_Train:
        No_mol_train+=1
        X_train.append(df_Xa.iloc[i,:-1])
        y_train.append(df_Xa.iloc[i,-1])
    if i in Set_Test:
        No_mol_test+=1
        X_test.append(df_Xa.iloc[i,:-1])
        y_test.append(df_Xa.iloc[i,-1])

print("O numero de moleculas para o Train é:  {}".format(No_mol_train))
print("O numero de moleculas para o Test é:  {}".format(No_mol_test))
df_ML_2class.iloc[i,10:-1]


# FUNÇÃO MATRIZ CONFUÇÃO


#matrix confução para obter um reusltado mais facil de visualizar
###############################################################################
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
###############################################################################


#importar o metodo classificatorio LDA foi escolhido para maximizar as
#distancias entre as duas classes que nos temos e para minimizar as distancias
#entre as amostras da mesma classe.


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
#Dados separados em train e teste em data frame e fazer auto escalamento
df_X_train=pd.DataFrame(X_train)
df_y_train=pd.DataFrame(y_train)
df_X_test=pd.DataFrame(X_test)
df_y_test=pd.DataFrame(y_test)
#Dados totais
X=df_X_train.append(df_X_test)
y=df_y_train.append(df_y_test)
lda.fit(X_train,y_train)
#como se for o meu R^2
print(lda.score(X_train,y_train))
#prova de predição
print(lda.predict_proba(X_train))
#variancia segundo os atributos ( numero de calses-1)
print(lda.explained_variance_ratio_)


y_pred = lda.fit(X_train, y_train).predict(X_test)
# Plot non-normalized confusion matrix
class_names = np.array(["Inactive","Active"])
plot_confusion_matrix(np.array(y_test), y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


#carateristicas do modelo calssificatorio segundos
#valores da matriz confussão para o LDA
vn=167
vp=241
fn=88
fp=16
Sensibilidade=100*vp/(vp+fn)
Seletividade=100*vn/(vn+fp)
Exatidão=100*(vp+vn)/(vp+fn+fp+vn)
​
print("Os parametros do Modelo Classificatorio criado são:\nSensibilidade de {
:.2f}% \nSeletividade de {:.2f}% \nExatidão de {
:.2f}%".format(Sensibilidade,Seletividade,Exatidão))


# ANALISE PELO METODO QDA


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train)
print(qda.score(X_train,y_train))
print(qda.predict_proba(X_test))

#MATRIZ CONFUSSÃO QDA
y_pred = qda.fit(X_train, y_train).predict(X_test)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


#carateristicas do modelo calssificatorio segundos
#valores da matriz confussão para o QDA


vn=153
vp=265
fn=64
fp=30
Sensibilidade=100*vp/(vp+fn)
Seletividade=100*vn/(vn+fp)
Exatidão=100*(vp+vn)/(vp+fn+fp+vn)


# ANALISE PELO METODO SVM


from sklearn.svm import SVC
svc = SVC(C=900,gamma="scale",decision_function_shape= 'ovo', kernel= 'rbf')
svc.fit(X_train,y_train)
print(svc.score(X_train,y_train))
print(svc.predict(X_train))


#MATRIZ CONFUSSÃO METODO SVM


y_pred = svc.fit(X_train, y_train).predict(X_test)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


#carateristicas do modelo calssificatorio segundos
#valores da matriz confussão para o SVM


vn=179
vp=250
fn=79
fp=4
Sensibilidade=100*vp/(vp+fn)
Seletividade=100*vn/(vn+fp)
Exatidão=100*(vp+vn)/(vp+fn+fp+vn)

print("Os parametros do Modelo Classificatorio criado são:\nSensibilidade de {
:.2f}% \nSeletividade de {:.2f}% \nExatidão de {
:.2f}%".format(Sensibilidade,Seletividade,Exatidão))


#CROSS VALIDATION PARA ENCONTRAR OS PARAMETROS OTIMOS PARA O METODO with
#GridSearchCV


from sklearn.model_selection import GridSearchCV
parameters_SVC = {"C":range(40,900)
                  ,"kernel":("rbf","poly","sigmoid"),
                  "gamma":("scale","auto"),
                  "decision_function_shape":("ovo","ovr")
                                            }
SVC_grid = GridSearchCV(svc, parameters_SVC, cv=4)
SVC_grid.fit(X_train,y_train)
print(SVC_grid.score(X_train,y_train))
print(SVC_grid.predict(X_test))
print(SVC_grid.best_params_)


# ANALISE PELO METODO MLPClassifier


from sklearn.neural_network import MLPClassifier
MLPC = MLPClassifier(solver='adam', alpha=1e-10,
                    hidden_layer_sizes=(70,60), random_state=200)
MLPC.fit(X_train, y_train)


MLPC.predict(X_test)

MLPC.predict_proba(X_test)


#MATRIZ CONFUSSÃO METODO MLPClassifier


y_pred = MLPC.fit(X_train, y_train).predict(X_test)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


#carateristicas do modelo calssificatorio segundos
#valores da matriz confussão para o MLPClassifier


vn=177
vp=302
fn=27
fp=6
Sensibilidade=100*vp/(vp+fn)
Seletividade=100*vn/(vn+fp)
Exatidão=100*(vp+vn)/(vp+fn+fp+vn)

print("Os parametros do Modelo Classificatorio criado são:\nSensibilidade de {
:.2f}% \nSeletividade de {:.2f}% \nExatidão de {
:.2f}%".format(Sensibilidade,Seletividade,Exatidão))


#CROSS VALIDATION PARA ENCONTRAR OS PARAMETROS OTIMOS PARA O METODO with
#GridSearchCV


parameters_MLPC = {"hidden_layer_sizes":(range(50,75,5),range(50,75,5)),
                   "solver":["adam"],
                   "alpha":[0.0001,0.0000000001],
                   "random_state":[1,200]}
MLPC_grid = GridSearchCV(MLPC, parameters_MLPC, cv=4)
MLPC_grid.fit(X_train,y_train)
print(MLPC_grid.score(X_train,y_train))
print(MLPC_grid.predict(X_test))
print(MLPC_grid.best_params_)


# ANALISE PELO METODO K-NN


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=246)
knn.fit(X_train,y_train)
print(knn.score(X_train,y_train))
print(knn.predict_proba(X_test))


#MATRIZ CONFUSSÃO METODO K-NN


y_pred = knn.fit(X_train, y_train).predict(X_test)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


#carateristicas do modelo calssificatorio segundos
#valores da matriz confussão para o K-NN


vn=170
vp=184
fn=145
fp=13
Sensibilidade=100*vp/(vp+fn)
Seletividade=100*vn/(vn+fp)
Exatidão=100*(vp+vn)/(vp+fn+fp+vn)

print("Os parametros do Modelo Classificatorio criado são:\nSensibilidade de {
:.2f}% \nSeletividade de {:.2f}% \nExatidão de {
:.2f}%".format(Sensibilidade,Seletividade,Exatidão))


#CROSS VALIDATION PARA ENCONTRAR OS PARAMETROS OTIMOS PARA O METODO with
#GridSearchCV


knn_grid = GridSearchCV(knn,{'n_neighbors':range(240,250)},cv=4)
knn_grid.fit(X_train,y_train)
print(knn_grid.score(X_train,y_train))
print(knn_grid.best_params_)
