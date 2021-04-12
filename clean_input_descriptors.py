# IMPORTAR O ARQUIVO

import pandas as pd
import numpy as np
import math

#Carregado do arquivo da base de dados Binding Data base(https://www.bindingdb.org/bind/index.jsp)
#para acetylcholinesterase

moleculas=pd.read_csv("ACHE_BD.csv")
moleculas



# LIMPEZA DO ARQUIVO
# visualização da coluna pIC50


#Limpeza dos arquivo segundo o IC50 tirando os caracteres especiais, espacios vazios e "not a number"
#Geração da coluna pIC50
moleculas_IC50=[]

for i in range(len(moleculas)):
    if not (str(moleculas.loc[i,"IC50 (nM)"]).startswith("<")) and not (str(moleculas.loc[i,"IC50 (nM)"]).startswith(">")) and not (str(moleculas.loc[i,"IC50 (nM)"]).startswith("nan"))  and not (str(moleculas.loc[i,"IC50 (nM)"]).startswith(" ")):
        moleculas_IC50.append({"BindingDB Reactant_set_id":moleculas.loc[i,"BindingDB Reactant_set_id"],
                                "Ligand SMILES":moleculas.loc[i,"Ligand SMILES"],
                            "Ligand InChI":moleculas.loc[i,"Ligand InChI"],
                              "Ligand InChI Key":moleculas.loc[i,"Ligand InChI Key"],
                              "BindingDB MonomerID":moleculas.loc[i,"BindingDB MonomerID"],
                              "BindingDB Ligand Name":moleculas.loc[i,"BindingDB Ligand Name"],
                              "Target Name Assigned by Curator or DataSource":moleculas.loc[i,"Target Name Assigned by Curator or DataSource"],
                              "IC50 (nM)":float(moleculas.loc[i,"IC50 (nM)"]),
                               "pIC50":-np.log10(float(moleculas.loc[i,"IC50 (nM)"])*(10**-9))})

df_moleculas_pIC50=pd.DataFrame(moleculas_IC50)

print("O numero de moleculas é:  {}".format(len(moleculas)))
print("O número de moleculas que tem IC50 especifico é: {}".format(len(moleculas_IC50)))
print(df_moleculas_pIC50["pIC50"].describe())



# EXPORTAR O ARQUIVO PARA GERAR OS DESCRITORES MOLECULARES


#Preparaçãodos dados para gerar os descritores moleculares em AlvaDesc
selection=["Ligand SMILES","BindingDB MonomerID"]
#Adição das colunas dos codigos smiles e identificação das moelculas
df_m_pIC50_Descriptors_input=df_moleculas_pIC50[selection]
df_m_pIC50_Descriptors_input


#Salvado do arquivo input para descritores
df_m_pIC50_Descriptors_input.to_csv("df_m_pIC50_Descriptors_input.smi", sep="\t", index=False, header=False)
