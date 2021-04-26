def df_sdf_files (direction_file):
    
    """
    This funtion associate the BindingDB_Reactant_set_id
    with the NAME, Target Source Organism According to 
    Curator or DataSource and IC50 (nM) value of .sdf files
    the Binding Database.
    """
    
    with open (direction_file, "r", encoding = "utf-8") as f:
        molecules3D_1 = f.readlines()

    BindingDB_Reactant_set_id =[line.strip("\n") for idx, line in enumerate(molecules3D_1) if (molecules3D_1[idx-1]).startswith("> <BindingDB Reactant_set_id>")]
    NAME = [line.strip("\n") for idx, line in enumerate(molecules3D_1) if (molecules3D_1[idx-1]).startswith("$$$$")]
    IC50_nM = [line.strip("\n") for idx, line in enumerate(molecules3D_1) if (molecules3D_1[idx-1]).startswith("> <IC50")]
    Target_Source_Organism = [line.strip("\n") for idx, line in enumerate(molecules3D_1) if (molecules3D_1[idx-1]).startswith("> <Target Source Organism According to Curator or DataSource>")]
    Dict = []

    for i in range (len(BindingDB_Reactant_set_id)):
            Dict.append({"BindingDB_Reactant_set_id": BindingDB_Reactant_set_id[i],
                         "Name": Name[i],
                         "Target Source Organism According to Curator or DataSource": Target_Source_Organism[i],
                         "IC50 (nM)": IC50_nM[i]
                        })
    df_molecules = pd.DataFrame(Dict)        

    return df_molecules


def filter_VBA_BO (Data_Frame, Target_Source_Organism):
    """
    This funtion filter Data Frame by Funtionality of 
    biological activity values to calculate pIC50 
    and to interpretate the results of the model, also
    filter by Target Source Organism in str format, It
    return a Data Frame ready to add the molecular
    descriptors and calcualte the prediction model and
    some statistics parameters
    """
    Molecules_IC50 = []
    
    for i in range(len(Data_Frame)):
        if Data_Frame.loc[i,"Target Source Organism According to Curator or DataSource"] == Organism and not Data_Frame.loc[i,"IC50 (nM)"].startswith("<") and not Data_Frame.loc[i,"IC50 (nM)"].startswith(">" or "<") and not Data_Frame.loc[i,"IC50 (nM)"].startswith("nan")  and not Data_Frame.loc[i,"IC50 (nM)"].startswith(" "):

            Molecules_IC50.append ({"BindingDB_Reactant_set_id": df_IC50.loc[i,"BindingDB Reactant_set_id"],
                                    "IC50 (nM)": float(df_IC50.loc[i,"IC50 (nM)"]),
                                    "pIC50": -np.log10((float(df_IC50.loc[i,"IC50 (nM)"])*10**(-9)))
                                   })


    df_Molecules_IC50 = pd.DataFrame(Molecules_IC50)

    return df_Molecules_IC50, df_Molecules_IC50["pIC50"].describe()

def run ():
    df = df_sdf_files ("../ACETYLCHOLINESTERASE_3D.sdf") 

    Organism = "Electrophorus electricus"   
    df_filter, statistics = filter_VBA_BO (df, Organism) 

    print(statistics)





if __name__ == "__main__":
    run()    