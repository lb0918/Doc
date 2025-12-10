import csv
import pandas as pd
import numpy as np
import ast
import os
import re

class ADC_MRI_extractor:

    def __init__(self,path):
        for file in os.listdir(path):
            match = re.match(r"protocol_(\d+)", file)
            if match:
                self.code =  match.group(1)
        self.path = path
        


    def path_generator(self,table):
        path = self.path+"/"+table+"_"+str(self.code)+".csv"
        return path
    
    def data_extractor(self):
        input_file = self.path+f"/protocol_{self.code}.csv"
        code = self.code

        with open(input_file) as fin:
            reader = list(csv.reader(fin))
        titles_list = ["Study Parameters", "Additional Calculations","Protocol Conformance","Calculations Included",  "ADC VOI Statistics", "Chart Data"]
        counter_list = [0 for x in range(len(titles_list))]
        output_file_list = []
        reader_list = []
        print(len(titles_list))
        for x in range(len(titles_list)):
            print(x)
            output_file = self.path+"/"+titles_list[x]+"_"+str(code)+".csv"
            output_file_list.append(output_file)
            for row in reader:
                counter_list[x] += 1
                if row:
                    if row[0] == titles_list[x]:
                        break

        for x in range(len(titles_list)):
            if titles_list[x] == "Additional Calculations":
                new_reader_ = reader[counter_list[x]-1:counter_list[x+1]-1]
                indices = [i for i, val in enumerate(new_reader_[0]) if val.startswith("Temperature")]
                if len(indices) == 1:
                    new_reader_[0][indices[0]] = "Temperature"
                else:
                    for z in range(len(indices)):
                        new_reader_[0][indices[z]] = f"Temperature{z+1}"
            if x+1 == len(titles_list):
                print("dernier!")
                new_reader_ = reader[counter_list[x]-1:]
            elif x == 0 :
                new_reader_ = reader[counter_list[x]-1:counter_list[x+1]-1]
            else:
                new_reader_ = reader[counter_list[x]-1:counter_list[x+1]-1]
            reader_list.append(new_reader_)
        for i, bloc in enumerate(reader_list):
            if bloc and bloc[0] and bloc[0][0] == "VOI Statistics":
                for j, ligne in enumerate(bloc[2:]):
                    for k, valeur in enumerate(ligne):
                        try:
                            if valeur.isdigit() or (valeur.startswith('-') and valeur[1:].isdigit()):
                                val = int(valeur)
                            else:
                                val = round(float(valeur), 3)
                            reader_list[i][j+2][k] = val   
                        except ValueError:
                            pass
        for x in range(len(titles_list)):
            with open(output_file_list[x], "w", newline='', encoding='utf-8') as fout:
                writer = csv.writer(fout)
                writer.writerows(reader_list[x])
        print("Les tableaux sont maintenant séparés dans différents fichiers csv!")


    def get_temp(self,column = "Temperature"):
        path = self.path_generator("Additional Calculations")
        table = pd.read_csv(path)
        try:
            return table[column][0]
        except:
            return table["Temperature2"][0]

    def ADC_VOI_Statistics(self,label_recherche, column):
        """
        Returns the value of column for the the line specified by the argument "Label" == label_recherche
        """
        path = self.path_generator("ADC VOI Statistics")
        with open(path, newline='', encoding="utf-8") as f:
            lecteur = csv.DictReader(f)
            for ligne in lecteur:
                if ligne["Label"].strip() == str(label_recherche):
                    return ligne[column]
        raise ValueError(f"Aucune ligne avec label={label_recherche} trouvée.")
    
    def VOI_Statistics_vLB(self,slice,ROI,b_values,column):
        """
        Returns the value of column for the the line specified by the argument "Label" == label_recherche
        """
        path = self.path+f"/ADC_metrics/ADC_metrics_slice={slice}.csv"
        df = pd.read_csv(path)
        valeur = df.loc[(df['ROI'] == ROI) & (df['b_values'] == b_values), column].values[0]
        return valeur
    
    def get_SNR(self,slice,ROI,b_value,column):
        """
        Returns the value of column for the the line specified by the argument "Label" == label_recherche
        """
        path = self.path+f"/SNR_data/SNR_b={b_value}_slice={slice}.csv"
        df = pd.read_csv(path)
        valeur = df.loc[(df['ROI'] == ROI), column].values[0]
        return valeur

    
    def Additional_Calculations(self, column):
        """
        Returns the specified column
        """
        path = self.path_generator("Additional Calculations")
        table = pd.read_csv(path)
        return ast.literal_eval(table[column][0])
    
    def Chart_Data(self, column, label=1, metric = "ADC"):
        """
        Returns the specified column
        """
        path = self.path_generator("Chart Data")
        if column == "Slice Heights":
            table = pd.read_csv(path)
            return table[column]
        else:
            table = pd.read_csv(path)
            column_new = f"Vial {label} - {metric} Value vs Axial Position"
            return table[column_new]
        
    
    def Study_Parameters(self, column):
        path = self.path_generator("Study Parameters")
        table = pd.read_csv(path)
        return table[column]
    
