import pandas as pd
import numpy as np
import nltk
import word2vec

from polyglot.text import Text


def similar_chars_counter(column_1, column_2):
    if len(column_1) > len(column_2):
        longest = column_1
    else:
        longest = column_2
    similar_chars = len(longest) -  nltk.edit_distance(column_1,column_2)
    return len(longest), similar_chars

def substringcalc(column_1, column_2):
    long =""
    short=""
    if len(column_1)>len(column_2):
        long = column_1
        short = column_2
    else:
        long = column_2
        short = column_1
    if short in long:
        return 1
    return 0


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")




def FLM1(header_1, header_2):
    similarity = np.empty([len(header_1), len(header_2)])
# 0.3 number of characters, 0.7 substring + w2v
    for i in range(len(header_1)):
        for j in range(len(header_2)):
            lon, sim = similar_chars_counter(header_1[i], header_2[j])
            similarity[i][j] = ("{:.2f}".format(round(sim / lon, 2)))
    word2vec.sentences.append(header_1)
    word2vec.sentences.append(header_2)
    model = word2vec.model_w2v()
    # print(word2vec.sentences)
    model.save('word2vec.model')
    #0.4 w2v + 0.6 substring
    i = 0
    while i < len(header_1):
        j = 0
        while j < len(header_2):
            instance_sim = (model.wv.similarity(header_1[i], header_2[j])) * 0.3
            instance_sim_2 = substringcalc(header_1[i], header_2[j]) * 0.7
            if instance_sim < 0:
                instance_sim = 0
            instance_sim_res = instance_sim + instance_sim_2
            if instance_sim_res > 1:
                instance_sim_res = 1
            similarity[i][j] ="{:.2f}".format(similarity[i][j]*0.3 +instance_sim_res*0.7)
            j = j + 1
        i = i + 1

    return similarity


def FLM2(types_1,types_2,lbls_1,lbls_2):
    similarity = np.empty([len(types_1), len(types_2)])
    for i in range(len(types_1)):
        for j in range(len(types_2)):
            if types_1[i] == types_2[j]:
                similarity[i][j] =1
            else:
                similarity[i][j] = 0
    for i in range(len(lbls_1)):
        for j in range(len(lbls_2)):
            if lbls_1[i] == lbls_2[j]:
                similarity[i][j] = "{:.2f}".format(similarity[i][j]*0.3 +0.7)
    return similarity



def SLM1(header_1,header_2,types_1,types_2,lbls_1,lbls_2):

    flm1 = FLM1(header_1,header_2)
    flm2 = FLM2(types_1,types_2,lbls_1,lbls_2)
    slm =  np.empty([len(header_1), len(header_2)])
    for i in range(len(header_1)):
        for j in range(len(header_2)):
            slm[i][j]=("{:.2f}".format(round(flm1[i][j]*0.7+flm2[i][j]*0.3, 2)))
    threshold =0.4
    row =0
    col =0
    res_1 = []
    res_2 =[]
    sim = []
    for i in range(len(header_1)):
        max = 0
        for j in range(len(header_2)):
            if slm[i][j]>max:
                max = slm[i][j]
                row = i
                col =j
        res_1.append(header_1[row])
        res_2.append(header_2[col])
        sim.append(slm[row][col])
    temp_1 = res_1
    temp_2 = res_2
    temp_sim = sim

    for i in range(len(res_1)):
        for j in range(len(res_1)):
            if res_1[i] == res_1[j] and i!=j:
                if sim[i]>sim[j]:
                    #### mutlaq means deleted in canupus language, i chose this word because
                    #### there is a very small chance of someone naming a column in 'mutlaq'
                    temp_1[j] = 'mutlaq'
                    temp_2[j] = 'mutlaq'
                    temp_sim[j] = 'mutlaq'
                else:
                    temp_1[i] = 'mutlaq'
                    temp_2[i] = 'mutlaq'
                    temp_sim[i] = 'mutlaq'
    ### updating the original lists
    res_1 = temp_1
    res_2= temp_2
    sim=temp_sim
    ### reinitiallizing to perform the exact same operation on the second dataset
    temp_1 = res_1
    temp_2 = res_2
    temp_sim = sim
    for i in range(len(res_2)):
        for j in range(len(res_2)):
            if res_2[i] == res_2[j] and i != j:
                if sim[i] > sim[j]:
                    temp_1[j] = 'mutlaq'
                    temp_2[j] = 'mutlaq'
                    temp_sim[j] ='mutlaq'
                else:
                    temp_1[i] = 'mutlaq'
                    temp_2[i] = 'mutlaq'
                    temp_sim[i] = 'mutlaq'
    res_1 = temp_1
    res_2 = temp_2
    sim = temp_sim

    final=[]
    for i in range(len(res_1)):
        if res_1[i]!= 'mutlaq' and res_2[i]!='mutlaq' and sim[i]>threshold:
            final.append( res_1[i]+","+ res_2[i] )


    return final

def Evaluate(result, exact):
    tp = 0
    fp = 0
    fn =0

    for i in range(len(result)):
            if result[i] in exact:
                tp = tp + 1
            else:
                fp = fp+1
    for i in range(len(exact)):
        if exact[i] not in result:
            fn = fn+1
    p = tp/(tp+fp)
    r = tp /(tp+fn)
    f_1 = 2*((p*r)/(p+r))
    f_50= (1+pow(50,2))*((p*r)/(pow(50,2)*p+r))

    return p,r,f_1,f_50

def Match(dataset_1,dataset_2,exact_corres):
    d_1 = pd.read_csv(dataset_1, error_bad_lines=False)
    d_2 = pd.read_csv(dataset_2, error_bad_lines=False)
    exact = pd.read_csv(exact_corres, error_bad_lines=False)
    d_1.drop(d_1.columns[d_1.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    d_2.drop(d_1.columns[d_1.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    d1_cols = list(d_1)
    d2_cols = list(d_2)

    for i in range(len(d1_cols)):
        d1_cols[i].replace(' ', '')
        d1_cols[i] = d1_cols[i].lower()
    for i in range(len(d2_cols)):
        d2_cols[i].replace(' ', '')
        d2_cols[i] = d2_cols[i].lower()

    inst_1 = []
    inst_2 = []
    for i in d_1.dtypes:
        inst_1.append(str(i))
    for i in d_2.dtypes:
        inst_2.append(str(i))


    labels_1 = []
    for column in d_1:
        columnSeriesObj = d_1[column]
        vals = ""
        for i in range(len(columnSeriesObj.values)):
            vals= vals +str(columnSeriesObj.values[i])+" "
        ner = Text(vals,hint_language_code='en').entities
        labels_col=[]
        if len(ner)>0:
            for i in range(len(ner)):
                if "LOC" in str(ner[0]):
                    labels_col.append("LOC")
                elif "PER" in str(ner[0]):
                    labels_col.append("PER")
                elif "ORG" in str(ner[0]):
                    labels_col.append("ORG")
                else:
                    labels_col.append("O")
        else:
            labels_col.append("O")
        labels_1.append(max(labels_col, key=labels_col.count))



    labels_2 = []
    for column in d_2:
        columnSeriesObj = d_2[column]
        vals = ""
        for i in range(len(columnSeriesObj.values)):
            vals = vals + str(columnSeriesObj.values[i]) + " "
        ner = Text(vals, hint_language_code='en').entities
        labels_col = []
        if len(ner) > 0:
            for i in range(len(ner)):
                if "LOC" in str(ner[0]):
                    labels_col.append("LOC")
                elif "PER" in str(ner[0]):
                    labels_col.append("PER")
                elif "ORG" in str(ner[0]):
                    labels_col.append("ORG")
                else:
                    labels_col.append("O")
        else:
            labels_col.append("O")
        labels_2.append(max(labels_col, key=labels_col.count))


    s = SLM1(d1_cols, d2_cols, inst_1, inst_2,labels_1,labels_2)
    field_1 = exact["field1"].tolist()
    field_2 = exact["field2"].tolist()

    for i in range(len(field_1)):
        field_1[i].replace(' ', '')
        field_1[i] = field_1[i].lower()
    for i in range(len(field_2)):
        field_2[i].replace(' ', '')
        field_2[i] = field_2[i].lower()

    exact_cor = []
    for i in range(len(field_1)):
        exact_cor.append(field_1[i] + "," + field_2[i])

    precision,recall,f1,f50 = Evaluate(s, exact_cor)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F 1 = ", f1)
    print("F 50 = ", f50)
    return precision,recall,f1,f50


######### USE ##########
Match("all_seasons.csv","NBA_Players.csv","correspondence.csv")
