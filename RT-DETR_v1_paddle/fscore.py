import json
import numpy as np


with open('output/blast/bbox.json') as f:
    d = json.load(f)
    n = len(d)//300
    cont = 0

    # Blight class_id 1
	# Tungro class_id 2
    # Blast class_id 3
    
    class_id = 3 - 1

    
    TP = 0 # Verdadero positivo, coincide
    FP = 0 # Falso Positivo, se detecto un objeto de mas que no esta presente en la verdad basica 
    FN = 0 # Falso negativo, no se detecto un objeto que debia ser detectado

    TP_acc = 0
    FP_acc = 0
    FN_acc = 0

    umbral = 0.10
    
    #print(d[0]['bbox'])
    for i in range(0, n):
        detection_pr = [[0],[0],[0]]
        class_arr = [0,0,0]
        ind = 0 

        for j in range(0, 300):
            if d[i*300+j]['score'] > umbral:
                detection_pr[d[i*300+j]['category_id']-1].append(d[i*300+j]['score'])

                cont += 1

        for j in range(0, 3):
            class_arr[j] = sum(detection_pr[j])

        max_element = class_arr[0]
        for j in range (1,len(class_arr)): #iterate over array
            if class_arr[j] > max_element: #to check max value
                max_element = class_arr[j]
                ind = j

        if ind == class_id and class_arr[ind] != 0:
            TP += 1
        else:
            FN += 1

        TP_acc += TP
        FP_acc += FP
        FN_acc += FN

        TP = 0
        FP = 0
        FN = 0

        detection_gt = []
        detection_pr = []

    #print(cont)
    
    print("Verdadero positivo ", TP_acc)
    #print("Falso positivo ", FP_acc)
    #print("Verdadero negativo ", TP_acc - FP_acc)
    print("Falso negativo ", FN_acc)

    precision = TP_acc/(TP_acc+FP_acc)
    recall = TP_acc/(TP_acc+FN_acc)
    f1_score = 2 * (precision * recall) / (precision + recall)
    #print("PRECISION: ",precision)
    print("RECALL: ",recall)
    #print("F1-SCORE: ",f1_score)