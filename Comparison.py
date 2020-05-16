import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from SplitData import Spliter
from LearnEmbeddingSpace import Embedding
from RejectClassifierWithLocalAdjustment import ClassifierWithLocalAdjustment
from RejectClassifier import Classifier
#%%
def main(prefixFileName, data_seed, predictive_alg, embedded_option, control_neighbor,
               shrink_parameter, threshold_rejection, proportion_unknown, left_out_proportion, filter_proportion):
    print("Run: ", "\n prefixFileName = ", str(prefixFileName), "\n data_seed = ", str(data_seed), 
          "\n predictive_alg = ", predictive_alg, "\n embedded_option = ", str(embedded_option),  
          "\n control_neighbor = ", str(control_neighbor), "\n shrink_parameter = ", str(shrink_parameter),
          "\n threshold_rejection = ", threshold_rejection, "\n proportion_unknown = ", proportion_unknown,
          "\n left_out_proportion = ", left_out_proportion, "\n filter_proportion = ", filter_proportion) 
    fileName = "Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
    df = pd.read_csv(fileName)
    Xy= df.values
    X= Xy[:,1:]
    y= Xy[:,0].astype(int)
    spl =  Spliter(proportion_unknown = proportion_unknown, left_out_proportion = left_out_proportion, random_seed = data_seed)
    train_indices, test_indices, unknown_classes = spl.Split(X, y)
# embedding features 

    ids = [i for i in range(X.shape[0])]
    emb = Embedding(embedded_option = embedded_option, control_neighbor = control_neighbor)
    X_embedded = emb.converter(X, y, train_indices)
    clf = ClassifierWithLocalAdjustment(predictive_alg = predictive_alg, control_neighbor = control_neighbor,
                                    filter_proportion = filter_proportion, threshold_rejection = threshold_rejection)
    y_predict_srnc = clf.predict(X_embedded[train_indices], y[train_indices], X_embedded[test_indices])
    clf = Classifier(predictive_alg = predictive_alg, threshold_rejection = threshold_rejection)
    y_predict_rejection = clf.predict(X_embedded[train_indices], y[train_indices], X_embedded[test_indices])
    predicted_labels_rejection = [y_predict_rejection[test_indices.index(i)] if i in test_indices else -1 for i in ids]
 #Saving results 
    train_1_test_0_ids = [1 if i in train_indices else 0 for i in ids]
    true_labels = [y[i] for i in ids]
    predicted_labels_srnc = [y_predict_srnc[test_indices.index(i)] if i in test_indices else -1 for i in ids]
    known_unknown_test = [0 if y[i] in unknown_classes else 1 for i in test_indices]
    known_1_unknown_0_classes = [known_unknown_test[test_indices.index(i)] if i in test_indices else predicted_labels_srnc[i] for i in ids]
    #probability = True
    df = pd.DataFrame(data= {'ids': ids, 'train_1_test_0_ids': train_1_test_0_ids, 'true_labels':true_labels, 'predicted_labels_srnc': predicted_labels_srnc, 'predicted_labels_rejection': predicted_labels_rejection, 
                             'known_1_unknown_0_classes': known_1_unknown_0_classes})
    df.to_csv("results/"+prefixFileName+"-ARI-dataseed-"+str(data_seed)+"-predictive_alg-"+str(predictive_alg)+"-embedded_option-"+str(embedded_option)+"-shrink_parameter-"+str(shrink_parameter)+"-left_out_proportion-"+str(left_out_proportion)+".csv", index=False)
    print("done!")
#%%
if __name__ == '__main__':
    main()