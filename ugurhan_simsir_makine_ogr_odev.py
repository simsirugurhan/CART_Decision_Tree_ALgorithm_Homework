#Uğurhan Şimşir
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Eğitim setini okuma
train_data = pd.read_csv("trainSet.csv")

# Test setini okuma
test_data = pd.read_csv("testSet.csv")

# Kategorik değişkenleri dönüştürmek için bir LabelEncoder nesnesi oluşturun
le = LabelEncoder()

# Eğitim ve test setindeki kategorik değişkenleri dönüştürün
for col in ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])    

# Eğitim setindeki özellik değişkenlerini ve hedef değişkeni ayırma ve
X_train = train_data.drop("class", axis=1)
y_train = train_data["class"]

# Test setindeki özellik değişkenlerini ve hedef değişkeni ayırma ve
X_test = test_data.drop("class", axis=1)
y_test = test_data["class"]


def gini_impurity(labels):
    # Sınıf etiketlerinin sıklıklarını sayma
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Gini impurity hesaplama
    impurity = 1
    for label in class_counts:
        probability = class_counts[label] / len(labels)
        impurity -= probability ** 2

    return impurity

#Uğurhan Şimşir
def calculate_gini_index(feature, labels):
    # Özellik değerlerini sıralama
    values = feature.unique()
    values.sort()

    # Olası bölünme noktalarını hesaplama
    split_points = []
    for i in range(len(values) - 1):
        split_points.append((values[i] + values[i+1]) / 2)

    # En iyi bölünme noktasını ve Gini impurity'sini hesaplama
    best_split_point = None
    best_gini_index = float('inf')
    for split_point in split_points:
        left_labels = labels[feature < split_point]
        right_labels = labels[feature >= split_point]
        gini_index = (len(left_labels) / len(labels)) * gini_impurity(left_labels) + (len(right_labels) / len(labels)) * gini_impurity(right_labels)
        if gini_index < best_gini_index:
            best_gini_index = gini_index
            best_split_point = split_point

    return best_split_point, best_gini_index

#decision tree
def build_decision_tree(X, y):
    # Ağaç yapısını temsil eden bir sözlük oluşturma
    tree = {}

    # Veri kümesinin Gini impurity değerini hesaplama
    impurity = gini_impurity(y)

    # Veri kümesindeki sınıf etiketleri aynıysa
    # yaprak düğümü oluşturma ve sınıf etiketi atama
    if impurity == 0:
        tree['class'] = y.iloc[0]
        return tree

    # En iyi özelliği ve bölünme noktasını bulma
    best_feature = None
    best_split_point = None
    best_gini_index = float('inf')
    for feature in X.columns:
        split_point, gini_index = calculate_gini_index(X[feature], y)
        if gini_index < best_gini_index:
            best_feature = feature
            best_split_point = split_point
            best_gini_index = gini_index

    # En iyi özellik ve bölünme noktası ile yeni düğüm oluşturma
    tree['feature'] = best_feature
    tree['split_point'] = best_split_point

    # Veriyi bölme
    left_indices = X[best_feature] < best_split_point
    right_indices = X[best_feature] >= best_split_point
    left_X, left_y = X.loc[left_indices], y.loc[left_indices]
    right_X, right_y = X.loc[right_indices], y.loc[right_indices]

    # Sol ve sağ dalları oluşturma
    tree['left'] = build_decision_tree(left_X, left_y)
    tree['right'] = build_decision_tree(right_X, right_y)

    return tree

# Karar ağacını eğitim veri kümesi ile oluşturma
decision_tree = build_decision_tree(X_train, y_train)

def predict(tree, sample):
    # Yaprak düğüme gelene kadar ağacı dolaşma
    while 'feature' in tree:
        if sample[tree['feature']] < tree['split_point']:
            tree = tree['left']
        else:
            tree = tree['right']
    # Tahmin sınıfı yaprak düğümündeki sınıf etiketi olacaktır
    return tree['class']

# Eğitim veri kümesi üzerinde tahminler yapma
train_predictions = []
for i in range(len(X_train)):
    train_predictions.append(predict(decision_tree, X_train.iloc[i]))
    
# Test veri kümesi üzerinde tahminler yapma
test_predictions = []
for i in range(len(X_test)):
    test_predictions.append(predict(decision_tree, X_test.iloc[i]))

#evaluate
def evaluate(y_true, y_pred):
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 'bad' and y_true[i] == 'bad':
            tn += 1
        elif y_pred[i] == 'bad' and y_true[i] == 'good':
            fp += 1
        elif y_pred[i] == 'good' and y_true[i] == 'bad':
            fn += 1
        elif y_pred[i] == 'good' and y_true[i] == 'good':
            tp += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tnrate = tn / (tn + fp)
    tprate = tp / (tp + fn)

    print("Accuracy: ", accuracy)
    print("TPrate: ", tprate)
    print("TNrate: ", tnrate)
    print("TP adedi: ", tp)
    print("TN adedi: ", tn)

    return accuracy, tprate, tnrate, tp, tn

#Uğurhan Şimşir
#txt ve console yazdırma
f = open("sonuclar_ugurhan_simsir.txt", "a")
f.write("Egitim (Train) sonucu: ")
f.write("\n")
#train
print("Eğitim (Train) sonucu: ")
result_train = evaluate(train_predictions, y_train)
print("----------")
text_train0 = "Accuracy: " + str(result_train[0])
text_train1 = "TPrate: " + str(result_train[1])
text_train2 = "TNrate: " + str(result_train[2])
text_train3 = "TP adedi: " + str(result_train[3])
text_train4 = "TN adedi: " + str(result_train[4])
f.write(text_train0)
f.write("\n")
f.write(text_train1)
f.write("\n")
f.write(text_train2)
f.write("\n")
f.write(text_train3)
f.write("\n")
f.write(text_train4)
f.write("\n")
f.write("----------")
f.write("\n")
f.write("Sinama (Test) sonucu:")
f.write("\n")
print("Sınama (Test) sonucu:")
result_test = evaluate(test_predictions, y_test)
print("----------")
text_test0 = "Accuracy: " + str(result_test[0])
text_test1 = "TPrate: " + str(result_test[1])
text_test2 = "TNrate: " + str(result_test[2])
text_test3 = "TP adedi: " + str(result_test[3])
text_test4 = "TN adedi: " + str(result_test[4])
f.write(text_test0)
f.write("\n")
f.write(text_test1)
f.write("\n")
f.write(text_test2)
f.write("\n")
f.write(text_test3)
f.write("\n")
f.write(text_test4)
f.close()
#karar ağacı modelini console yazdırma 
print("Karar Ağacı Modelim:")
print(decision_tree)

#karar ağacı görselleştirme için
# Karar ağacı modeli oluşturma
clf = DecisionTreeClassifier()

# Modeli eğitme
clf.fit(X_train, y_train)

# Ağaç modelini görselleştirme
dot_data = export_graphviz(clf, filled=True, rounded=True, feature_names=X_train.columns, class_names=['bad', 'good'])
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("decision_tree_ugurhan_simsir")
#Uğurhan Şimşir