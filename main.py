# Base Tools
import sys
import numpy as np
from numpy import trapezoid as trapz
import pandas as pd
import matplotlib.pyplot as plt

# Feature Engineering
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

# Prediction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Validation & Metrics
from sklearn.metrics import roc_curve,auc

"-----------------------------------------------------------------------------------------------------------------------------------"

mode = "Double"

"-----------------------------------------------------------------------------------------------------------------------------------"

sheet_param = "Raw Data"
slice_param = "Full Interval"
normalization_param = None
smoothing_param = "On"
split_param = "LOOCV"
leave_one_out_reference = "A1"
cross_val_size = 13

"-----------------------------------------------------------------------------------------------------------------------------------"

data = pd.DataFrame(pd.read_excel("Data/Data.xlsx", sheet_param))
total_features = len(list(data[data.columns[0]]))

"-----------------------------------------------------------------------------------------------------------------------------------"

id0001 = list(data["Id"])
status0001 = list(data["Status"])

data = data.drop(columns=["Id","Status"])

new_lengths = []
old_lengths = list(data.columns)

for i in old_lengths:
    new_data = int(i)
    new_lengths.append(new_data)

data.columns = new_lengths
if slice_param == "Full Interval":
    data = data.loc[:, 600:3801]
    new_lengths = list(data.columns)
elif slice_param  == "Fingerprint Region":
    data = data.loc[:, 800:1701]
    new_lengths = list(data.columns)
elif slice_param  == "High-frequency Region":
    data = data.loc[:, 2600:3001]
    new_lengths = list(data.columns)
elif slice_param  == "Double Interval":
    df1 = data.loc[:, 2600:3001]
    df2 = data.loc[:, 800:1701]
    data = pd.concat([df2,df1], axis = 1)
    new_lengths = list(data.columns)

wavelengths = list(data.columns)
total_features_of_wavelengths = len(wavelengths)
data = data.copy()
data["Id"] = list(id0001)

"-----------------------------------------------------------------------------------------------------------------------------------"

area, rreeff, max_y_detected = [], [], []

for i in id0001:
    crack = pd.DataFrame(data.loc[data["Id"] == i]).values.reshape(total_features_of_wavelengths+1,1).tolist()
    
    y = []
    rreeff.append(crack[-1][0])
    crack.pop(-1)

    for q in crack:
        y.append(q[0])
    x = wavelengths
    max_y_detected.append(max(y))

    a = round(trapz(y, dx=max(x)),3)
    area.append(a)

"-----------------------------------------------------------------------------------------------------------------------------------"

data = data.drop(columns=["Id"])

if smoothing_param == "On":
    smoothing = savgol_filter(data, window_length=15, polyorder=3)
    data = pd.DataFrame(smoothing)

data.columns = new_lengths

"-----------------------------------------------------------------------------------------------------------------------------------"

if normalization_param == "Area Normalization":
    for AAA,NNN in zip(list(data.index),area):
        data.iloc[AAA] = data.iloc[AAA]/NNN

"-----------------------------------------------------------------------------------------------------------------------------------"

# # # LOOCV Loop Starts Here Below

"-----------------------------------------------------------------------------------------------------------------------------------"

PINKM4N = id0001
RBFX, POLYX, LINEARX, MINKOWSKIX, EUCLIDEANX, COSINEX, HEISENBERG = [], [], [], [], [], [], pd.DataFrame()
TL31, SL31 = [], []

HEISENBERG["Reference"] = id0001
HEISENBERG["Label"] = status0001

for ABQ in PINKM4N:
    data["Reference"] = id0001
    data["Y"] = status0001

    y_test, y_train = [], []
    x_test, x_train = [], []

    # Selecting the one & filling the y_test.
    YY = data.loc[data["Reference"] == ABQ]
    YY = YY.drop(columns=["Reference"])
    y_test.append(list(YY["Y"])[0])

    # Leaving-one-out & filling the y_train.
    ref_index = list(data["Reference"]).index(ABQ)
    XX = data.drop(ref_index)
    XX = XX.drop(columns=["Reference"])
    y_train = list(XX["Y"])

    # List type to numpy array type conversion.
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Extracting Y variable to convert dataframes into X.
    XX = XX.drop(columns=["Y"])
    YY = YY.drop(columns=["Y"])

    # Filling the x_train & x_test.
    x_train = XX.values
    x_test = YY.values

    # Let"s define the big X and the big Y in terms of the full feature matrix.
    temporary_defined_variable = data
    Y = list(temporary_defined_variable["Y"])
    
    temporary_defined_variable = temporary_defined_variable.drop(columns=["Reference","Y"])
    X = temporary_defined_variable.values
    Y = np.array(Y)

    "-----------------------------------------------------------------------------------------------------------------------------------"

    SCALER = StandardScaler()
    if normalization_param == "Vector Normalization":
        x_train = SCALER.fit_transform(x_train)
        x_test = SCALER.transform(x_test)
    else:
        pass
    
    "-----------------------------------------------------------------------------------------------------------------------------------"

    y_train_new, y_test_new, Y_NEW = [], [], []

    for i in list(y_train):
        if i == "H":
            y_train_new.append(0)
        elif i == "A":
            y_train_new.append(1)

    for i in list(y_test):
        if i == "H":
            y_test_new.append(0)
        elif i == "A":
            y_test_new.append(1)

    for i in list(Y):
        if i == "H":
            Y_NEW.append(0)
        elif i == "A":
            Y_NEW.append(1)

    y_train = np.array(y_train_new)
    y_test = np.array(y_test_new)
    Y = np.array(Y_NEW)

    # Statistical variable definition.
    total_train_features = len(list(x_train))
    total_test_features = len(list(x_test))
    total_matrix_features = total_train_features + total_test_features

    "-----------------------------------------------------------------------------------------------------------------------------------"
    
    # RBF-gamma, RBF-tol, RBF-c.
    # POLY-gamma, POLY-tol, POLY-c.
    # Linear-gamma, Linear-tol, Linear-c.
    # Minkowski-n, Minkwoski-weight.
    # Euclidean-n, Euclidean-weight.
    # Cosine-n, Cosine-weight.

    "-----------------------------------------------------------------------------------------------------------------------------------"

    reg_params = {
        "Full Interval": {
            "Raw Data": { 
                None:[
                1e-01,1e-07,0.36,
                1e-02,1e-07,1.56,
                1e-05,1.000,1.11,
                2,"uniform",
                2,"uniform",
                4,"distance"],
                "Vector Normalization":[
                1e-01,1e-07,0.36,
                1e-02,1e-07,1.56,
                1e-05,1.000,1.11,
                2,"uniform",
                2,"uniform",
                4,"distance"],
                "Area Normalization":[
                1e-04,1e-07,1.16,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.01,
                4,"uniform",
                4,"uniform",
                4,"distance"]
            },
            "Second Derivative": {
                None:[
                1e-02,1e-07,0.01,
                1e-01,1e-07,0.11,
                1e-05,1e-07,0.01,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
                "Vector Normalization":[
                1e-02,1e-07,0.01,
                1e-01,1e-07,0.11,
                1e-05,1e-07,0.01,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
                "Area Normalization":[
                1e-01,1e-07,0.01,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.01,
                4,"uniform",
                4,"uniform",
                4,"uniform"]
            }
        },
        "Fingerprint Region": {
            "Raw Data": { 
                None:[
                1e-01,1e-07,5.46,
                1e-02,1e-07,0.26,
                1e-05,1.000,0.26,
                3,"uniform",
                3,"uniform",
                2,"uniform"],
                "Vector Normalization":[
                1e-01,1e-07,5.46,
                1e-02,1e-07,0.26,
                1e-05,1.000,0.26,
                3,"uniform",
                3,"uniform",
                2,"uniform"],
                "Area Normalization":[
                1e-03,1e-07,1.91,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.01,
                2,"uniform",
                2,"uniform",
                2,"uniform"],
            },
            "Second Derivative": {
                None:[
                1e-03,1e-07,0.01,
                1e-05,1e-07,0.10,
                1e-05,1e-07,0.10,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
                "Vector Normalization":[
                1e-03,1e-07,0.01,
                1e-05,1e-07,0.10,
                1e-05,1e-07,0.10,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
                "Area Normalization":[
                1e-02,1e-07,0.01,
                1e-01,1e-07,0.51,
                1e-05,1e-07,0.01,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
            }
        },
        "High-frequency Region": {
            "Raw Data": { 
                None:[
                0.500,1e-07,2.30,
                0.500,1e-01,0.01,
                1e-05,1e-07,0.31,
                2,"uniform",
                2,"uniform",
                4,"distance"],
                "Vector Normalization":[
                0.500,1e-07,0.01,
                0.500,1e-01,0.01,
                1e-05,1e-07,0.31,
                2,"uniform",
                2,"uniform",
                4,"distance"],
                "Area Normalization":[
                1e-03,1e-07,1.66,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.01,
                4,"distance",
                4,"distance",
                4,"distance"]
            },
            "Second Derivative": {
                None:[
                1.000,1e-07,0.01,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.10,
                2,"uniform",
                2,"uniform",
                3,"uniform"],
                "Vector Normalization":[
                1.000,1e-07,0.01,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.10,
                2,"uniform",
                2,"uniform",
                3,"uniform"],
                "Area Normalization":[
                1e-05,1e-07,0.01,
                1e-02,1e-07,0.01,
                1e-05,1e-07,0.01,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
            }
        },
        "Double Interval": {
            "Raw Data": { 
                None:[
                1e-01,1e-07,9.21,
                1e-02,1e-07,3.31,
                1e-05,1e-07,0.11,
                3,"uniform",
                3,"uniform",
                3,"distance"],
                "Vector Normalization":[
                1e-01,1e-07,9.21,
                1e-02,1e-07,3.31,
                1e-05,1e-07,0.11,
                3,"uniform",
                3,"uniform",
                3,"distance"],
                "Area Normalization":[
                1e-04,1e-07,6.76,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.01,
                3,"distance",
                3,"distance",
                3,"distance"]
            },
            "Second Derivative": {
                None:[
                1e-03,1e-07,0.01,
                1e-05,1e-07,0.10,
                1e-05,1e-07,0.10,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
                "Vector Normalization":[
                1e-03,1e-07,0.01,
                1e-05,1e-07,0.10,
                1e-05,1e-07,0.10,
                4,"uniform",
                4,"uniform",
                4,"uniform"],
                "Area Normalization":[
                1e-01,1e-07,0.01,
                1e-05,1e-07,0.01,
                1e-05,1e-07,0.01,
                2,"uniform",
                2,"uniform",
                2,"uniform"],
            }
        }
    }

    "-----------------------------------------------------------------------------------------------------------------------------------"

    # SVM
    SVC_RBF = SVC(kernel="rbf",gamma=reg_params[slice_param][sheet_param][normalization_param][0],tol=reg_params[slice_param][sheet_param][normalization_param][1],C=reg_params[slice_param][sheet_param][normalization_param][2])
    SVC_POLY = SVC(kernel="poly",gamma=reg_params[slice_param][sheet_param][normalization_param][3],tol=reg_params[slice_param][sheet_param][normalization_param][4],C=reg_params[slice_param][sheet_param][normalization_param][5])
    SVC_LINEAR = SVC(kernel="linear",gamma=reg_params[slice_param][sheet_param][normalization_param][6],tol=reg_params[slice_param][sheet_param][normalization_param][7],C=reg_params[slice_param][sheet_param][normalization_param][8])

    # kNN
    KNN_MINKOWSKI = KNeighborsClassifier(algorithm="brute",metric="minkowski",n_neighbors=reg_params[slice_param][sheet_param][normalization_param][9],weights=reg_params[slice_param][sheet_param][normalization_param][10])
    KNN_EUCLIDEAN = KNeighborsClassifier(algorithm="brute",metric="euclidean",n_neighbors=reg_params[slice_param][sheet_param][normalization_param][11],weights=reg_params[slice_param][sheet_param][normalization_param][12])
    KNN_COSINE = KNeighborsClassifier(algorithm="brute",metric="cosine",n_neighbors=reg_params[slice_param][sheet_param][normalization_param][13],weights=reg_params[slice_param][sheet_param][normalization_param][14])

    "-----------------------------------------------------------------------------------------------------------------------------------"

    def resume(title, model, pool, truee_list_roc, score_list_roc):

        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)
        
        if y_test[0] == y_predicted[0]:
            pool.append(int(1))
        else:
            pool.append(int(0))

        if title == "RBF":
            truee_list_roc.append(y_test[0])
            score_list_roc.append(model.decision_function(x_test)[0])
        else:
            pass

    resume("RBF",SVC_RBF,RBFX,TL31,SL31)
    resume("POLY",SVC_POLY,POLYX,TL31,SL31)
    resume("Linear",SVC_LINEAR,LINEARX,TL31,SL31)
    resume("Minkowski",KNN_MINKOWSKI,MINKOWSKIX,TL31,SL31)
    resume("Euclidean",KNN_EUCLIDEAN,EUCLIDEANX,TL31,SL31)
    resume("Cosine",KNN_COSINE,COSINEX,TL31,SL31)

HEISENBERG["RBF"] = RBFX
HEISENBERG["POLY"] = POLYX
HEISENBERG["Linear"] = LINEARX
HEISENBERG["Minkowski"] = MINKOWSKIX
HEISENBERG["Euclidean"] = EUCLIDEANX
HEISENBERG["Cosine"] = COSINEX

"-----------------------------------------------------------------------------------------------------------------------------------"

FPR, TPR, thresholds = roc_curve(np.array(TL31), np.array(SL31))
roc_auc = auc(FPR, TPR)

plt.figure(figsize=(16,9),dpi=72)
plt.plot(FPR, TPR, color="darkorange", lw=2, label="ROC Curve (Area = %0.2f)" %roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"SVC-RBF Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")

"-----------------------------------------------------------------------------------------------------------------------------------"

if mode == "Double":
    HEISENBERG.to_excel(f"{normalization_param} - {slice_param} - {sheet_param}.xlsx")
    plt.show()
elif mode == "LOOCV":
    HEISENBERG.to_excel(f"{normalization_param} - {slice_param} - {sheet_param}.xlsx")
elif mode == "Plot":
    plt.show()
else:
    print("Please select a valid parameter setting for 'mode'.")
    sys.exit()