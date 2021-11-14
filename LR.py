import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_data(mode: int) -> pd.DataFrame:
    if mode == 1:
        a = []
        name = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE",
                "CP", "SE"]
        for i in name:
            ps = np.loadtxt(f"data/{i}.txt")
            ps = ps.mean(axis=1)
            a.append(ps)

        df = pd.DataFrame({x: y for x, y in zip(name, a)})

        target = np.loadtxt("data/profile.txt")
        df_temp = pd.DataFrame(target, columns=["Cooler_Condition", "Valve_Condition", "Internal_Pump_Leakage",
                                                "Hydraulic_Accumulator", "Stable_Flag"])

        df_final = pd.concat([df, df_temp], axis=1)
        with open("data_parsed/LR", "wb") as f:
            pickle.dump(df_final, f)
        return df_final
    else:
        with open("data_parsed/LR", "rb") as f:
            df_final = pickle.load(f)
        return df_final


def linear_reg(x, target):
    train_x, test_x, train_y, test_y = train_test_split(x, target, test_size=0.2, random_state=1)

    sc = StandardScaler()

    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    l_classifier = LogisticRegression(random_state=42, max_iter=1000)
    l_classifier.fit(train_x, train_y)

    pred_train_y = l_classifier.predict(train_x)
    pred_test_y = l_classifier.predict(test_x)
    accuracy_score_lr_train = accuracy_score(train_y, pred_train_y)
    accuracy_score_lr_test = accuracy_score(test_y, pred_test_y)
    print("accuracy_score_lr_train= ", accuracy_score_lr_train)
    print("accuracy_score_lr_test= ", accuracy_score_lr_test)
    sns.heatmap(confusion_matrix(test_y, pred_test_y), annot=True)

    precision_score_lr_train = precision_score(train_y, pred_train_y, average="weighted")
    precision_score_lr_test = precision_score(test_y, pred_test_y, average="weighted")
    print("precision_score_lr_train= ", precision_score_lr_train)
    print("precision_score_lr_test= ", precision_score_lr_test)

    scores = cross_validate(l_classifier, train_x, train_y, scoring=["accuracy", "precision_weighted"], cv=10)
    accuracy_score_lr_cross = scores["test_accuracy"].mean()
    print("accuracy_score_lr_cross= ", accuracy_score_lr_cross)
    precision_score_lr_cross = scores["test_precision_weighted"].mean()
    print("precision_score_lr_cross= ", precision_score_lr_cross)

    plt.show()


def main():
    df_final = load_data(0)
    x = df_final.iloc[:, :-5]
    y = df_final.iloc[:, -5:]

    target_1 = y.iloc[:, -5]
    target_2 = y.iloc[:, -4]
    target_3 = y.iloc[:, -3]
    target_4 = y.iloc[:, -2]
    final_target = y.iloc[:, -1]
    final_target = (final_target.astype(int))

    linear_reg(x, target_2)
    print()
    linear_reg(x, target_3)


if __name__ == '__main__':
    main()
