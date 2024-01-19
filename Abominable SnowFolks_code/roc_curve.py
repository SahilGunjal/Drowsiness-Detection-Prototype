"""
This code plots ROC curve to select best possible threshold for Yawn Detection and Drowsiness Detection.
Authors: Siddhesh Abhijeet Dhonde (sd1386), Sahil Sanjay Gunjal (sg2736), Atharva Manoj Chiplunkar (ac2434),
        Vaibhav Sharma (vs1654)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data1():
    """
    Reads data for Drowsiness detection.
    :return: None
    """
    df = pd.read_csv('drowsiness_data.csv')
    return df


def read_data2():
    """
    Reads data for Yawn Detection.
    :return: None
    """
    df = pd.read_csv('yawn_data.csv')
    return df


def yawn_threshold(df):
    """
    This code calculates True Positive rates and False Positive Rates for Mouth Yawn Threshold
    :param df:
    :return:
    """
    average_mar_values = df['Average_MAR']
    best_threshold = 0
    min_val = min(average_mar_values)
    max_val = max(average_mar_values)
    gap = 0.025
    min_mistakes = float('inf')
    MAR_range = np.arange(min_val, max_val + gap, gap)
    total_fpr = []
    total_tpr = []
    # Traversing the range of possible threshold
    for threshold in MAR_range:
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0
        # Calculating FPR and TPR for each Threshold to select best Threshold
        for index, val in enumerate(average_mar_values):
            if val > threshold and df['Yawn'].iloc[index] == 0:
                false_negative += 1

            elif val <= threshold and df['Yawn'].iloc[index] == 1:
                false_positive += 1

            elif val < threshold and df['Yawn'].iloc[index] == 0:
                true_positive += 1

            elif val >= threshold and df['Yawn'].iloc[index] == 1:
                true_negative += 1

        total_tpr.append(true_positive / (true_positive + false_negative))
        total_fpr.append(false_positive / (false_positive + true_negative))

        curr_total_mistakes = false_negative + false_positive

        if curr_total_mistakes < min_mistakes:
            min_mistakes = curr_total_mistakes
            best_threshold = threshold

    return best_threshold, total_tpr, total_fpr


def eye_threshold(df):
    """
    This code calculates True Positive rates and False Positive Rates for Eye Threshold
    :param df:
    :return:
    """
    average_ear_values = df['Average_EAR']
    best_threshold = 0
    min_val = min(average_ear_values)
    max_val = max(average_ear_values)
    gap = 0.025
    min_mistakes = float('inf')
    EAR_range = np.arange(min_val, max_val + gap, gap)
    total_fpr = []
    total_tpr = []
    # Traversing the range of possible threshold
    for threshold in EAR_range:
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0
        # Calculating FPR and TPR for each Threshold to select best Threshold
        for index, val in enumerate(average_ear_values):
            if val < threshold and df['Drowsiness'].iloc[index] == 0:
                false_negative += 1

            elif val >= threshold and df['Drowsiness'].iloc[index] == 1:
                false_positive += 1

            elif val > threshold and df['Drowsiness'].iloc[index] == 0:
                true_positive += 1

            elif val <= threshold and df['Drowsiness'].iloc[index] == 1:
                true_negative += 1

        total_tpr.append(true_positive / (true_positive + false_negative))
        total_fpr.append(false_positive / (false_positive + true_negative))

        curr_total_mistakes = false_negative + false_positive

        if curr_total_mistakes < min_mistakes:
            min_mistakes = curr_total_mistakes
            best_threshold = threshold

    return best_threshold, total_tpr, total_fpr


def main():
    df1 = read_data1()
    df2 = read_data2()

    # Get predicted labels using the one-rule threshold approach
    best_threshold_eye, total_tpr_eye, total_fpr_eye = eye_threshold(df1)
    best_threshold_yawn, total_tpr_yawn, total_fpr_yawn = yawn_threshold(df2)

    print(f'Eye Threshold:{best_threshold_eye}')
    print(f'Mouth Threshold:{best_threshold_yawn}')

    # Plot ROC curve
    plt.figure()
    plt.plot(total_fpr_eye, total_tpr_eye, color='blue', marker='o', label='ROC curve on avg. EAR')
    plt.plot(total_fpr_yawn, total_tpr_yawn, color='red', marker='o', label='ROC curve on avg. MAR')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison between ROC Curve on average EAR and average MAR')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
