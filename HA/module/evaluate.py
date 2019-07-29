import numpy as np
import statistics as s

THRESHHOLD = 0.5


def load_dev_labels(data_path='data/dev.txt'):
    # data_path = 'data/train.txt'
    CONV_PAD_LEN = 3
    target_list = []
    f_data = open(data_path, 'r', encoding='utf8')
    data_lines = f_data.readlines()
    f_data.close()

    for i, text in enumerate(data_lines):
        # Ignore the first line as it is the name of the columns
        if i == 0:
            continue
        tokens = text.split('\t')
        emo = tokens[CONV_PAD_LEN + 1].strip()
        target_list.append(EMOS_DIC[emo])
    return np.asarray(target_list)


def to_categorical(vec):
    to_ret = np.zeros((vec.shape[0], NUM_EMO))
    for idx, val in enumerate(vec):
        to_ret[idx, val] = 1
    return to_ret


def get_metrics(ground, predictions):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref -
        https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
#     ground = np.array([1 if x == 1 else 0 for x in ground ])
    ground = np.asarray(ground)
    discretePredictions = np.array([1 if x >= THRESHHOLD else 0 for x in predictions])
    truePositives = np.sum([1 for (x,y) in zip(discretePredictions, ground) if x * y == 1])
    print(discretePredictions.shape)
    print(ground.shape)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    #  Macro level calculation

    precision = truePositives / (truePositives + falsePositives)
    recall = truePositives/ (truePositives + falseNegatives)
    accuracy = np.mean(discretePredictions == ground)
    f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
    print("Accuracy : %.4f, Precision : %.3f, Recall : %.3f, F1 : %.3f" % (accuracy,  precision, recall, f1))


    return accuracy, precision, recall, f1