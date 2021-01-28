import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics, itertools

import pydicom as dicom
import skimage.transform, scipy.misc

import scipy.stats
import shutil, os

from tf_utils.delong import *  # confidence interval for roc auc


def draw_roc_curve(predictions, labels, target_index, name):
    print('---------- ROC Statistics ----------')
    predictions = np.asarray(predictions)
    labels = np.asarray(labels).astype(np.int32)

    predictions = predictions[:, target_index]  # target probability
    if not np.ndim(labels) == 1:  # if one-hot encoded
        labels = labels[:, target_index]  # target reference

    false_pos_rate, true_pos_rate, thresholds = sklearn.metrics.roc_curve(labels, predictions, drop_intermediate=False)
    roc_auc = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = np.abs(array - value).argmin()
        return idx

    # general threshold = 0.5
    idx = find_nearest_idx(thresholds, value=0.5)
    plt.plot(false_pos_rate[idx], true_pos_rate[idx], 'o')

    # youden's index maximizing both sensitivity and specificity
    optimal_idx = abs(true_pos_rate + false_pos_rate - 1.).argsort()[1]

    print('Threshold={0:0.3f}, True Positive Rate={1:0.2f}, True Negative Rate={2:0.2f}'
          .format(thresholds[idx], true_pos_rate[idx], 1 - false_pos_rate[idx]))
    print('Threshold={0:0.3f}, True Positive Rate={1:0.2f}, True Negative Rate={2:0.2f}\n'
          .format(thresholds[optimal_idx], true_pos_rate[optimal_idx], 1 - false_pos_rate[optimal_idx]))

    plt.plot(false_pos_rate, true_pos_rate, color='green', lw=2, label='ROC curve (area = {:0.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.legend(loc='lower right')

    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity');
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic')

    folder = os.path.split(name)[0]
    if not os.path.exists(folder): os.makedirs(folder)
    plt.savefig(name);
    plt.clf()

    alpha = 0.95
    auc, auc_cov = delong_roc_variance(predictions=predictions, labels=labels)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1

    print('ROC-AUC={0:0.4f}, 95% AUC CI: [{1:0.4f}, {2:0.4f}]'.format(roc_auc, ci[0], ci[1]))
    print('------------------------------------\n')
    return thresholds[optimal_idx]  # optimal cut-off


def draw_confusion_matrix(predictions, labels, label_names=None, name=None):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels).astype(np.int32)

    predictions = np.argmax(predictions, axis=-1)
    if not np.ndim(labels) == 1:  # if one-hot encoded
        labels = np.argmax(labels, axis=-1)

    cnf_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    # normalize matrix to 0.0 ~ 1.0
    cnf_matrix = cnf_matrix.astype(np.float32) / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    plt.colorbar()
    if not label_names == None:
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names);
        plt.yticks(tick_marks, label_names)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], '0.2f' if cnf_matrix.dtype == np.float64 else 'd'), \
                 horizontalalignment='center', color='white' if cnf_matrix[i, j] > thresh else 'black')

    plt.xlabel('Predicted label');
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    folder = os.path.split(name)[0]
    if not os.path.exists(folder): os.makedirs(folder)
    plt.savefig(name);
    plt.clf()


def qualitative_analysis(filenames, predictions, labels, target_index, threshold, preserve_file=False):
    assert len(predictions) == len(labels), 'Predictions and labels have different length'

    predictions = np.asarray(predictions)[:, target_index]  # target probability
    if not np.ndim(labels) == 1:  # if one-hot encoded
        labels = np.asarray(labels)[:, target_index]  # target reference

    true_positive, false_negative = [], []
    true_negative, false_positive = [], []

    for idx in range(len(predictions)):
        if not (predictions[idx] > threshold) == labels[idx]:
            if labels[idx] == 1:
                false_negative.append(idx)
            else:
                false_positive.append(idx)
        else:
            if labels[idx] == 1:
                true_positive.append(idx)
            else:
                true_negative.append(idx)

    true_positive, false_negative = np.asarray(true_positive), np.asarray(false_negative)
    true_negative, false_positive = np.asarray(true_negative), np.asarray(false_positive)

    # check images in practice
    for arr_name, arr in {'true positive': true_positive, 'false negative': false_negative,
                          'true negative': true_negative, 'false positive': false_positive}.items():

        save_path = os.path.join('qualitative_results', arr_name)
        if not os.path.exists(save_path): os.makedirs(save_path)

        for filename in np.asarray(filenames[arr]).flatten()[:20]:
            if preserve_file:
                shutil.copyfile(filename, os.path.join(save_path, os.path.split(filename)[-1]))
                continue

            name, extension = os.path.split(filename)[-1].split('.')
            if extension == 'dcm':
                dicom_info = dicom.read_file(filename)
                image = dicom_info.pixel_array
                if str(dicom_info[0x28, 0x04].value) == 'MONOCHROME1':
                    white_image = np.full_like(image, np.max(image), image.dtype)
                    image = np.subtract(white_image, image)
            else:
                image = scipy.misc.imread(filename)

            image = skimage.transform.resize(image, [512, 512], preserve_range=True)
            scipy.misc.imsave(os.path.join(save_path, name + '.png'), image)