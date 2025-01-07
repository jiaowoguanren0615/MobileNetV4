import torch, json, os
import seaborn as sns
from sklearn.metrics import auc, f1_score, roc_curve, classification_report, confusion_matrix, roc_auc_score
from itertools import cycle
from numpy import interp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Iterable
from optim_AUC import OptimizeAUC
from terminaltables import AsciiTable


@torch.inference_mode()
def Plot_ROC(net: torch.nn.Module, val_loader: Iterable, save_name: str, device: torch.device):
    """
        Plot ROC Curve

        Save the roc curve as an image file in the current directory

        Args:
            net (torch.nn.Module): The model to be evaluated.
            val_loader (Iterable): The data loader for the valid data.
            save_name (str): The file path of your model weights
            device (torch.device): The device used for training (CPU or GPU).

        Returns:
            None
    """

    try:
        json_file = open('./classes_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    score_list = []
    label_list = []

    net.load_state_dict(torch.load(save_name)['model'])

    for i, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(net(images), dim=1)
        score_tmp = outputs
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # convert label to one-hot form
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], len(class_indict.keys()))
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # compute tpr and fpr for each label by using sklearn lib
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(len(class_indict.keys())):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(class_indict.keys()))]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(set(label_list))):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_indict.keys())
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # plot roc curve for each label
    plt.figure(figsize=(12, 12))
    lw = 2

    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(class_indict.keys())), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_indict[str(i)], roc_auc_dict[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Chance', color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./multi_classes_roc.png')
    # plt.show()


@torch.inference_mode()
def predict_single_image(model: torch.nn.Module, device: torch.device, weight_path: str):
    """
        Predict Single Image.

        Save the prediction as an image file which including pred label and prob in the current directory

        Args:
            model (torch.nn.Module): The model to be evaluated.
            device (torch.device): The device used for training (CPU or GPU).
            weight_path (str): The model weights file

        Returns:
            None
    """

    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'valid': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    img_transform = data_transform['valid']

    # load image
    img_path = "rose.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = img_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './classes_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # load model weights

    assert os.path.exists(weight_path), "weight file dose not exist."
    model.load_state_dict(torch.load(weight_path, map_location=device)['model'])

    model.eval()
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.savefig(f'./pred_{img_path}')
    # plt.show()


@torch.inference_mode()
def Predictor(net: torch.nn.Module, test_loader: Iterable, save_name: str, device: torch.device):
    """
        Evaluate the performance of the model on the given dataset.

        1. This function will print the following metrics:
            - F1 score
            - Confusion matrix
            - Classification report

        2. Save the confusion matrix as an image file in the current directory.

        Args:
            net (torch.nn.Module): The model to be evaluated.
            test_loader (Iterable): The data loader for the valid data.
            save_name (str): The file path of your model weights
            device (torch.device): The device used for training (CPU or GPU).

        Returns:
            None
    """

    try:
        json_file = open('./classes_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    errors = 0
    y_pred, y_true = [], []
    net.load_state_dict(torch.load(save_name)['model'])

    net.eval()

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        preds = torch.argmax(torch.softmax(net(images), dim=1), dim=1)
        for i in range(len(preds)):
            y_pred.append(preds[i].cpu())
            y_true.append(labels[i].cpu())

    tests = len(y_pred)
    for i in range(tests):
        pred_index = y_pred[i]
        true_index = y_true[i]
        if pred_index != true_index:
            errors += 1

    acc = (1 - errors / tests) * 100
    print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}%')

    ypred = np.array(y_pred)
    ytrue = np.array(y_true)

    f1score = f1_score(ytrue, ypred, average='weighted') * 100

    print(f'The F1-score was {f1score:.3f}')
    class_count = len(list(class_indict.values()))
    classes = list(class_indict.values())

    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.xticks(np.arange(class_count) + .5, classes, rotation=45, fontsize=14)
    plt.yticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.1%')
    plt.xticks(np.arange(class_count) + .5, classes, rotation=45, fontsize=14)
    plt.yticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.savefig('./confusion_matrix.png')
    # plt.show()

    clr = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n----------------------\n", clr)


@torch.inference_mode()
def OptAUC(net: torch.nn.Module, val_loader: Iterable, save_name: str, device: torch.device):
    """
        Optimize model for improving AUC

        Print a table of initial and optimized AUC and F1-score.

        This function takes the initial and optimized AUC and F1-score, and generates
        an ASCII table to display the results. The table will have the following format:

        Optimize Results
        +----------------------+----------------------+----------------------+----------------------+
        | Initial AUC          | Initial F1-Score     | Optimize AUC         | Optimize F1-Score    |
        +----------------------+----------------------+----------------------+----------------------+
        | 0.654321             | 0.654321             | 0.876543             | 0.876543            |
        +----------------------+----------------------+----------------------+----------------------+

        The optimized AUC and F1-score are obtained by using the `OptimizeAUC` class (in ./optim_AUC.py), which
        performs optimization on the initial metrics.

        Args:
            net (torch.nn.Module): The model to be evaluated.
            test_loader (Iterable): The data loader for the valid data.
            save_name (str): The file path of your model weights
            device (torch.device): The device used for training (CPU or GPU).

        Returns:
            None
    """

    score_list = []
    label_list = []

    net.load_state_dict(torch.load(save_name)['model'])

    for i, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(net(images), dim=1)
        score_tmp = outputs
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.detach().cpu().numpy())

    score_array = np.array(score_list)
    label_list = np.array(label_list)
    y_preds = np.argmax(score_array, axis=1)
    f1score = f1_score(label_list, y_preds, average='weighted') * 100
    auc_score = roc_auc_score(label_list, score_array, average='weighted', multi_class='ovo')

    opt_auc = OptimizeAUC()
    opt_auc.fit(score_array, label_list)
    opt_preds = opt_auc.predict(score_array)
    opt_y_preds = np.argmax(opt_preds, axis=1)
    opt_f1score = f1_score(label_list, opt_y_preds, average='weighted') * 100
    opt_auc_score = roc_auc_score(label_list, opt_preds, average='weighted', multi_class='ovo')

    TITLE = 'Optimize Results'
    TABLE_DATA = (
        ('Initial AUC', 'Initial F1-Score', 'Optimize AUC', 'Optimize F1-Score'),
        ('{:.6f}'.format(auc_score),
         '{:.6f}'.format(f1score),
         '{:.6f}'.format(opt_auc_score),
         '{:.6f}'.format(opt_f1score)
         ),
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)
