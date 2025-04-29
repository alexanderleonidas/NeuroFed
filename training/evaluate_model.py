import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(trainable, loss_fn, test_loader, validation=False, save_results=False):
    """
    Evaluates a trained model on the given test dataset and computes various performance
    metrics including loss, accuracy, precision, recall, F1 score, and the confusion matrix.
    The results can be optionally displayed and saved if specified.

    :param trainable: The trainable object containing the model and configuration settings.
    :type trainable: Trainable.
    :param loss_fn: The loss function used to compute the test loss.
    :param test_loader: A DataLoader object representing the test dataset.
    :param validation: A boolean flag indicating whether the evaluation is for validation
        data. Defaults to False.
    :param save_results: A boolean flag indicating if the test results should be saved.
        Defaults to False.
    :return: A tuple containing the following:
        - test_loss: The computed test loss.
        - accuracy: The percentage accuracy of the model on the test dataset.
        - precision: The weighted precision score of the model's predictions.
        - recall: The weighted recall score of the model's predictions.
        - f1: The weighted F1 score of the model's predictions.
        - conf_matrix: The confusion matrix for the model's predictions.
    """
    trainable.model.eval()
    size = len(test_loader.dataset)
    test_loss, correct = 0, 0
    correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(trainable.device), target.to(trainable.device)
            pred = trainable.model.forward(data)
            test_loss += loss_fn(pred, target).item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == target).sum().item()
            all_labels.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    test_loss /= len(test_loader)
    accuracy = 100. * correct / size
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    if trainable.config.VERBOSE:
        if not validation:
            print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')
            print('Confusion Matrix:')
            print(conf_matrix)
            print("-" * 20)
        else:
            print(f'Validation Loss: {test_loss:.4f} | Validation Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')

    # if save_results:
    #     save_test_results(model.name, loss.item(), accuracy, precision, recall, f1, conf_matrix)
    #     if BaseConfig.VERBOSE: print(f'Test results saved to {PERFORMANCE_RESULTS_PATH}{model.name}_test_results.csv')

    return test_loss, accuracy, precision, recall, f1, conf_matrix