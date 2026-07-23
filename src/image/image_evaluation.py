from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_image_model(model, X_test, y_test, class_names):
    """
    Avalia o modelo de imagem usando relatório de classificação
    e matriz de confusão.
    """

    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Matriz de Confusão - Modelo de Imagem")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.show()

    return cm