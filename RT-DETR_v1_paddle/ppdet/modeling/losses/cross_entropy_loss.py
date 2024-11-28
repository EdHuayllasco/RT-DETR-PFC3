import paddle

# Función de pérdida para clasificación
def classification_loss_fn(logits, labels):
    """
    Calcula la pérdida de clasificación usando Cross Entropy Loss.

    Args:
        logits (Tensor): Predicciones del modelo, forma (batch_size, num_classes).
        labels (Tensor): Etiquetas verdaderas, forma (batch_size,), con índices de clase.

    Returns:
        Tensor: Pérdida promedio de clasificación.
    """
    # Crear la función de pérdida
    criterion = paddle.nn.CrossEntropyLoss()
    
    # Calcular la pérdida
    loss = criterion(logits, labels)
    return loss

def classification_loss_fn_with_weights(input, logits, labels, class_weights):
    """
    Calcula la pérdida de clasificación con pesos para clases desbalanceadas.

    Args:
        logits (Tensor): Predicciones del modelo, forma (batch_size, num_classes).
        labels (Tensor): Etiquetas verdaderas, forma (batch_size,).
        class_weights (Tensor): Pesos para cada clase, forma (num_classes,).

    Returns:
        Tensor: Pérdida promedio de clasificación.
    """
    criterion = paddle.nn.CrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, labels)
    return loss

# Pesos para las clases
class_weights = paddle.to_tensor(input, dtype='float32')  # Peso para cada clase

# Calcular pérdida con pesos
loss = classification_loss_fn_with_weights(logits, labels, class_weights)
print(f"Pérdida con Pesos: {loss.numpy()}")
