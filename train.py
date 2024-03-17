import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist,mnist
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import wandb
wandb.login()
# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# Display sample images
def display_sample_images():
    label = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
    print_Once = [1] * 10
    count = 10
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))

    for i in range(60000):
        if count == 0:
            break
        if print_Once[y_train[i]]:
            print_Once[y_train[i]] -= 1
            count -= 1
            col = 10 - count
            axes[col - 1].imshow(x_train[i], cmap='gray')
            axes[col - 1].set_title("{}".format(label[y_train[i]]))
            axes[col - 1].axis('off')

    plt.tight_layout()
    plt.show()
    wandb.log({"Sample Image":plt})

# Initialize weights and biases
def initialize_weights(initialization_func, prev_layer_neurons, no_of_hidden_layers, classes,each_layer_neuron):
    theta = []
    for i in range(2 * (no_of_hidden_layers + 1)):
        theta.append([])
    for i in range(no_of_hidden_layers):
        neurons_in_layer = each_layer_neuron
        make_weights(theta, neurons_in_layer, prev_layer_neurons, i, initialization_func)
        make_biases(theta, neurons_in_layer, no_of_hidden_layers + 1 + i, initialization_func)
        prev_layer_neurons = neurons_in_layer
    make_weights(theta, classes, prev_layer_neurons, no_of_hidden_layers, initialization_func)
    make_biases(theta, classes, 2 * (no_of_hidden_layers + 1) - 1, initialization_func)
    return theta

# Make weights
def make_weights(theta, curr_layer_neurons, prev_layer_neurons, layer_no, initialization_func):
    if initialization_func == "random":
        theta[layer_no] = np.float64(np.random.randn(curr_layer_neurons, prev_layer_neurons))
    elif initialization_func == "Xavier":
        factor = np.sqrt(6.0 / (curr_layer_neurons + prev_layer_neurons))
        theta[layer_no] = np.float64(np.random.uniform(low=-factor, high=factor, size=(curr_layer_neurons, prev_layer_neurons)))

# Make biases
def make_biases(theta, curr_layer_neurons, layer_no, initialization_func):
    if initialization_func == "random":
        theta[layer_no] = np.float64(np.random.randn(curr_layer_neurons, 1))
    elif initialization_func == "Xavier":
        theta[layer_no] = np.float64(np.zeros((curr_layer_neurons, 1)))

# Calculate activation function
def calc_activation(a, activation_func):
    h = []
    a=np.round(a,6)
    for i in range(len(a)):
        if activation_func == "sigmoid":
          if(a[i][0]<-30):
            h.append(0.0)
          else:
            h.append(1/(1+np.exp(-a[i][0])))

        elif activation_func == "ReLU":
            h.append(max(0, a[i][0]))

        elif activation_func == "tanh":
           if(a[i][0]<-30):
              h.append(-1.0)
           else:
              h.append(2 * (1 / (1 + np.exp(-2 * a[i][0]))) - 1)
        elif activation_func=="identity":
            h.append(a[i][0])
    h = np.array(h)
    h_new = h.reshape((len(h), 1))
    return h_new

# Calculate activation derivative
def calc_activation_derivative(a, activation_func):
    if activation_func == "sigmoid":
        a=0.0 if(a<-30) else 1/(1+np.exp(a))
        return a * (1 - (a))
    elif activation_func == "ReLU":
        return np.where(a > 0, 1, 0)
    elif activation_func == "tanh":
        return 1-(np.tanh(a)**2)
    elif activation_func=="identity":
        return 1

def calc_gdash(ak,activationFunc):
  gdsh=[]
  for i in ak:
      gdsh.append(calc_activation_derivative(i[0],activationFunc))
  gdsh=np.array(gdsh)
  gdshNew=gdsh.reshape((len(gdsh),1))
  return gdshNew

def calc_aL(aL,y,loss_type):
  if(loss_type=="cross_entropy"):
    aL[y][0]=-(1-aL[y][0])
    return aL
  elif(loss_type=="mean_squared_error"):
    Y=np.zeros_like(aL)
    Y[y][0]=1
    return np.multiply(-2*(aL-Y),np.multiply(aL,(1-aL)))

# Calculate softmax
def calc_softmax(a):
    #return np.exp(a) / np.sum(np.exp(a), axis=0)
    exp_a = np.exp(a - np.max(a, axis=0))
    return exp_a / np.sum(exp_a, axis=0)

# Forward propagation
def forward_propagation(theta, inp_list, activation_func,no_of_hidden_layers):
    a_h_list = []
    h = inp_list
    for i in range(no_of_hidden_layers):
        a = np.dot(theta[i], h) + theta[no_of_hidden_layers + 1 + i]
        a_h_list.append(a)
        h = calc_activation(a, activation_func)
        a_h_list.append(h)
    a = np.dot(theta[no_of_hidden_layers], h) + theta[-1]
    a_h_list.append(a)
    y_hat = calc_softmax(a)
    a_h_list.append(y_hat)
    return a_h_list

# Calculate loss
def calc_loss(yhat, actual, loss_type):

    if loss_type == "mean_squared_error":
        sum=0
        for i in range(10):
          if(i==actual):
            sum+=(1-yhat[i][0])**2
          else:
            sum+=yhat[i][0]**2
        return  sum/10

    elif loss_type == "cross_entropy":
        prediction=yhat[actual][0]
        if(not prediction):
          prediction=1e-10
        return -np.log(prediction)

def calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,which_loss,alpha):
      correct = 0
      total = int(60000 * validation_split)
      loss = 0.0
      if(which_loss=="validation"):
        for i in range(59999, 59999 - total - 1, -1):
            a_h_list = forward_propagation(theta, x_train[i].flatten().reshape((784, 1)), activation_func,no_of_hidden_layers)
            prediction = np.argmax(a_h_list[-1])
            if prediction == y_train[i]:
                correct += 1
            loss += calc_loss(a_h_list[-1], y_train[i], loss_type)
        sumW = sum([np.sum(theta[i]**2) for i in range(no_of_hidden_layers+1)])
        regularization_term = (alpha / 2) * sumW
        accuracy = correct / total
        loss = (loss + regularization_term) / total
        return accuracy,loss
      elif(which_loss=="train"):
        for i in range(0, 60000-total):
            a_h_list = forward_propagation(theta, x_train[i].flatten().reshape((784, 1)), activation_func,no_of_hidden_layers)
            prediction = np.argmax(a_h_list[-1])
            if prediction == y_train[i]:
                correct += 1
            loss += calc_loss(a_h_list[-1], y_train[i], loss_type)
        sumW=sum([np.sum(theta[i]**2) for i in range(no_of_hidden_layers+1)])
        regularization_term = (alpha / 2) * sumW
        accuracy = correct / (60000-total)
        loss = (loss + regularization_term) / (60000-total)
        return accuracy,loss
      elif(which_loss=="test"):
        classes=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt", "Sneaker","Bag", "Ankle boot"]
        y_true=[]
        y_pred=[]
        for i in range(10000):
          a_h_list = forward_propagation(theta, x_test[i].flatten().reshape((784, 1)), activation_func,no_of_hidden_layers)
          prediction = np.argmax(a_h_list[-1])
          if prediction == y_test[i]:
              correct += 1
          loss += calc_loss(a_h_list[-1], y_test[i], loss_type)
          y_true.append(classes[y_test[i]])
          y_pred.append(classes[prediction])
        accuracy = correct /10000
        loss = loss / 10000
        return accuracy,loss,y_true,y_pred

# Back propagation
def back_propagation(a_h_list, y, inp, del_theta, theta, batch_size, activation_func,no_of_hidden_layers,loss_type):
    h_counter = len(a_h_list) - 1
    grad_a = calc_aL(a_h_list[h_counter],y,loss_type)
    h_counter -= 2
    for i in range(no_of_hidden_layers, -1, -1):
        if i == 0:
            del_w = np.dot(grad_a, inp.T)
            del_b = grad_a
            del_theta[i] = np.add(del_theta[i], del_w)
            del_theta[i + no_of_hidden_layers + 1] = np.add(del_theta[i + no_of_hidden_layers + 1], del_b)
            break
        del_w = np.dot(grad_a, a_h_list[h_counter].T)
        del_b = grad_a
        del_theta[i] = np.add(del_theta[i], del_w)
        del_theta[i + no_of_hidden_layers + 1] = np.add(del_theta[i + no_of_hidden_layers + 1], del_b)
        grad_h_prev = np.dot(theta[i].T, grad_a)
        grad_a = grad_h_prev * calc_gdash(a_h_list[h_counter - 1], activation_func)
        h_counter -= 2

# Gradient Descent
def gradient_descent(eta, batch_size, epoch, theta, activation_func, validation_split, loss_type, alpha,no_of_hidden_layers):
    for itr in range(epoch):
        # Initialize gradients and total loss
        del_theta = [np.zeros_like(param) for param in theta]

        # Iterate through the training data
        for i in tqdm(range(int(60000 * (1 - validation_split)))):
            # Forward propagation
            a_h_list = forward_propagation(theta, np.float64(x_train[i].flatten().reshape((784, 1))), activation_func,no_of_hidden_layers)

            # Backpropagation
            back_propagation(a_h_list, y_train[i], np.float64((x_train[i].flatten()).reshape((784, 1))), del_theta, theta, batch_size, activation_func,no_of_hidden_layers,loss_type)

            # Update weights after every mini-batch
            if i % batch_size == 0 and i != 0:
                for j in range(len(theta)):
                    if(j<=no_of_hidden_layers):
                      del_theta[j] = (del_theta[j] / batch_size)
                    theta[j] = np.subtract(theta[j], eta * del_theta[j])-eta*alpha*theta[j]
                    del_theta[j] = del_theta[j] * 0

        # Calculate loss and accuracy
        train_accuracy,train_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"train",alpha)
        validation_accuracy,validation_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"validation",alpha)

        # Print epoch statistics
        wandb.log({'epoch':itr+1,
          'loss': train_loss ,
          'accuracy': train_accuracy,
          'val_loss': validation_loss,
          'val_accuracy': validation_accuracy
      })


# Momentum Gradient Descent
def momentum_gradient_descent(eta, batch_size, epoch, theta, beta, activation_func, validation_split, loss_type, alpha,no_of_hidden_layers):
    # Initialize previous history
    prev_history = [np.zeros_like(param) for param in theta]

    for itr in range(epoch):
        del_theta = [np.zeros_like(param) for param in theta]

        # Iterate through the training data
        for i in tqdm(range(int(60000 * (1 - validation_split)))):
            # Forward propagation
            a_h_list = forward_propagation(theta, np.float64(x_train[i].flatten().reshape((784, 1))), activation_func,no_of_hidden_layers)

            # Backpropagation
            back_propagation(a_h_list, y_train[i], np.float64((x_train[i].flatten()).reshape((784, 1))), del_theta, theta, batch_size, activation_func,no_of_hidden_layers,loss_type)

            # Update weights using momentum
            if i % batch_size == 0 and i != 0:
                for j in range(len(del_theta)):
                    if(j<=no_of_hidden_layers):
                      del_theta[j] = (del_theta[j] / batch_size)
                    prev_history[j] = np.add(beta * prev_history[j],eta * (del_theta[j]))
                    theta[j] = np.subtract(theta[j],  prev_history[j])-eta*alpha*theta[j]
                    del_theta[j] = del_theta[j] * 0

        # Calculate loss and accuracy
        train_accuracy,train_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"train",alpha)
        validation_accuracy,validation_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"validation",alpha)

        # Print epoch statistics
        wandb.log({'epoch':itr+1,
          'loss': train_loss ,
          'accuracy': train_accuracy,
          'val_loss': validation_loss,
          'val_accuracy': validation_accuracy
      })

# Nestrov Gradient Descent
def nesterov_gradient_descent(eta, batch_size, epoch, theta, beta, activation_func, validation_split, loss_type, alpha,no_of_hidden_layers):
    # Initialize previous history
    prev_history = [np.zeros_like(param) for param in theta]

    for itr in range(epoch):
        del_theta = [np.zeros_like(param) for param in theta]

        # Iterate through the training data
        for i in tqdm(range(int(60000 * (1 - validation_split)))):
            # Update weights using Nesterov accelerated gradient descent
            updated_theta = [theta[j] - beta * prev_history[j] for j in range(len(theta))]

            # Forward propagation
            a_h_list = forward_propagation(updated_theta, np.float64(x_train[i].flatten().reshape((784, 1))), activation_func,no_of_hidden_layers)

            # Backpropagation
            back_propagation(a_h_list, y_train[i], np.float64((x_train[i].flatten()).reshape((784, 1))), del_theta, updated_theta, batch_size, activation_func,no_of_hidden_layers,loss_type)

            # Update weights using momentum
            if i % batch_size == 0 and i != 0:
                for j in range(len(del_theta)):
                    if(j<=no_of_hidden_layers):
                      del_theta[j] = (del_theta[j] / batch_size)
                    prev_history[j] = np.add(beta * prev_history[j],eta*(del_theta[j]))
                    theta[j] = np.subtract(theta[j], prev_history[j])-eta*alpha*theta[j]
                    del_theta[j] = del_theta[j] * 0

        # Calculate loss and accuracy
        train_accuracy,train_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"train",alpha)
        validation_accuracy,validation_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"validation",alpha)

        # Print epoch statistics
        wandb.log({'epoch':itr+1,
          'loss': train_loss ,
          'accuracy': train_accuracy,
          'val_loss': validation_loss,
          'val_accuracy': validation_accuracy
      })

# RMS_Prop
def rmsprop(eta, batch_size, epoch, theta, beta, eps, activation_func, validation_split, loss_type, alpha,no_of_hidden_layers):
    # Initialize first  moment estimates
    v_theta = [np.zeros_like(param) for param in theta]

    for itr in range(epoch):
        del_theta = [np.zeros_like(param) for param in theta]

        # Iterate through the training data
        for i in tqdm(range(int(60000 * (1 - validation_split)))):
            # Forward propagation
            a_h_list = forward_propagation(theta, np.float64(x_train[i].flatten().reshape((784, 1))), activation_func,no_of_hidden_layers)

            # Backpropagation
            back_propagation(a_h_list, y_train[i], np.float64((x_train[i].flatten()).reshape((784, 1))), del_theta, theta, batch_size, activation_func,no_of_hidden_layers,loss_type)

            # Update weights using Adam optimizer
            if i % batch_size == 0 and i != 0:
                for j in range(len(theta)):
                    if(j<=no_of_hidden_layers):
                      del_theta[j] = (del_theta[j] / batch_size)
                    v_theta[j] = beta * v_theta[j] + (1 - beta) * (del_theta[j] ** 2)

                    # Update weights
                    theta[j] = np.subtract(theta[j], eta * (del_theta[j]/ (np.sqrt(v_theta[j] + eps))))-eta*alpha*theta[j]
                    del_theta[j] = del_theta[j] * 0

        # Calculate loss and accuracy
        train_accuracy,train_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"train",alpha)
        validation_accuracy,validation_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"validation",alpha)

        # Print epoch statistics
        wandb.log({'epoch':itr+1,
          'loss': train_loss ,
          'accuracy': train_accuracy,
          'val_loss': validation_loss,
          'val_accuracy': validation_accuracy
      })


# Adam Optimizer
def adam_optimizer(eta, batch_size, epoch, theta, beta1, beta2, eps, activation_func, validation_split, loss_type, alpha,no_of_hidden_layers):
    # Initialize first and second moment estimates
    m_theta = [np.zeros_like(param) for param in theta]
    v_theta = [np.zeros_like(param) for param in theta]

    for itr in range(epoch):
        del_theta = [np.zeros_like(param) for param in theta]

        # Iterate through the training data
        for i in tqdm(range(int(60000 * (1 - validation_split)))):
            # Forward propagation
            a_h_list = forward_propagation(theta, np.float64(x_train[i].flatten().reshape((784, 1))), activation_func,no_of_hidden_layers)

            # Backpropagation
            back_propagation(a_h_list, y_train[i], np.float64((x_train[i].flatten()).reshape((784, 1))), del_theta, theta, batch_size, activation_func,no_of_hidden_layers,loss_type)

            # Update weights using Adam optimizer
            if i % batch_size == 0 and i != 0:
                for j in range(len(theta)):
                    if(j<=no_of_hidden_layers):
                      del_theta[j] = (del_theta[j] / batch_size)
                    m_theta[j] = beta1 * m_theta[j] + (1 - beta1) * del_theta[j]
                    v_theta[j] = beta2 * v_theta[j] + (1 - beta2) * (del_theta[j] ** 2)

                    # Bias correction
                    m_hat = m_theta[j] / (1 - np.power(beta1, itr + 1))
                    v_hat = v_theta[j] / (1 - np.power(beta2, itr + 1))

                    # Update weights
                    theta[j] = np.subtract(theta[j], eta * m_hat / (np.sqrt(v_hat) + eps))-eta*alpha*theta[j]
                    del_theta[j] = del_theta[j] * 0

         # Calculate loss and accuracy
        train_accuracy,train_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"train",alpha)
        validation_accuracy,validation_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"validation",alpha)

        # Print epoch statistics
        wandb.log({'epoch':itr+1,
          'loss': train_loss ,
          'accuracy': train_accuracy,
          'val_loss': validation_loss,
          'val_accuracy': validation_accuracy
      })

# nadam Optimizer
def nadam_optimizer(eta, batch_size, epoch, theta, beta1, beta2, eps, activation_func, validation_split, loss_type, alpha ,no_of_hidden_layers):
    # Initialize first and second moment estimates
    m_theta = [np.zeros_like(param) for param in theta]
    v_theta = [np.zeros_like(param) for param in theta]

    for itr in range(epoch):
        del_theta = [np.zeros_like(param) for param in theta]

        # Iterate through the training data
        for i in tqdm(range(int(60000 * (1 - validation_split)))):
            # Forward propagation
            a_h_list = forward_propagation(theta, np.float64(x_train[i].flatten().reshape((784, 1))), activation_func,no_of_hidden_layers)

            # Backpropagation
            back_propagation(a_h_list, y_train[i], np.float64((x_train[i].flatten()).reshape((784, 1))), del_theta, theta, batch_size, activation_func,no_of_hidden_layers,loss_type)

            # Update weights using Adam optimizer
            if i % batch_size == 0 and i != 0:
                for j in range(len(theta)):
                    if(j<=no_of_hidden_layers):
                      del_theta[j] = (del_theta[j] / batch_size)
                    m_theta[j] = beta1 * m_theta[j] + (1 - beta1) * del_theta[j]
                    v_theta[j] = beta2 * v_theta[j] + (1 - beta2) * (del_theta[j] ** 2)

                    # Bias correction
                    m_hat = m_theta[j] / (1 - np.power(beta1, itr + 1))
                    v_hat = v_theta[j] / (1 - np.power(beta2, itr + 1))

                    # Update weights
                    theta[j] = np.subtract(theta[j], (eta / (np.sqrt(v_hat) + eps))*((beta1*m_hat)+((1-beta1)*(del_theta[j]/(1 - np.power(beta1, itr + 1))))))-eta*alpha*theta[j]
                    del_theta[j] = del_theta[j] * 0

         # Calculate loss and accuracy
        train_accuracy,train_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"train",alpha)
        validation_accuracy,validation_loss=calc_loss_acc(theta,validation_split,activation_func,loss_type,no_of_hidden_layers,"validation",alpha)

        # Print epoch statistics
        wandb.log({'epoch':itr+1,
          'loss': train_loss ,
          'accuracy': train_accuracy,
          'val_loss': validation_loss,
          'val_accuracy': validation_accuracy
      })

def plotConfusionMatrix(theta,activation_func,loss_type,no_of_hidden_layers,which_loss,alpha=0):
    classes=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt", "Sneaker","Bag", "Ankle boot"]
    accuracy,loss,y_true,y_pred=calc_loss_acc(theta,0,activation_func,loss_type,no_of_hidden_layers,which_loss,0)
    print(accuracy,loss)
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    wandb.log({"confusion_matrix": plt})
    plt.show()

def run_optimizer(eta, batch_size, epoch, theta, momentum, beta, beta1, beta2, eps, activation_func, validation_split, loss_type, alpha,optimizer,no_of_hidden_layers):
    if(optimizer=="sgd"):
      gradient_descent(eta,1,epoch,theta,activation_func,validation_split,loss_type,alpha,no_of_hidden_layers)
    elif(optimizer=="momentum"):
      momentum_gradient_descent(eta,batch_size,epoch,theta,momentum,activation_func,validation_split,loss_type,alpha,no_of_hidden_layers)
    elif(optimizer=="nag"):
      nesterov_gradient_descent(eta,batch_size,epoch,theta,momentum,activation_func,validation_split,loss_type,alpha,no_of_hidden_layers)
    elif(optimizer=="rmsprop"):
      rmsprop(eta,batch_size,epoch,theta,beta,eps,activation_func,validation_split,loss_type,alpha,no_of_hidden_layers)
    elif(optimizer=="adam"):
      adam_optimizer(eta,batch_size,epoch,theta,beta1,beta2,eps,activation_func,validation_split,loss_type,alpha,no_of_hidden_layers)
    elif(optimizer=="nadam"):
      nadam_optimizer(eta,batch_size,epoch,theta,beta1,beta2,eps,activation_func,validation_split,loss_type,alpha,no_of_hidden_layers)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Training Arguments')

    parser.add_argument('-wp', '--wandb_project', type=str, default='deep_learn_assignment_1',
                        help='Project name used to track experiments in Weights & Biases dashboard')

    parser.add_argument('-we', '--wandb_entity', type=str, default='deep_learn_assignment_1',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=["mnist", "fashion_mnist"],
                        help='Dataset choice: ["mnist", "fashion_mnist"]')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train neural network.')

    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size used to train neural network.')

    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=["mean_squared_error", "cross_entropy"],
                        help='Loss function choice: ["mean_squared_error", "cross_entropy"]')

    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help='Optimizer choice: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Learning rate used to optimize model parameters')

    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum used by momentum and nag optimizers.')

    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Beta used by rmsprop optimizer')

    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 used by adam and nadam optimizers.')

    parser.add_argument('-beta2', '--beta2', type=float, default=0.999,
                        help='Beta2 used by adam and nadam optimizers.')

    parser.add_argument('-eps', '--epsilon', type=float, default=1e-10,
                        help='Epsilon used by optimizers.')

    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005,
                        help='Weight decay used by optimizers.')

    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier',
                        choices=["random", "Xavier"],
                        help='Weight initialization choice: ["random", "Xavier"]')

    parser.add_argument('-nhl', '--num_layers', type=int, default=4,
                        help='Number of hidden layers used in feedforward neural network.')

    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help='Number of hidden neurons in a feedforward layer.')

    parser.add_argument('-a', '--activation', type=str, default='ReLU',
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help='Activation function choice: ["identity", "sigmoid", "tanh", "ReLU"]')

    return parser.parse_args()

args = parse_arguments() 	
# Initialize wandb
wandb.init(project=args.wandb_project)
if args.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif args.dataset == 'fashion_mnist' or args.dataset == None:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train,y_train=shuffle(x_train,y_train)
x_test, y_test=shuffle(x_test, y_test)
x_train=x_train/255.0
x_test=x_test/255.0	
validation_split=0.1	
# Set your hyperparameters from wandb config
wandb.run.name=f'hl_{args.num_layers}_hs_{args.hidden_size}_bs_{args.batch_size}_ac_{args.activation}_wInit_{args.weight_init}_lr_{args.learning_rate}_opt_{args.optimizer}_dataset_{args.dataset}'
theta = initialize_weights(args.weight_init, 784, args.num_layers, 10, args.hidden_size)
# Train your model
run_optimizer(args.learning_rate, args.batch_size, args.epochs, theta, args.momentum, args.beta, args.beta1, args.beta2, args.epsilon, args.activation,validation_split, args.loss, args.weight_decay, args.optimizer, args.num_layers)
