# XOR Neural Network (NumPy Implementation)

This project implements a simple **Multi-Layer Perceptron (MLP)** from scratch using **NumPy** to solve the classic **XOR classification problem**.  
It demonstrates forward propagation, backpropagation, weight updates with gradient descent, and evaluation using custom metrics.

---

## ✨ Features
- Implements a 2-layer MLP (hidden + output layer) from scratch.  
- Supports **Sigmoid** and **ReLU** activations for the hidden layer.  
- Training using **gradient descent**.  
- Custom metrics: Accuracy, Precision, Recall, F1-score.  
- **ROC Curve** plotting with AUC score using `matplotlib`.  
- Experiments with different learning rates and training epochs.  

---

## 📂 Project Structure
```
├── xor_nn.py        # Main script (NeuralNet class + metrics + experiments)
├── README.md        # Documentation
```

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/xor-nn-numpy.git
cd xor-nn-numpy
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
matplotlib
scikit-learn
```

---

## ▶️ Usage
Run the main script:

```bash
python xor_nn.py
```

The script will:
1. Train the MLP on the XOR dataset with different hyperparameters.
2. Print evaluation metrics.
3. Display ROC curves for each experiment.

---

## 📊 Example Output
Training with:
- **Learning rate = 0.1, Epochs = 1000, Activation = Sigmoid**
- **Learning rate = 0.5, Epochs = 5000, Activation = ReLU**

Example results (values may vary slightly due to random initialization):

```
Running experiment with: learning_rate=0.1, epochs=1000, activation=sigmoid
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-score: 1.00

Running experiment with: learning_rate=0.5, epochs=5000, activation=relu
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-score: 1.00
```

Additionally, ROC curves will be displayed in popup windows.  

---

## 📘 Learning Notes
- XOR cannot be solved by a single-layer perceptron → requires **non-linear hidden layer**.  
- Backpropagation updates weights using gradients from the chosen activation function.  
- Sigmoid vs ReLU:
  - **Sigmoid** works but can suffer from vanishing gradients.
  - **ReLU** is faster and often converges better.

---

## 📜 License
This project is open-source and available under the **MIT License**.
