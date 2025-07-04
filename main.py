import json
import random
import csv
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from config import *
from matrix import Matrix


def get_data(test_only):
    """
    Loads the data into 4 lists, 2 of which become matrices.
    :param test_only: we only grab test data in this case
    y_test: (TEST_DATA_SIZE) the list of answers used for testing later
    x_test: (TEST_DATA_SIZE, 784) the matrix the has the pixels of each image in the test set
    y_train: (WHOLE_SET-TEST_DATA_SIZE) the list of answers used for training the model
    x_train: (WHOLE_SET-TEST_DATA_SIZE, 784) the matrix the has the pixels of each image in the training set
    :return: y_test, x_test, y_train, x_train
    """

    y_test, x_test, y_train, x_train = [], [], [], []
    if test_only:
        print("Reading data, this may take some time...")
    else:
        print("Reading full data, this takes ~10 minutes...")

    with open(DATA_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for i, row in enumerate(reader):
            label = int(row[0])
            pixels = [int(p) / 255 for p in row[1:]]

            if test_only:
                y_test.append(label)
                x_test.append(pixels)
                if i + 1 >= TEST_DATA_SIZE:
                    break
            else:
                if i < TEST_DATA_SIZE:
                    y_test.append(label)
                    x_test.append(pixels)
                else:
                    y_train.append(label)
                    x_train.append(pixels)

    return y_test, Matrix(data=x_test), y_train, Matrix(data=x_train)



def create_initial_model():
    """
    Creates the initial weights and biases for the model. This is random.
    :return: w1, b1, w2, b2
    """

    # Layer 1 weights
    w1 = Matrix(NEURON_COUNT, 784)

    # Layer 1 biases
    b1 = Matrix(NEURON_COUNT, 1)

    # Layer 2 weights
    w2 = Matrix(26, NEURON_COUNT)

    # Layer 2 biases
    b2 = Matrix(26, 1)

    w1.randomize()
    b1.randomize()
    w2.randomize()
    b2.randomize()
    return w1, b1, w2, b2

def forward_propagate(x, w1, b1, w2, b2):
    """
    Performs forward propagation on the matrix x. Returns the "prediction" of a2, along with the intermediary
    steps used to get there: z2, a1, z1
    :param x: The input, pixels for each item in the batch (784, BATCH_SIZE)
    :param w1: Weights for layer 1 (784, NEURON_COUNT)
    :param b1: Biases for layer 1 (NEURON_COUNT, 1)
    :param w2: Weights for layer 2 (NEURON_COUNT, 26)
    :param b2: Biases for layer 2 (26, 1)
    :return: a2, z2, a1, z1
    """

    # Idea: compute z1 using weights and biases from layer 1

    z1 = w1.dot(x.T()) + b1             # (NEURON_COUNT, BATCH_SIZE) = (NEURON_COUNT, 784)dot(784, BATCH_SIZE) + (NEURON_COUNT, 1)
    a1 = z1.ReLU()                      # (NEURON_COUNT, BATCH_SIZE) = ReLU each item (NEURON_COUNT, BATCH_SIZE)

    z2 = w2.dot(a1) + b2                # (26, BATCH_SIZE) = (26, NEURON_COUNT)dot(NEURON_COUNT, BATCH_SIZE) + (26, 1)
    a2 = z2.softmax()                   # (26, BATCH_SIZE) = softmax across cols (26, BATCH_SIZE)

    return a2, z2, a1, z1

def back_propagate(y, x, a2, z2, a1, z1, w2):
    """
    Performs backpropagation on the predictions (a2, z2, a1, z1) vs the answers (y).
    :param y: The solutions, formatted it one_hot matrix orientation
    :param x: The pixel input. Used to calculate the derivative of dw1.
    :param a2: Our final predictions calculated in forward propagation
    :param z2: Our z2 values. Essentially, our predictions before using softmax.
    :param a1: Our a1 values, found in forward propagation
    :param z1: Our z1 values, found in forward propagation
    :param w2: The weights in layer 2. Critical for finding the derivatives of dw1 and db1
    :return: dw1, db1, dw2, db2
    """
    m = y.m
    dz2 = a2 - y                            # (26, m) = (26, m) - (26, m)
    dw2 = a1.dot(dz2.T())/m                 # (NEURON_COUNT, 26) = (NEURON_COUNT, m)dot(m, 26)
    db2 = dz2.get_matrix_averaged_column()  # (26, 1) = average the dz2 columns into one

    da1 = w2.T().dot(dz2)                   # (NEURON_COUNT, m) = (NEURON_COUNT, 26)dot(26, m)
    dz1 = z1.d_ReLU() * da1                 # (NEURON_COUNT, m) = (NEURON_COUNT, m) * (NEURON_COUNT, m)
    dw1 = dz1.dot(x)/m                      # (NEURON_COUNT, 784) = (NEURON_COUNT, m)dot(m, 784)
    db1 = dz1.get_matrix_averaged_column()  # (NEURON_COUNT, 1) = average the dz1 columns into one

    return dw1, db1, dw2.T(), db2

def get_one_hot_y(y):
    """
    One-hots the 1d list y. This puts it in format such that it can be subtracted from a2, as
    it will have the same dimensions. https://en.wikipedia.org/wiki/One-hot
    For example:
    [1, 4] > [0, 0]
             [1, 0]
             [0, 0]
             [0, 1]
    :param y: 1d array y
    :return: 2d one-hotted array y
    """
    one_hot_y = Matrix(26, len(y))
    for i, ans in enumerate(y):
        one_hot_y[ans, i] = 1
    return one_hot_y

def save_model(filename, w1, b1, w2, b2):
    """
    Saves the model to {filename}
    :param filename: the filename to save the model to
    :param w1: weights from layer 1
    :param b1: biases from layer 1
    :param w2: weights from layer 2
    :param b2: biases from layer 2
    :return: None
    """
    print("Saving the model...")
    model = {
        "w1": w1.matrix,
        "b1": b1.matrix,
        "w2": w2.matrix,
        "b2": b2.matrix,
    }
    with open(filename, "w") as f:
        json.dump(model, f)
    print("Model saved!")

def load_model(filename):
    """
    Loads the model from {filename}
    :param filename: filename to load from
    :return: weights and biases of the model
    """
    with open(filename, "r") as f:
        model = json.load(f)

    w1 = Matrix(data=model["w1"])
    b1 = Matrix(data=model["b1"])
    w2 = Matrix(data=model["w2"])
    b2 = Matrix(data=model["b2"])

    return w1, b1, w2, b2

def select_game_mode():
    """
    Selects the game mode between quiz, train and metrics.
    :return: "q" or "t" or "m"
    """
    game_mode = -1
    while game_mode not in ["q", "t", "m"]:
        game_mode = input("Please press the letter associated with your selection: \n"
                        "\t > Run quiz (q)\n"
                        "\t > Train model (t)\n"
                        "\t > Get model metrics (m)\n"
                        "Choice: ").lower()
    return game_mode

def train_model(w1, b1, w2, b2, x_train, y_train):
    """
    This is the main loop that trains the model. It trains by calling {back_propagate}
    over and over again on batches of the data. It repeats for {NUM_EPOCHS} epochs. The model is saved
    every {SAVE_EVERY} batches.
    :param w1: weights layer 1
    :param b1: biases layer 1
    :param w2: weights layer 2
    :param b2: biases layer 2
    :param x_train: training data pixels
    :param y_train: training data solutions
    :return: None
    """
    m = len(y_train)  # total training examples
    num_batches = (m + BATCH_SIZE - 1) // BATCH_SIZE
    print("Entering main training loop...")
    for epoch in range(NUM_EPOCHS):

        # ── shuffle dataset indices once per epoch ───────────────
        indices = list(range(m))
        random.shuffle(indices)

        for batch_idx, start in enumerate(range(0, m, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, m)
            batch_ix = indices[start:end]

            # slice mini-batch
            xb = Matrix(data=[x_train.matrix[i][:] for i in batch_ix])  # (batch, 784)
            yb = [y_train[i] for i in batch_ix]  # labels

            # forward / backward
            a2, z2, a1, z1 = forward_propagate(xb, w1, b1, w2, b2)
            dw1, db1, dw2, db2 = back_propagate(get_one_hot_y(yb), xb,
                                                a2, z2, a1, z1, w2)

            # SGD update
            w1 -= dw1 * LEARNING_RATE
            b1 -= db1 * LEARNING_RATE
            w2 -= dw2 * LEARNING_RATE
            b2 -= db2 * LEARNING_RATE

            # ── batch-level metric every PRINT_EVERY batches ─────
            if batch_idx % PRINT_EVERY == 0:
                preds_b = a2.get_matrix_maxes()
                acc_b = sum(preds_b[k] == yb[k] for k in range(len(yb))) / len(yb)
                print(f"epoch {epoch:3d} | batch {batch_idx:3d}/{num_batches} "
                      f"| batch-acc = {acc_b:.3f}")
            if batch_idx % SAVE_EVERY == 0:
                print("\n\n")
                save_model("model.json", w1, b1, w2, b2)
                print("\n\n")


def compete_with_ai(w1, b1, w2, b2, x_test, y_test):
    """
    Performs a small game that displays the images on the screen and allows the player to guess the value
    :param x_test: the pixel input for the test data
    :param y_test: the answers for the test data
    :param w1: weights for layer 1
    :param b1: biases for layer 1
    :param w2: weights for layer 2
    :param b2: biases for layer 2
    :return: None
    """
    a2, *_ = forward_propagate(x_test, w1, b1, w2, b2)
    preds = a2.get_matrix_maxes()
    print("\n--- Human vs AI ---")

    score_ai = 0
    score_human = 0

    alphabet = [chr(ord('a') + i) for i in range(26)]

    for _ in range(10):

        # Get column i (one image) and reshape to 28x28 for plotting
        i = random.randint(0, x_test.m - 1)  # pick random row/image

        # Get the i-th row (a flat 784-pixel image)
        flat = x_test.matrix[i]

        # Reshape into 28x28 for plotting
        img = [flat[r * 28:(r + 1) * 28] for r in range(28)]

        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.title("What digit is this?")
        plt.axis('off')
        plt.show()

        try:
            guess = input("Your guess: ")
            guess = alphabet.index(guess)
        except:
            guess = -1

        correct = int(y_test[i])
        ai_guess = preds[i]

        print(f"AI guessed: {alphabet[ai_guess]} | Correct: {alphabet[correct]}")
        if guess == correct:
            score_human += 1
        if ai_guess == correct:
            score_ai += 1

    print(f"\nFinal Score — Human: {score_human} | AI: {score_ai}")
    if input("Play again? (y/n) ") == "y":
        compete_with_ai(w1, b1, w2, b2, x_test, y_test)

def get_metrics(w1, b1, w2, b2, x_test, y_test):
    """
    Evaluates and prints the percentage accuracy of the model on the test data
    :param w1: Weights layer 1
    :param b1: Biases layer 1
    :param w2: Weights layer 2
    :param b2: Biases layer 2
    :param x_test: Pixel data from test data
    :param y_test: answers from test data
    :return: None
    """
    start = time.time()
    a2_full, *_ = forward_propagate(x_test, w1, b1, w2, b2)
    end = time.time()
    preds_full = a2_full.get_matrix_maxes()
    acc_full = sum(preds_full[j] == y_test[j] for j in range(x_test.m)) / x_test.m * 100


    def get_trickiest_chars():
        """
        Gets the trickiest characters with the worst rates of AI guessing accurately.
        :return: those characters
        """
        correct_by_label = defaultdict(int)
        total_by_label = defaultdict(int)

        # Go through predictions
        for j in range(x_test.m):
            true = y_test[j]
            pred = preds_full[j]
            total_by_label[true] += 1
            if pred == true:
                correct_by_label[true] += 1

        # Calculate per-letter accuracy
        accuracy_by_label = {
            label: correct_by_label[label] / total_by_label[label]
            for label in total_by_label
        }

        # Sort by lowest accuracy
        sorted_labels = sorted(accuracy_by_label.items(), key=lambda x: x[1])

        # Convert numeric labels to letters
        alphabet = [chr(ord('a') + j) for j in range(26)]
        return [alphabet[label] for label, _ in sorted_labels[:3]], sorted_labels

    print(f"{"\n"*5}\t\t\t    SUMMARY OF RESULTS")

    print(f"Metrics are based on the testing data of size {TEST_DATA_SIZE}.\n"
          f"\t Accuracy: {acc_full:.2f}%\t\tTime: {end-start:.4f}s\n\n"
          f"\t\t\t    Trickiest letters: ")
    chars, labels = get_trickiest_chars()
    for i in range(len(chars)):
        print(f"\t\t\t\t    {chars[i]}: {labels[i][1]*100:.2f}%")
    print("\n")
    time.sleep(1)


def run():
    """
    Manages the control through the terminal.
    :return: None
    """

    try:
        print("Loading the model...")
        w1, b1, w2, b2 = load_model("model.json")
    except:
        print("No model found. Starting from scratch and creating initial weights...")
        w1, b1, w2, b2 = create_initial_model()

    game_mode = select_game_mode()
    y_test, x_test, y_train, x_train= get_data(test_only=game_mode != "t")  # only care about training data if {game_mode} is training


    match game_mode:
        case "t":
            train_model(w1, b1, w2, b2, x_train, y_train)
        case "q":
            compete_with_ai(w1, b1, w2, b2, x_test, y_test)
        case "m":
            get_metrics(w1, b1, w2, b2, x_test, y_test)




if __name__ == '__main__':
    run()