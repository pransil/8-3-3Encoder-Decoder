# test.py
import torch
from datetime import datetime

# Create test data, all 8 one-hot vectors
def create_test_inputs(n):
    test_inputs = torch.zeros((n, n))
    for i in range(n):
        test_inputs[i, i] = 1
    #print(test_inputs)
    # Convert test_inputs to torch tensor if necessary
    if not isinstance(test_inputs, torch.Tensor):
        test_inputs = torch.tensor(test_inputs, dtype=torch.float32)    # maybe change to int8?    
    return test_inputs

# Test the model on all 8 one-hot vectors
# Record accuracy
def test_model2(model, test_inputs):
    correct = 0
    total = 0   
    #run the model on the test inputs, and check if the output is the same as the input, create a confusion matrix  
    confusion_matrix = torch.zeros(test_input_dim, test_input_dim)
    with torch.no_grad():
        outputs = model(test_inputs)
        # If outputs are probabilities, round to nearest int
        predicted = torch.round(outputs)
        
        # Compare predicted to input
        correct = (predicted == test_inputs).all(dim=1).sum().item()
        total = test_inputs.size(0)
        accuracy = correct / total
        for i in range(test_input_dim):
            confusion_matrix[i, :] = predicted[0]  # Take the first (and only) row of predictions
        write_confusion_matrix_to_file(confusion_matrix, accuracy, correct, total)
        print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total} correct)")
    return accuracy, confusion_matrix

def test_model(model, test_inputs):
    correct = 0
    total = 0   
    dim = test_inputs.shape[0]
    
    #run the model on the test inputs, and check if the output is the same as the input, create a confusion matrix  
    #confusion_matrix = torch.zeros(test_input_dim, test_input_dim)
    with torch.no_grad():
        outputs = model(test_inputs)
        # If outputs are probabilities, round to nearest int
        predicted = torch.round(outputs)
        print("test-predicted: \n", predicted)
        # Compare predicted to input
        correct = (predicted == test_inputs).all(dim=1).sum().item()
        total = test_inputs.size(0)
        accuracy = correct / total
    confusion_matrix = create_confusion_matrix(model, dim)
    for i in range(dim):
        confusion_matrix[i, :] = predicted[0]  # Take the first (and only) row of predictions

    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total} correct)")
    write_confusion_matrix_to_file(confusion_matrix, accuracy, correct, total)
    return accuracy, confusion_matrix

def test_model_sav(model, test_inputs):
    correct = 0
    total = 0   
    dim = test_inputs.shape[0]
    
    #run the model on the test inputs, and check if the output is the same as the input, create a confusion matrix  
    #confusion_matrix = torch.zeros(test_input_dim, test_input_dim)
    for i in range(dim):
        with torch.no_grad():
            outputs = model(test_inputs)
            print("test-outputs: \n", outputs)
            # If outputs are probabilities, round to nearest int
            predicted = torch.round(outputs)
            print("test-predicted: \n", predicted)
            # Compare predicted to input
            correct = (predicted == test_inputs).all(dim=1).sum().item()
            total = test_inputs.size(0)
            accuracy = correct / total
    confusion_matrix = create_confusion_matrix(model, dim)
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total} correct)")
    write_confusion_matrix_to_file(confusion_matrix, accuracy, correct, total)

    return accuracy, confusion_matrix


# Create a confusion matrix
def create_confusion_matrix(model, test_input_dim):
    test_inputs = create_test_inputs(test_input_dim)
    confusion_matrix = torch.zeros(test_input_dim, test_input_dim)
    for i in range(test_input_dim):
        with torch.no_grad():
            output = model(test_inputs[i:i+1])  # Process one input at a time
           
            predicted = torch.round(output)
            print("predicted: ", predicted)
            confusion_matrix[i, :] = predicted[0]  # Take the first (and only) row of predictions
    return confusion_matrix

# Write the confusion matrix to a file
def write_confusion_matrix_to_file(confusion_matrix, accuracy, correct, total):
    with open('838.txt', 'w') as f:
        # add a header to the file
        f.write("Confusion Matrix for 838.py:\n")
        f.write(str(confusion_matrix))
        # append the accuracy to the file
        f.write(f"\nAccuracy: {accuracy*100:.2f}% ({correct}/{total} correct)")
        # append the date and time to the file
        f.write(f"\nDate and Time: {datetime.now()}")