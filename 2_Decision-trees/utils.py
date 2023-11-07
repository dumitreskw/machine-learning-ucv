def compute_accuracy_and_error_rate(actual_values, predicted_values, total_predictions):
    correct_predictions = 0
    wrong_predictions = 0

    for actual, predicted in zip(actual_values, predicted_values):
        if actual == predicted:
            correct_predictions += 1
        else:
            wrong_predictions += 1
        
    print('\n --- Correct predictions: ', correct_predictions)
    print('\n --- Wrong predictions: ', wrong_predictions)   
    accuracy = correct_predictions / total_predictions 
    error_rate = wrong_predictions / total_predictions 
    pair = (accuracy, error_rate)
    
    return pair