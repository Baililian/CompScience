import math

class NaiveBayesClassifier:
    def __init__(self):
        self.class_prob = {}
        self.feature_prob = {}
        self.classes = []
    
    def train(self, X, y):
        total_samples = len(y)
        self.classes = list(set(y))
        
        # CALCULATE PRIOR PROBABILITIES FOR EACH CLASS 
        for c in self.classes:
            self.class_prob[c] = y.count(c) / total_samples
        
        # CALCULATE CONDITIONAL PROBABILITIES FOR EACH FEATURE GIVEN A CLASS
        self.feature_prob = {c: {} for c in self.classes}
        
        for c in self.classes:
            class_samples = [X[i] for i in range(len(X)) if y[i] == c]
            total_class_samples = len(class_samples)
            
            for j in range(len(X[0])):
                feature_values = [sample[j] for sample in class_samples]
                unique_values = set(feature_values)
                self.feature_prob[c][j] = {}
                
                for value in unique_values:
                    self.feature_prob[c][j][value] = feature_values.count(value) / total_class_samples
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            class_scores = {}
            for c in self.classes:
                # START WITH THE PRIOR PROBABILITY
                class_scores[c] = math.log(self.class_prob[c])
                
                for j in range(len(x)):
                    feature_value = x[j]
                    
                    # APPLY LAPLACE SMOOTHING TO HANDLE ZERO PROBABILITIES
                    if feature_value in self.feature_prob[c][j]:
                        class_scores[c] += math.log(self.feature_prob[c][j][feature_value])
                    else:
                        class_scores[c] += math.log(1e-6)  # SMALL PROBABILITY FOR UNSEEN VALUES
            
            # ASSIGN THE CLASS WITH THE HIGHEST PROBABILITY
            predictions.append(max(class_scores, key=class_scores.get))
        
        return predictions

# EXAMPLE
X_train_ex = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
                   [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
                   [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
y_train_ex = ['N', 'N', 'Y', 'Y', 'N',
                   'N', 'N', 'Y', 'Y', 'Y',
                   'Y', 'Y', 'Y', 'Y', 'N']

X_test_ex = [[2, 'S'], [3, 'M'], [1, 'L']]

ex = NaiveBayesClassifier()
ex.train(X_train_ex, y_train_ex)
predictions_ex = ex.predict(X_test_ex)
print("Example Predictions:", predictions_ex)
