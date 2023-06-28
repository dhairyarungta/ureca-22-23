#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 16000 // Assuming input size of 1 second (16,000 samples)
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 2

typedef struct {
    double weights1[16][HIDDEN_SIZE];
    double bias1[HIDDEN_SIZE];
    double weights2[HIDDEN_SIZE][OUTPUT_SIZE];
    double bias2[OUTPUT_SIZE];
} VoiceDetectionModel;

void initializeModel(VoiceDetectionModel *model) {
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

int predict(VoiceDetectionModel *model, double *inputs) {
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];

    // Compute hidden layer values using sigmoid activation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += inputs[j] * model->weights1[j][i];
        }
        hidden[i] = sigmoid(sum + model->bias1[i]);
    }

    // Compute output layer values using sigmoid activation
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden[j] * model->weights2[j][i];
        }
        output[i] = sigmoid(sum + model->bias2[i]);
    }

    // Perform argmax to determine the predicted class (Yes/No)
    int max_index = 0;
    double max_value = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_index = i;
        }
    }

    return max_index;
}

int main() {
    // Initialize the model
    VoiceDetectionModel model;
    initializeModel(&model);

    // Read the audio data from a file into the 'inputs' array
    double inputs[INPUT_SIZE];






    // Perform inference
    int predicted_class = predict(&model, inputs);

    printf("Predicted class: %s\n", (predicted_class == 0) ? "No" : "Yes");

    return 0;
}
