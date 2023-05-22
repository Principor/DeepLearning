#include <iostream>
#include <deep_learning.h>

int main()
{
    int training_size = 20;
    //Training inputs are x and y co-ordinates in the range (-1,1)
    Tensor train_x = Tensor::uniform({ training_size, 2 }, -1.0f, 1.0f);
    //Training labels are distances from the origin
    float* testValues = new float[training_size];
    for (int i = 0; i < training_size; i++) {
        float x = train_x.at({ i,0 }), y = train_x.at({ i,1 });
        testValues[i] = std::sqrt(x * x + y * y);
    }
    Tensor train_y = Tensor::fromValues(testValues, { 20,1 });

    Tensor weights1 = Tensor::uniform({ 2,5 }, 0.0, 1.0).requireGradient();
    Tensor weights2 = Tensor::uniform({ 5,1 }, 0.0, 1.0).requireGradient();

    for (int i = 0; i < 100; i++)
    {
        Tensor z1 = Tensor::matrixMultiply(train_x, weights1);
        Tensor a1 = Tensor::ReLU(z1);
        Tensor z2 = Tensor::matrixMultiply(a1, weights2);

        Tensor loss = Tensor::meanSquaredErrorLoss(z2, train_y);
        std::cout << loss.item() << std::endl;
        loss.backwards();
        for (Tensor* weight : { &weights1, &weights2 })
        {
            Tensor offset = Tensor::multiply(*weight->getGradient(), 0.1f);
            *weight = Tensor::subtract(*weight, offset).detached().requireGradient();
        }
    }
}