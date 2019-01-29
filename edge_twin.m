function [] = applyStochasticSquaredErrorTwoLayerPerceptronMNIST()
%applyStochasticSquaredErrorTwoLayerPerceptronMNIST Train the two-layer
%perceptron using the MNIST dataset and evaluate its performance.

    % Load MNIST.
    inputValues = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    
       
    % Transform the labels to correct target values.
    targetValues = 0.*ones(10, size(labels, 1));
    
    
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end;
    
    % Choose form of MLP:
    numberOfHiddenUnits = 700;
    numberOfHiddenUnits_twin = 70;
    
    % Choose appropriate parameters.
    learningRate = 0.1;
    
    % Choose activation function.
    activationFunction = @logisticSigmoid;
    dActivationFunction = @dLogisticSigmoid;
    
    % Choose batch size and epochs. Remember there are 60k input values.
    batchSize = 4;
    epochs = 4000;
     fprintf('%d\n',size(inputValues));
    fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
    fprintf('Learning rate: %d.\n', learningRate);
    
    inputValues_test = loadMNISTImages('t10k-images.idx3-ubyte');
    labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

    inputValues_test = inputValues_test(:,1:1000);
    labels_test = labels_test(1:1000);
    
   % targetValues =  targetValues(:,1:20000);
   % inputValues = inputValues(:,1:20000);
    size(inputValues)
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    hiddenWeights_twin = rand(numberOfHiddenUnits_twin, inputDimensions);
    outputWeights_twin = rand(outputDimensions, numberOfHiddenUnits_twin);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    hiddenWeights_twin = hiddenWeights_twin./size(hiddenWeights_twin, 2);
    outputWeights_twin = outputWeights_twin./size(outputWeights_twin, 2);

    %Distribute sample
    randindex = randperm(size(inputValues,2));
    device_num = 10;
    initial_size = 50;
    device_size = floor((size(inputValues,2)-initial_size)/device_num);
    initial_size = size(inputValues,2) - device_size * device_num;
    initial_data = inputValues(:,randindex(1:initial_size));
    initial_label = targetValues(:,randindex(1:initial_size));
    device_data =  zeros(device_num,size(inputValues,1),device_size);
    device_label = zeros(device_num,size(targetValues,1),device_size);
    for cnt=1:device_num
        device_data(cnt,:,:) = inputValues(:,randindex(initial_size + (cnt-1) * device_size + 1:initial_size + cnt * device_size));
        device_label(cnt,:,:) = targetValues(:,randindex(initial_size + (cnt-1) * device_size + 1:initial_size + cnt * device_size));
    end
    device_flag = zeros(device_num,device_size);
    device_index = zeros(device_num);
    
    
    [hiddenWeights, outputWeights, hiddenWeights_twin, outputWeights_twin,error] = train_twin(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights,numberOfHiddenUnits_twin, hiddenWeights_twin, outputWeights_twin, initial_data, initial_label, epochs, batchSize, learningRate);
    
    % Load validation set.
    
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
    
    accuracy_array = [correctlyClassified/(correctlyClassified+classificationErrors)]

    Communication_time = 2000;
    epochs = 50;

    for cnt=1:Communication_time
        last_device = floor(rand(1)*device_num)+1;
        if cnt>Communication_time/4
            data_num = -1;
            max_error = 0;
            for data_cnt=1:device_size
                if device_flag(last_device, data_cnt)>0
                    continue
                end
                inputVector = device_data(last_device,:, data_cnt);
                targetVector = device_label(last_device,:, data_cnt);
                error = norm(activationFunction(outputWeights_twin*activationFunction(hiddenWeights_twin*inputVector')) - targetVector, 2);
                if data_num<0
                    max_error = error;
                    data_num = data_cnt;
                    continue
                end
                if error > max_error
                    max_error = error;
                    data_num = data_cnt;
                end
            end
        else
            device_index(last_device) = device_index(last_device) + 1;
            data_num = device_index(last_device);
        end
        device_flag(last_device, data_num) = 1;
        initial_data(:,end+1) = device_data(last_device,:,data_num);
        initial_label(:,end+1) = device_label(last_device,:,data_num);
        [hiddenWeights, outputWeights, hiddenWeights_twin, outputWeights_twin,error] = train_twin(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights,numberOfHiddenUnits_twin, hiddenWeights_twin, outputWeights_twin, initial_data, initial_label, epochs, batchSize, learningRate);
        
        if mod(cnt,20)==0
            [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
             if mod(cnt,50)==0
                cnt
                correctlyClassified/(correctlyClassified+classificationErrors)
            end
        
            accuracy_array(end+1) = correctlyClassified/(correctlyClassified+classificationErrors);
        end
         
    end
    plot(accuracy_array);
    save('notwin_twin','accuracy_array')
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
end