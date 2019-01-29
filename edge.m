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
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);

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
    device_index = zeros(device_num,1);
    
    
    [hiddenWeights, outputWeights, error] = train(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, epochs, batchSize, learningRate);
    
    % Load validation set.
    
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
    
    accuracy_array = [correctlyClassified/(correctlyClassified+classificationErrors)]

    Communication_time = 2000;
    last_device = 0;
    policy = 0;
    history = [];
    history_h = [];
    Ex = sum(sum(inputValues.^2))/(size(inputValues,1)*size(inputValues,2));
    P = (10^0.7);
    epochs = 50;
    theta_snr = 4;
    theta = 2;
    for cnt=1:Communication_time
       
        if policy==0
            last_device = floor(rand(1)*10)+1;
        end
        data = device_data(last_device,:,device_index(last_device)+1);
        h = wgn(1,2,-3);
        noisy1 = wgn(1,size(inputValues,1),0);
        noisy2 = wgn(1,size(inputValues,1),0);
        history_h(end+1) = sum(h.^2);
        data = sum(h.^2) * data + sqrt(Ex/P) * (h(1) * noisy1 + h(2) * noisy2);
        history = [history;data];
        data = sum(history)/sum(history_h);
        if policy==0
            initial_data(:,end+1) = data;
            initial_label(:,end+1) = device_label(last_device,:,device_index(last_device)+1);
        else
            initial_data(:,end) = data;
            initial_label(:,end) = device_label(last_device,:,device_index(last_device)+1);
        end
        [hiddenWeights, outputWeights, error, sample_error] = train(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, epochs, batchSize, learningRate);
        %theta_max = min(theta_snr, theta*sample_error);
        theta_max = 0;
        if sum(history_h) < theta_max
            policy = 1;
        else
            policy = 0;
            device_index(last_device) = device_index(last_device)+1;
            history = [];
            history_h = []; 
        end

        if mod(cnt,10)==0
            [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
             if mod(cnt,50)==0
                cnt
                correctlyClassified/(correctlyClassified+classificationErrors)
            end
        
            accuracy_array(end+1) = correctlyClassified/(correctlyClassified+classificationErrors);
        end
         
    end
    plot(accuracy_array);
    save('noretrans','accuracy_array')
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
end