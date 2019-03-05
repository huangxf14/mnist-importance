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
    
    
    data = inputValues(:,1111);
    datax = imresize(imresize(reshape(data,[28,28]),[4,4]),[28,28]);
    imwrite(datax,'4.jpg')
    datax = imresize(imresize(reshape(data,[28,28]),[5,5]),[28,28]);
    imwrite(datax,'5.jpg')
    datax = imresize(imresize(reshape(data,[28,28]),[7,7]),[28,28]);
    imwrite(datax,'7.jpg')
    datax = imresize(imresize(reshape(data,[28,28]),[14,14]),[28,28]);
    imwrite(datax,'14.jpg')
    datax = imresize(imresize(reshape(data,[28,28]),[21,21]),[28,28]);
    imwrite(datax,'21.jpg')
    datax = imresize(imresize(reshape(data,[28,28]),[28,28]),[28,28]);
    imwrite(datax,'28.jpg')
    
    %Distribute sample
    randindex = randperm(size(inputValues,2));
    load('weights');
    device_num = 10;
    initial_size = 50;
    device_size = floor((size(inputValues,2)-initial_size)/device_num);
    initial_size = size(inputValues,2) - device_size * device_num;
    initial_data = inputValues(:,randindex(1:initial_size));
    initial_label = targetValues(:,randindex(1:initial_size));
    device_data =  zeros(device_num,size(inputValues,1),device_size);
    device_label = zeros(device_num,size(targetValues,1),device_size);
    device_size_arrary = zeros(device_num,1);
    
    for cnt=initial_size+1:size(inputValues,2)
        temp_index = find(targetValues(:,cnt)~=0);
        device_size_arrary(temp_index) = device_size_arrary(temp_index) + 1;
        device_data(temp_index,:,device_size_arrary(temp_index)) = inputValues(:,cnt);
        device_label(temp_index,:,device_size_arrary(temp_index)) = targetValues(:,cnt);
    end
    
    device_index = zeros(device_num,1);
    
    
    %[hiddenWeights, outputWeights, error] = train_digit(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, epochs, batchSize, learningRate);
%     save('weights','randindex','hiddenWeights','outputWeights')
    % Load validation set.
    
    [x, y, error_tot] = train_device(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, 25, batchSize, learningRate,0);
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
    
    accuracy_array = [correctlyClassified/(correctlyClassified+classificationErrors)]

    Communication_time = 28*28*1000*2;
    last_device = 0;
    epochs = 50;
    policy = 2;
    theta = 8;
    cnt = 0;
    last_cnt = 0;
%     policy_array=[];
    device_error = ones(device_num,1);
    alpha = 2;
    while cnt<Communication_time
       
        if true
            device_error_exp = exp(device_error);
            randerror = rand(1) * sum(device_error_exp);
            last_device = 1;
            sumerror = device_error_exp(1);
            while sumerror < randerror
                last_device = last_device + 1;
                sumerror = sumerror + device_error_exp(last_device);
            end
%             if sum(device_index==0) == 0
%                 max_reward = device_error_exp(1) + sqrt((alpha * log(sum(device_index)) / device_index(1))/2);
%                 for last_device_cnt=2:device_num
%                     temp_reward = device_error_exp(last_device_cnt) + sqrt((alpha * log(sum(device_index)) / device_index(last_device_cnt))/2);
%                     if temp_reward > max_reward
%                         last_device = last_device_cnt;
%                         max_reward = temp_reward;
%                     end                        
%                 end
%             else
%                 last_device = floor(rand(1)*10)+1;  
%             end
        end
        last_device = floor(rand(1)*10)+1;  
%         policy_array(end+1) = policy;
        data = device_data(last_device,:,device_index(last_device)+1);

        data = reshape(imresize(imresize(reshape(data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
        cnt = cnt + 7*7*policy*policy;
        
        if policy==2
            initial_data(:,end+1) = data;
            initial_label(:,end+1) = device_label(last_device,:,device_index(last_device)+1);
        else
            initial_data(:,end) = data;
            initial_label(:,end) = device_label(last_device,:,device_index(last_device)+1);
        end
        
        newepochs = ceil(epochs * 7 * 7 * policy * policy / (28*28)); 
        [hiddenWeights, outputWeights, error, sample_error,error_tot] = train_device(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, newepochs, batchSize, learningRate,error_tot);
        
        %theta = 100000000;
%         if (sample_error > theta * error_tot)&&(policy<4)
%             policy = policy + 1;
%         else
%             policy = 1;
%             device_index(last_device) = device_index(last_device) + 1;
%         end
        device_error(last_device) = device_error(last_device) * 0.5 + 0.5 * sample_error/error_tot;
                
        device_index(last_device) = device_index(last_device) + 1;

        if cnt-last_cnt > 28*28*10
            [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
            last_cnt = last_cnt + 28*28*10;
            cnt/(28*28)
            correctlyClassified/(correctlyClassified+classificationErrors)
                    
            accuracy_array(end+1) = correctlyClassified/(correctlyClassified+classificationErrors);
        end
         
    end
    plot(accuracy_array);
    save('device_nochoice_2_14','accuracy_array','device_index');
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
    device_size_arrary
    device_index
    device_size_arrary-device_index
    device_error
end