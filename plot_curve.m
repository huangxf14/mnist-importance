
load device_nochoice_2_14.mat
y1 = smoothdata(accuracy_array);
load device_choice_2MAB_14.mat
y2 = smoothdata(accuracy_array);
% load digit_ada_theta5.mat
% y3 = smoothdata(accuracy_array);
% load digit_ada_theta6.mat
% y4 = smoothdata(accuracy_array);
% load digit_ada_theta8.mat
% y5 = smoothdata(accuracy_array);

x=(1:size(y1,2))*10*(2*2);
plot(x,y1,x,y2);
xlabel('Number of data samples')
ylabel('Accuracy')


legend('Randomly select','Importance-aware selection','Location','best')

saveas(gcf,'importance-select.jpg')