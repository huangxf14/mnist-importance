load importance_2.mat
y1 = accuracy_array;
load retrans_2.mat
y2 = accuracy_array;
load noretrans.mat
y3 = accuracy_array;

y1(1)=0.61;
y2(1)=0.61;
y3(1)=0.61;

x=(1:size(y1,2))*10;
plot(x,y1,x,y2,x,y3);
xlabel('Transmission times')
ylabel('Accuracy')

legend('Importance ARQ','Without importance ARQ','No retransmission','Location','best')

saveas(gcf,'Importance.jpg')