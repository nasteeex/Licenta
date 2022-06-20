figure(1);
plot(Step, learning_rate);
hold on;
plot(Step, Lossclassification_loss);
plot(Step, Losslocalization_loss);
plot(Step, Lossregularization_loss);
plot(Step, Losstotal_loss);
hold off;
legend('Learning rate', 'Classification loss', 'Localization loss', 'Regularization loss', 'Total loss');
grid on;

figure(2);
plot(Step, learning_rate);
legend('Learning rate');
grid on;

figure(3);
plot(Step, Lossclassification_loss);
legend('Classification loss');
grid on;

figure(4);
plot(Step, Lossclassification_loss);
plot(Step, Losslocalization_loss);
legend('Localization loss');
grid on;

figure(5);
plot(Step, Lossregularization_loss);
legend('Regularization loss');
grid on;

figure(6);
plot(Step, Losstotal_loss);
legend('Total loss');
grid on;