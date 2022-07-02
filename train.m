close all;
clear ;
clc
for m = 2:30
    fprintf(['training...on ',num2str(m),' video...']);
    SmokeDetection1(m);
end