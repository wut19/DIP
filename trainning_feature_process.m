close all;
clear;
clc;
f_dS = figure(1) ;
f_std = figure(2);
f_D = figure(3);
f_Rl = figure(4);
std_smoke = zeros(1,30);
std_nosmoke = zeros(1,30);

for m = 1:30
   path_s = sprintf('data/smoke%02d.mat',m);
   load(path_s);
   std_smoke(m)=median(std);
    x = 1:length(dS);
    figure(f_dS);plot(x,dS,'r');hold on;
    figure(f_std);plot(x,std,'r');hold on;
    figure(f_D);plot(x,D,'r');hold on;
    figure(f_Rl);plot(x,Rl,'r');hold on;
end

for m = 1:30
   path_ns = sprintf('data/no-smoke%02d.mat',m);
   load(path_ns);
   std_nosmoke(m)=median(std);
    x = 1:length(dS);
    figure(f_dS);plot(x,dS,'b');hold on;
    figure(f_std);plot(x,std,'b');hold on;
    figure(f_D);plot(x,D,'b');hold on;
    figure(f_Rl);plot(x,Rl,'b');hold on;
end
figure(f_std);
x=1:600;
y = 0.5*ones(1,600);
plot(x,y,'g');
figure(f_D);
plot(x,-y,'g');
figure(f_Rl);
y = 0.8*ones(1,600);
plot(x,y,'g');