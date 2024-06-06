%% Authors
%   Asmus Skar 
%   Copyright DTU Sustain, Technical University of Denmark.

% Clear workspace and set plotting flags
clear all, close all, clc

cardata = importdata("gm_segment_005.csv");
p79data = importdata("p79_segment_005.csv");

% Aligned and interpolated measurements
az   = smoothdata((cardata.data(:,5)-mean(cardata.data(:,5)))*9.81,...
    'lowess',length(cardata.data(:,28))*0.0005);
spd  = smoothdata(cardata.data(:,28),'lowess',length(cardata.data(:,28))*0.01);
xcar = cardata.data(:,2);
tcar = cardata.data(:,1)-cardata.data(1,1); 

zp   = (p79data.data(:,7)+p79data.data(:,23))/2*1e-3;
xp79 = (p79data.data(:,1));

% Resample to spatial resolution of 0.05 m
dx   = 0.01;
xs   = xp79(1):dx:xp79(end);
zps  = clean_int(xp79,zp,xs);
ts   = clean_int(xp79,tcar,xs);
% dt   = diff(ts);
% time = [0; cumsum(dt)];

% Filter signal
Lb   = 0.25;                 % tire contact length (m)
j    = max(1,floor(Lb/dx));  % k-point moving average
zpf  = movmean(zps,j);       % Road profile (filtered)
zpfo = clean_int(ts,zpf,tcar); % Sync w/time 

figure; title('Elevation')
plot(xs,zps*1e3,'-b','LineWidth',1.5)
hold on, grid on
plot(xs,zpf*1e3,'--r','LineWidth',1.5)
hold on, grid on
plot(xp79,zpfo*1e3,':g','LineWidth',1.5)
hold on, grid on
legend({'raw signal (time)','filtered signal (spatial)',...
    'filtered signal (time)'},'Location','SouthEast', 'FontSize',9)
xlabel('Chainage [m]')
ylabel('Elevation [mm]')
xlim([23 28])
hold off

sys.zpfm  = zpfo-zpfo(1);
sys.dt    = round(mean(diff(tcar)),3);

% Model parameters (found from controlled test)
ms  = 1750/4;
ktf = 138292.0; sys.K1 = ktf/ms;  % 159305.0; % [N/m]
ksf = 55927.0;  sys.K2 = ksf/ms;  %  66158.4; % [N/m]
csf = 2414.0;   sys.C  = csf/ms;  %   2840.4; % [Ns/m]
muf = 56.7;     sys.U  = muf/ms;  %     46.6; % [kg]

sys.Zu0  = zeros(length(tcar),1);
sys.Zs0  = sys.Zu0;
sys.Zp0  = sys.Zu0;
sys.acc0 = sys.Zu0;

% -------------------------------------------------------------------------
% Run simulation 
% -------------------------------------------------------------------------
K1        = sys.K1;
K2        = sys.K2;
C         = sys.C;
U         = sys.U;
dt        = sys.dt;
Zu        = sys.Zu0;
Zs        = sys.Zs0;
Zs_dot    = sys.acc0;
Zu_dot    = sys.acc0;
Zs_dotdot = sys.acc0;
Zp        = sys.zpfm;

for i = 2:length(Zu)-1
    %dt  = ((time(i)-time(i-1)) + (time(i+1)-time(i)))/2;
    Zu(i+1) = ((dt*C+2)...
        *(dt^2*K1*(Zp(i)-Zu(i))-U*(Zu(i-1)-2*Zu(i))+2*Zs(i)-Zs(i-1))...
        + 2*dt^2*K2*(Zs(i)-Zu(i))+dt*C*(Zu(i-1)-Zs(i-1))+2*Zs(i-1)-4*Zs(i))...
        /(dt*C*(1+U)+2*U);
    Zs(i+1) = dt^2*K1*(Zp(i)-Zu(i))-U*(Zu(i+1)-2*Zu(i)+Zu(i-1))+2*Zs(i)-Zs(i-1);
    Zu_dot(i) = (Zu(i+1)-Zu(i-1))/(2*dt);
    Zs_dot(i) = (Zs(i+1)-Zs(i-1))/(2*dt);
    Zs_dotdot(i) = (Zs(i+1)-2*Zs(i)+Zs(i-1))/dt^2;
end

azsim = Zs_dotdot;

figure;
subplot(3,1,1); title('Speed')
plot(xcar,spd/3.6,'-b','LineWidth',1.5)
hold on, grid on
%xlabel('Chainage [m]')
ylabel('Speed [m/s]')
xlim([0 100])
hold on

subplot(3,1,2); title('Elevation')
plot(xp79,sys.zpfm*1e3,'-b','LineWidth',1.5)
hold on, grid on
%xlabel('Chainage [m]')
ylabel('Elevation [mm]')
xlim([0 100])
hold on

subplot(3,1,3); title('Acceleration')
plot(xcar,az,'-b','LineWidth',1.5)
hold on, grid on
plot(xcar+1.6,azsim,'--r','LineWidth',1.5)
hold on
legend({'raw signal','synthetic signal'},'Location','SouthEast', 'FontSize',9)
xlabel('Chainage [m]')
ylabel('Acceleration [m/s2]')
xlim([0 100])
hold off