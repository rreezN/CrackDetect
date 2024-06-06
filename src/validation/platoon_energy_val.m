%% Authors
%   Asmus Skar 
%   Copyright DTU Sustain, Technical University of Denmark.

% Clear workspace and set plotting flags
clear all, close all, clc

%--------------------------------------------------------------------------
% Define path 
%--------------------------------------------------------------------------
sys.path = 'C:/Users/asska/Documents/ASCH - Matlab/LiRA - Live Road Assessment/LiRA - Static_hdf5';

sys.rt    = 0.30;               % Radius of Renault Zoe tire in [m]
sys.cw    = 1700;               % Renault Zoe long range battery 
sys.fres  = 10;                 % Resampling/downsampling factor
sys.fcut  = 0.04;               % Highpass filter cut-off frequency 
sys.fs    = 100;                % Sampling frequency
sys.dx    = 0.1;                % Spatial resolution
sys.lseg  = 10;                 % Segment length in [m]
sys.nseg  = floor(sys.lseg/sys.dx)+1;   % number of points in segment
ps        = 0.005;              % smoothing parameter (portion of data length) - 0.01 = 1% length of data

taskIds = [16006,16006,16006,16006,16006,16006,16006,16006,16006,16006,...
    16006,16006,16006,16006,16006,16006,16006,16006,16006,...
    16008,16008,16008,16008,16008,16008,16008,16008,16008,...
    16008,16008,16008,16008,16008,16008,16008,16008,...
    16009,16009,16009,16009,16009,16009,16009,...
    16009,16009,16009,16009,16009,16009,16009,...
    16010,16010,16010,16010,16010,16010,16010,...
    16010,16010,16010,16010,16010,16010,16010,...
    16011,16011,16011,16011,16011,16011,16011,...
    16011,16011,16011,16011,16011,16011,16011];
pass = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,...
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,...
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,...
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,...
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,...
        1,2,3,4,5,6,7,8,9,10,11,12,13,14];

npass   = length(taskIds);          % number of passes
clist   = colormap(hsv(npass));     % color passes

for i=1:npass
    task_ID = string(taskIds(i));
    pass_ID = string(pass(i));


    if rem(pass(i),2) == 0
        sys.route = "platoon_CPH1_VH";
    elseif rem(pass(i),2) == 1
        sys.route = "platoon_CPH1_HH";
    end

    [gm, ds_names]  = read_hdf5_platoon(sys.route, "gm", task_ID, pass_ID);
    car.tectrip{i}  = gm.trip_cons(1,:)';   car.ectrip{i} = gm.trip_cons(2,:)';%'-gm.trip_cons(2,1)';     
    car.twhlt{i}    = gm.whl_trq_est(1,:)'; car.whltrq{i}   = smoothdata(gm.whl_trq_est(2,:)','lowess',length(gm.whl_trq_est(2,:))*ps); % Wheel torque 
end

sys.npass = 19;
for i =1:sys.npass
    sys.tripdata{i} = [car.tectrip{i} car.ectrip{i}];
end
sys       = collect_files(sys);
Etrip     = sys.coldata-sys.coldata(1);

for i =1:sys.npass
    sys.tripdata{i} = [car.twhlt{i} car.whltrq{i}];
end
sys       = collect_files(sys);
Wlt       = sys.coldata;

Ewlt = (0.5*(Wlt(1:end-1,2)+Wlt(2:end,2))./sys.rt.*diff(Wlt(:,1)))./3600;

%     etr_point_pos = movsum((0.5*(Wlt(1:end-1,2)+Wlt(2:end,2)).*diff(Wlt(:,1)))./3600,sys.nseg);
%     sys.etrd_pos  = movsum(etr_point_pos,sys.nseg);


figure
hold on
yyaxis left
plot(Etrip(:,1)-Etrip(1,1),(Etrip(:,2)-Etrip(1,2))./1000,'-k','LineWidth',1.5)
yyaxis right
plot(Wlt(1:end-1,1)-Wlt(1,1),cumsum(Ewlt)./1000,':r','LineWidth',1.5)
hold off
