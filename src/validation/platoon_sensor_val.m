%% Authors
%   Asmus Skar 
%   Copyright DTU Sustain, Technical University of Denmark.

% Clear workspace and set plotting flags
clear all, close all, clc

%--------------------------------------------------------------------------
% Define path and route name CPH#1
%--------------------------------------------------------------------------
sys.path = 'C:/Users/asska/Documents/ASCH - Matlab/LiRA - Live Road Assessment/LiRA - Static_hdf5';
sys.route = "platoon_CPH1_HH"; 

sys.rt    = 0.30;               % Radius of Renault Zoe tire in [m]
sys.cw    = 1700;               % Renault Zoe long range battery 
sys.fres  = 10;                 % Resampling/downsampling factor
sys.fcut  = 0.04;               % Highpass filter cut-off frequency 
sys.fs    = 10;                 % Sampling frequency
sys.dx    = 0.1;                % Spatial resolution
sys.lseg  = 10;                 % Segment length in [m]
ps        = 0.005;              % smoothing parameter (portion of data length) - 0.01 = 1% length of data

%--------------------------------------------------------------------------
% Get Car data
%--------------------------------------------------------------------------
if strcmp(sys.route,'platoon_CPH1_HH') > 0
    taskIds = [16006,16006,16006,16006,16006,16006,16006,16006,16006,16006,...
               16008,16008,16008,16008,16008,16008,16008,16008,16008,...
               16009,16009,16009,16009,16009,16009,16009,...
               16010,16010,16010,16010,16010,16010,16010,...
               16011,16011,16011,16011,16011,16011,16011];
    pass    = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19,...
               1, 3, 5, 7, 9, 11, 13, 15, 17,...
               1, 3, 5, 7, 9, 11, 13,...
               1, 3, 5, 7, 9, 11, 13,...
               1, 3, 5, 7, 9, 11, 13];
elseif strcmp(sys.route,'platoon_CPH1_VH') > 0
    taskIds = [16006,16006,16006,16006,16006,16006,16006,16006,16006,...
               16008,16008,16008,16008,16008,16008,16008,16008,...
               16009,16009,16009,16009,16009,16009,16009,...
               16010,16010,16010,16010,16010,16010,16010,...
               16011,16011,16011,16011,16011,16011,16011];
    pass    = [2, 4, 6, 8, 10, 12, 14, 16, 18,...
               2, 4, 6, 8, 10, 12, 14, 16,...
               2, 4, 6, 8, 10, 12, 14,...
               2, 4, 6, 8, 10, 12, 14,...
               2, 4, 6, 8, 10, 12, 14];
end

npass   = length(taskIds);          % number of passes
clist   = colormap(hsv(npass));     % color passes

for i=1:npass
    task_ID = string(taskIds(i));
    pass_ID = string(pass(i));

    % Get car trip data
    if strcmp(sys.route,'platoon_CPH1_HH') > 0
        [gm, ds_names] = read_hdf5_platoon(sys.route, "gm", task_ID, pass_ID);
    elseif strcmp(sys.route,'platoon_CPH1_VH') > 0
        [gm, ds_names] = read_hdf5_platoon(sys.route, "gm", task_ID, pass_ID);
    else
        error('insert correct filename')
    end

    car.tgps{i}     = gm.gps(1,:)';                                                                                                     % AutoPi gps times
    car.lat{i}      = gm.gps(2,:)';                                                                                                     % AutoPi gps latitude
    car.lon{i}      = gm.gps(3,:)';                                                                                                     % AutoPi gps longitude
    car.taccrpi{i}  = gm.acc_xyz(1,:)';                                                                                                 % AutoPi acceleration times
    car.xaccrpi{i}  = smoothdata(gm.acc_xyz(2,:)','lowess',length(gm.acc_xyz(2,:))*ps);                                                 % AutoPi x-axis acceleration
    car.yaccrpi{i}  = smoothdata(gm.acc_xyz(3,:)','lowess',length(gm.acc_xyz(3,:))*ps);                                                 % AutoPi y-axis acceleration
    car.zaccrpi{i}  = smoothdata(gm.acc_xyz(4,:)','lowess',length(gm.acc_xyz(4,:))*ps);                                                 % AutoPi z-axis acceleration
    car.tspd{i}     = gm.spd_veh(1,:)';     car.spd{i} = smoothdata(gm.spd_veh(2,:)','lowess',length(gm.spd_veh(2,:))*ps);              % Car speed
    car.todo{i}     = gm.odo(1,:)';         car.odo{i} = gm.odo(2,:)'*1e3+cumsum(gm.f_dist(2,:)')*1e-2;                                 % Car odometer (total distance travelled)
    car.talon{i}    = gm.acc_long(1,:)';    car.alon{i}     = smoothdata(gm.acc_long(2,:)','lowess',length(gm.acc_long(2,:))*ps);       % Car longitudinal acceleration
    car.tatra{i}    = gm.acc_trans(1,:)';   car.atra{i}     = smoothdata(gm.acc_trans(2,:)','lowess',length(gm.acc_trans(2,:))*ps);     % Car transverse acceleration
    car.str_tacc{i} = gm.strg_acc(1,:)';    car.str_acc{i}  = smoothdata(gm.strg_acc(2,:)','lowess',length(gm.strg_acc(2,:))*ps);       % Steering acceleration
    car.str_tpos{i} = gm.strg_pos(1,:)';    car.str_pos{i}  = smoothdata(gm.strg_pos(2,:)','lowess',length(gm.strg_pos(2,:))*ps);       % Steering position
    car.rpm_teng{i} = gm.rpm(1,:)';         car.rpm_eng{i}  = smoothdata(gm.rpm(2,:)','lowess',length(gm.rpm(2,:))*ps);
    car.rpm_tfl{i}  = gm.rpm_fl(1,:)';      car.rpm_fl{i}   = smoothdata(gm.rpm_fl(2,:)','lowess',length(gm.rpm_fl(2,:))*ps);           % Wheel RPM
    car.rpm_tfr{i}  = gm.rpm_fr(1,:)';      car.rpm_fr{i}   = smoothdata(gm.rpm_fr(2,:)','lowess',length(gm.rpm_fr(2,:))*ps);
    car.rpm_trl{i}  = gm.rpm_rl(1,:)';      car.rpm_rl{i}   = smoothdata(gm.rpm_rl(2,:)','lowess',length(gm.rpm_rl(2,:))*ps);
    car.rpm_trr{i}  = gm.rpm_rr(1,:)';      car.rpm_rr{i}   = smoothdata(gm.rpm_rr(2,:)','lowess',length(gm.rpm_rr(2,:))*ps);
    car.prs_tfl{i}  = gm.whl_prs_fl(1,:)';  car.prs_fl{i}   = smoothdata(gm.whl_prs_fl(2,:)','lowess',length(gm.whl_prs_fl(2,:))*ps);   % Wheel tire pressure
    car.prs_tfr{i}  = gm.whl_prs_fr(1,:)';  car.prs_fr{i}   = smoothdata(gm.whl_prs_fr(2,:)','lowess',length(gm.whl_prs_fr(2,:))*ps);
    car.prs_trl{i}  = gm.whl_prs_rl(1,:)';  car.prs_rl{i}   = smoothdata(gm.whl_prs_rl(2,:)','lowess',length(gm.whl_prs_rl(2,:))*ps);
    car.prs_trr{i}  = gm.whl_prs_rr(1,:)';  car.prs_rr{i}   = smoothdata(gm.whl_prs_rr(2,:)','lowess',length(gm.whl_prs_rr(2,:))*ps);
    car.twhlt{i}    = gm.whl_trq_est(1,:)'; car.whltrq{i}   = smoothdata(gm.whl_trq_est(2,:)','lowess',length(gm.whl_trq_est(2,:))*ps); % Wheel torque
    car.ttrac{i}    = gm.trac_cons(1,:)';   car.traccon{i}  = smoothdata(gm.trac_cons(2,:)','lowess',length(gm.trac_cons(2,:))*ps);     % Traction power
    car.tbrk{i}     = gm.brk_trq_elec(1,:)';car.brktrq{i}   = smoothdata(gm.brk_trq_elec(2,:)','lowess',length(gm.brk_trq_elec(2,:))*ps); % Braking

    % Speed distance
    car.dspd{i}    = [0; cumsum(car.spd{i}(2:end).*diff(car.tspd{i}))];

    % Normalize accelerations
    car.xaccrpi{i}  = car.xaccrpi{i} - mean(car.xaccrpi{i});
    car.yaccrpi{i}  = car.yaccrpi{i} - mean(car.yaccrpi{i});
    car.zaccrpi{i}  = car.zaccrpi{i} - mean(car.zaccrpi{i});

    car.atra{i} = car.atra{i} - mean(car.atra{i});
    car.alon{i} = car.alon{i} - mean(car.alon{i});

    % Predict road slope
    alonf           = highpass(car.alon{i},sys.fcut,floor(1./mean(diff(car.talon{i}))));
    beta_acan       = (car.alon{i}-alonf)./9.81; beta_acan(abs(beta_acan) > 0.1) = 0;
    car.beta_can{i} = beta_acan;
    
    %-------------------------------------------------------------------------
    % Interpolate and syncronize data
    %-------------------------------------------------------------------------

    % Resample data to 100 Hz
    time_start     = [car.taccrpi{i}(1);car.tatra{i}(1);car.talon{i}(1);car.tspd{i}(1);car.tgps{i}(1);...
        car.todo{i}(1);car.str_tacc{i}(1);car.str_tpos{i}(1);car.rpm_teng{i}(1);...
        car.rpm_tfl{i}(1);car.rpm_tfr{i}(1);car.rpm_trl{i}(1);car.rpm_trr{i}(1);...
        car.prs_tfl{i}(1);car.prs_tfr{i}(1);car.prs_trl{i}(1);car.prs_trr{i}(1);...
        car.twhlt{i}(1);car.ttrac{i}(1);car.tbrk{i}(1)];
    time_start_max = max(time_start);
    time_end       = [car.taccrpi{i}(end);car.tatra{i}(end);car.talon{i}(end);car.tspd{i}(end);car.tgps{i}(end);...
        car.todo{i}(end);car.str_tacc{i}(end);car.str_tpos{i}(end);car.rpm_teng{i}(end);...
        car.rpm_tfl{i}(end);car.rpm_tfr{i}(end);car.rpm_trl{i}(end);car.rpm_trr{i}(end);...
        car.prs_tfl{i}(end);car.prs_tfr{i}(end);car.prs_trl{i}(end);car.prs_trr{i}(end);...
        car.twhlt{i}(end);car.ttrac{i}(end);car.tbrk{i}(end)];
    time_end_min   = min(time_end);
    t0             = time_start_max;
    tend           = time_end_min-t0;

    sys.dt{i}   = 1/sys.fs;       % time increment
    sys.time{i} = 0:sys.dt{i}:tend;  % time vector
    sys.time{i} = sys.time{i}';

    sys.lat_100hz{i}   = clean_int(car.tgps{i}-t0,car.lat{i},sys.time{i});
    sys.lon_100hz{i}   = clean_int(car.tgps{i}-t0,car.lon{i},sys.time{i});
    sys.vel_100hz{i}   = clean_int(car.tspd{i}-t0,car.spd{i},sys.time{i});
    sys.dis_100hz{i}   = clean_int(car.tspd{i}-t0,car.dspd{i},sys.time{i});
    sys.odo_100hz{i}   = clean_int(car.todo{i}-t0,car.odo{i},sys.time{i});
    sys.axrpi_100hz{i} = clean_int(car.taccrpi{i}-t0,car.xaccrpi{i},sys.time{i});
    sys.ayrpi_100hz{i} = clean_int(car.taccrpi{i}-t0,car.yaccrpi{i},sys.time{i});
    sys.azrpi_100hz{i} = clean_int(car.taccrpi{i}-t0,car.zaccrpi{i},sys.time{i});
    sys.axcan_100hz{i} = clean_int(car.talon{i}-t0,car.alon{i},sys.time{i});
    sys.aycan_100hz{i} = clean_int(car.tatra{i}-t0,car.atra{i},sys.time{i});
    sys.beta_100hz{i}  = clean_int(car.tatra{i}-t0,car.beta_can{i},sys.time{i});
    sys.astr_100hz{i}  = clean_int(car.str_tacc{i}-t0,car.str_acc{i},sys.time{i});
    sys.pstr_100hz{i}  = clean_int(car.str_tpos{i}-t0,car.str_pos{i},sys.time{i});
    sys.rpm_100hz{i}   = clean_int(car.rpm_teng{i}-t0,car.rpm_eng{i},sys.time{i});
    sys.rtfl_100hz{i}  = clean_int(car.rpm_tfl{i}-t0,car.rpm_fl{i},sys.time{i});
    sys.rtfr_100hz{i}  = clean_int(car.rpm_tfr{i}-t0,car.rpm_fr{i},sys.time{i});
    sys.rtrl_100hz{i}  = clean_int(car.rpm_trl{i}-t0,car.rpm_rl{i},sys.time{i});
    sys.rtrr_100hz{i}  = clean_int(car.rpm_trr{i}-t0,car.rpm_rr{i},sys.time{i});
    sys.ptfl_100hz{i}  = clean_int(car.prs_tfl{i}-t0,car.prs_fl{i},sys.time{i});
    sys.ptfr_100hz{i}  = clean_int(car.prs_tfr{i}-t0, car.prs_fr{i},sys.time{i});
    sys.ptrl_100hz{i}  = clean_int(car.prs_trl{i}-t0,car.prs_rl{i},sys.time{i});
    sys.ptrr_100hz{i}  = clean_int(car.prs_trr{i}-t0,car.prs_rr{i},sys.time{i});
    sys.whlt_100hz{i}  = clean_int(car.twhlt{i}-t0,car.whltrq{i},sys.time{i});
    sys.power_100hz{i} = clean_int(car.ttrac{i}-t0,car.traccon{i},sys.time{i});
    sys.brk_100hz{i}   = clean_int(car.tbrk{i}-t0,car.brktrq{i},sys.time{i});

    % Find longest section with zero-speed
    ids0     = find(sys.vel_100hz{i}==0);
    id_split = find(diff(ids0)>1);
    ids0_l   = ids0(id_split); ids0_r = ids0(id_split+1);
    seg0     = sort([min(ids0); ids0_l; ids0_r; max(ids0)]);
    dseg0    = zeros(floor(length(seg0)),1);
    for j=1:length(seg0)-1
        dseg0(j) =  seg0(j+1)-seg0(j);
    end
    dseg = dseg0(1:2:end);
    segm = find(dseg==max(dseg));
    sys.id1{i}  = seg0(segm*2-1);
    sys.id2{i}  = seg0(segm*2);

    % Re-orientation part 1
    alon   = sys.axcan_100hz{i};
    atrans = sys.aycan_100hz{i};
    axpn   = sys.axrpi_100hz{i}*9.81;
    aypn   = sys.ayrpi_100hz{i}*9.81;
    azpn   = sys.azrpi_100hz{i}*9.81;

    % Calculate correlation with CAN accelerations
    pcxr_lon = corrcoef(axpn,alon); pcxl = pcxr_lon(2);
    pcyr_lon = corrcoef(aypn,alon); pcyl = pcyr_lon(2);

    pcxr_tra = corrcoef(axpn,atrans); pcxt = pcxr_tra(2);
    pcyr_tra = corrcoef(aypn,atrans); pcyt = pcyr_tra(2);

    if abs(pcxl) < abs(pcxt) && abs(pcyl) > abs(pcyt)
        if pcxt < 0
            sys.axrpi_100hz{i} = aypn;
            sys.ayrpi_100hz{i} = -axpn;
        else
            sys.axrpi_100hz{i} = aypn;
            sys.ayrpi_100hz{i} = axpn;
        end
    else
        if pcyt < 0
            sys.axrpi_100hz{i} = axpn;
            sys.ayrpi_100hz{i} = -aypn;
        else
            sys.axrpi_100hz{i} = axpn;
            sys.ayrpi_100hz{i} = aypn;
        end
    end

    sys.axcan_100hz{i} = alon;
    sys.aycan_100hz{i} = atrans;
    sys.azrpi_100hz{i} = azpn;

    % Reorientation of AutoPi device (only while parked) - NO NEED! - THERE IS
    % ALSO AN ISSUE WITH THE SCRIPT BELOW

    % reacc = [sys.axrpi_100hz{i} sys.ayrpi_100hz{i} sys.azrpi_100hz{i}];
    %
    % [alpha2,beta2,gamma2]  = axis_reorientation(reacc(sys.id1{i}:sys.id2{i},:));                %alt.: [alpha1,beta1]         = euler_alphabeta(accp);
    % [ax,ay,az,axp,ayp,azp] = bt_orientation_fixed(reacc,alpha2,beta2,gamma2);     %alt.: [ax,ay,az,axp,ayp,azp] = euler_orientation_fixed(accp,alpha1,beta1)
    % sys.axrpi_100hz{i}     = axp; sys.ayrpi_100hz{i} = ayp; sys.azrpi_100hz{i} = azp;
    % sys.ax{i} = ax; sys.ay{i} = ay; sys.az{i} = az;
end

%-------------------------------------------------------------------------
% Validation plots
%-------------------------------------------------------------------------

find(pass==1);

pasno = 34; % select pass

%% Acceleration measurements
figure; title('AutoPi vs. CAN acceleration')
hold on, grid on
% plot(sys.time{pasno},sys.ax{pasno},'--k','LineWidth',1.5)
% hold on
% plot(sys.time{pasno},sys.ay{pasno},':k','LineWidth',1.5)
% hold on
% plot(sys.time{pasno},sys.az{pasno},'-k','LineWidth',1.5)
% hold on
plot(sys.time{pasno},sys.axrpi_100hz{pasno},'--b','LineWidth',1.5)
hold on
plot(sys.time{pasno},sys.ayrpi_100hz{pasno},':b','LineWidth',1.5)
hold on
% plot(sys.time{pasno},sys.azrpi_100hz{pasno},'-b','LineWidth',1.5)
% hold on
plot(sys.time{pasno},sys.axcan_100hz{pasno},'--r','LineWidth',1.5)
hold on
plot(sys.time{pasno},sys.aycan_100hz{pasno},':r','LineWidth',1.5)
hold on
xlabel('Distance [m]')
ylabel('Acceleration [m/s^2]')
hold off

%% Distance /speed / gps / odometer measurements
GPS_distance = [0; cumsum(distance_gps([sys.lat_100hz{pasno} sys.lon_100hz{pasno}]))];

figure; title('Distance_{spd} vs. Odometer vs Distance_{gps}')
hold on, grid on
plot(sys.time{pasno},sys.dis_100hz{pasno},'--b','LineWidth',1.5)
hold on
plot(sys.time{pasno},sys.odo_100hz{pasno}-sys.odo_100hz{pasno}(1),'--r','LineWidth',1.5)
hold on
plot(sys.time{pasno},GPS_distance,'--g','LineWidth',1.5)
xlabel('Time [s]')
ylabel('Distance [m]')
hold off

%% Wheel rpm measurements
V_rpm_fl = 2*pi*sys.rtfl_100hz{pasno}*1/60*sys.rt;
V_rpm_fr = 2*pi*sys.rtfr_100hz{pasno}*1/60*sys.rt;
V_rpm_rl = 2*pi*sys.rtrl_100hz{pasno}*1/60*sys.rt;
V_rpm_rr = 2*pi*sys.rtrr_100hz{pasno}*1/60*sys.rt;

figure; title('SPD_{rpm} vs. SPD')
hold on, grid on
plot(sys.time{pasno},V_rpm_fl,'--b','LineWidth',1.5)
hold on
plot(sys.time{pasno},sys.vel_100hz{pasno},'--r','LineWidth',1.5)
xlabel('Time [s]')
ylabel('Speed [m/s]')
hold off

%% Wheel pressure measurements 
% should increase with time (i.e, with increase in temperature) / should be around 2500 mbar
figure; title('Tire pressure')
hold on, grid on
plot(sys.time{1},sys.ptfl_100hz{1},'--k','LineWidth',1.5)
hold on
plot(sys.time{3},sys.ptfl_100hz{3},':b','LineWidth',1.5)
hold on
plot(sys.time{5},sys.ptfl_100hz{5},'-.r','LineWidth',1.5)
hold on
plot(sys.time{7},sys.ptfl_100hz{7},'--g','LineWidth',1.5)
hold on
plot(sys.time{9},sys.ptfl_100hz{9},'-m','LineWidth',1.5)
hold on
xlabel('Time [s]')
ylabel('Tire pressure [mbar]')
hold off

%% Braking / torque measurements
figure; title('Braking torque vs. Wheel torque')
hold on, grid on
plot(sys.time{pasno},sys.brk_100hz{pasno},'--b','LineWidth',1.5)
hold on
plot(sys.time{pasno},sys.whlt_100hz{pasno},'--r','LineWidth',1.5)
hold on
xlabel('Time [s]')
ylabel('Torque [N/m]')
hold off

%% Energy measurements
% kg*m*m/s2 conversion to [Wh]:3600 sec to hrs

% Parameters for energy components (physical model)
cd  = 0.29;                             % air drag coefficient
rho = 1.225;                            % density of air [kg/m3]
gw  = 9.80665;                          % gravitational acceleration [m/s2]
sw  = 0;                                % wind speed [m/s]
A   = 2.3316;                           % cross-sectional area [m2]
mc  = sys.cw;                           % mass of car [kg]
mp  = 80;                               % mass of passengers [kg]
mt  = mc+mp;                            % total weight of vehcile [kg]
ci0 = 0.05;                             % rolling inertia coefficient
krt = 0.008.*(1+(sys.vel_100hz{pasno}*3.6)./100);      % tire rolling resistance coefficient

Fmtrp   = sys.power_100hz{pasno}./sys.vel_100hz{pasno}; % Force from traction instant consumption
E_mtrp  = 1/3600*sys.lseg*Fmtrp ;                       % Energy [Wh] per x meter 

Fwhlt   = sys.whlt_100hz{pasno}./sys.rt;                % Force from wheel torque
E_whlt  = 1/3600.*sys.lseg.*Fwhlt;                      % Energy [Wh] per x meter 
    
Fc      = mt.*gw.*sys.beta_100hz{pasno};                % Gravity force from slope
E_slope = 1/3600.*Fc.*sys.lseg;                         % Energy [Wh] per x meter

Facc    = mt.*sys.axcan_100hz{pasno}-Fc;                % Force from inertia
E_acc   = 1/3600.*sys.lseg.*Facc;                       % kg*m*m/s2 onversion to [Wh]:3600 sec to hrs

Fbrk    = (sys.brk_100hz{pasno}./sys.rt);               % Braking force
E_brk   = 1/3600*sys.lseg*Fbrk;                         % Energy [Wh] per x meter

Fdrag   = 0.5.*rho.*A.*cd.*(sys.vel_100hz{pasno}+sw).^2;% Aerodynamic drag force
E_drag  = 1/3600.*sys.lseg.*Fdrag;                      % Energy [Wh] per x meter

Ftire   = mt.*gw.*krt;                                  % Tire rolling resistance force
E_tire  = 1/3600.*sys.lseg.*Ftire;                      % Energy [Wh] per x meter

Fmodel = Fc + Facc + Fdrag + Ftire;

id_out = find(sys.vel_100hz{pasno} < 5);
pltime = sys.time{pasno}; 
pltime(id_out) = []; 
Fmtrp(id_out) = [];
Fwhlt(id_out) = [];
Fmodel(id_out) = [];

figure; title('Traction power force vs. Wheel torque force')
hold on, grid on
plot(pltime,Fmtrp,'sk','MarkerSize',2.5,'LineWidth',0.5)
hold on
plot(pltime,Fwhlt,'ob','MarkerSize',2.5,'LineWidth',0.5)
hold on
plot(pltime,Fmodel,'*r','MarkerSize',2.5,'LineWidth',0.5)
hold on
legend({'traction sensor','wheel torque sensor','physical model'},...
    'Location','NorthEast', 'FontSize',9)
xlabel('Time [s]')
ylabel('Force [N]')
hold off
