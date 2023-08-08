more off
close all
clear all

numDrops = 10;                           % Number of random user drops to simulate
N_symbol = 13;      %number of OFDM symbols per sub-frame
N_subframe = 10;    %nuber of sub-frames in one frame
sequenceLength = N_symbol*N_subframe;                % Number of channel samples in a track to simulate for each drop

K = 1;                              % Number of users
centerFrequency = 2.53e9;            % Center frequency in Hz
bandwidth = 7.6e6;                     % Bandwidth in Hz
numSubcarriers = 1024;               % Number of sub-carriers
cp_length = 160;

antennaHeight = 25;                  % Antenna height of the bse station in m
antennaSpacing = 10;                % Antenna spacing in multiples of the wave length
M_V = 4;                             % BS, Number of vertical antenna elements
M_H = 1;                             % BS, Number of horizontal antenn elements
M = M_V*M_H;                       % Total number of antennas (factor 2 due to dual polarization)

minDistance = 10;                    % Minimum distance from the base station
maxDistance = 500;                   % Maximum distance from the base station
userHeight = 1.5;                    % Antenna height of the users
N_V = 4;                        % UE, Number of vertical antenna elements
N_H = 1;                        % UE, Number of horizontal antenna elements
N = N_V*N_H;

sectorAngle = 60;                    % Width of the simulated cell sector in deg
sectorAngleRad = sectorAngle/180*pi; % Width of the simulated cell sector in radians

lambda = 3e8/centerFrequency;
speed_km_h = 5; % 30;
speed_m_s = speed_km_h*1000/3600;

save_name = 'channel.mat';

% Scenario
s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 0;                               % Disable progress bars
s.center_frequency = centerFrequency;                   % Set center frequency
s.sample_density = 2;                                   % 2 samples per half-wavelength
s.use_absolute_delays = 0;                              % Include delay of the LOS path

% Layout
l = qd_layout(s);                                       % Create new QuaDRiGa layout

% Base station
l.no_tx = 1;
l.tx_position(3) = antennaHeight;
l.tx_array = qd_arrayant('3gpp-3d', M_V, M_H, centerFrequency, 1, 0, antennaSpacing);

for n=1:M_V
    for nn=1:M_H
        indeces = (n-1)*M_H+nn;
        l.tx_array.element_position(1,indeces) =  (nn)*antennaSpacing*lambda  - lambda/4 - M_H/2*antennaSpacing*lambda;
        l.tx_array.element_position(2,indeces) = 0;
        l.tx_array.element_position(3,indeces) = (n)*antennaSpacing*lambda - lambda/4 - M_V/2*antennaSpacing*lambda;
    end
end

% Users
l.no_rx = K;                                            % Number of users
% l.rx_array = qd_arrayant( 'omni' );                     % Omnidirectional MT antenna
l.rx_array = qd_arrayant('3gpp-3d', N_V, N_H, centerFrequency, 1, 0, antennaSpacing);

for user_idx=1:K
    for n=1:N_V
        for nn=1:N_H
            indeces = (n-1)*N_H+nn;
            l.rx_array(1,user_idx).element_position(1,indeces) =  (nn)*antennaSpacing*lambda  - lambda/4 - N_H/2*antennaSpacing*lambda;
            l.rx_array(1,user_idx).element_position(2,indeces) = 0;
            l.rx_array(1,user_idx).element_position(3,indeces) = (n)*antennaSpacing*lambda - lambda/4 - N_V/2*antennaSpacing*lambda;
        end
    end
end

% Update Map
l.set_scenario('3GPP_3D_UMa_NLOS');

par.minDistance = minDistance;
par.maxDistance = maxDistance;
par.sectorAngleRad = sectorAngleRad;
par.bandwidth = bandwidth;
par.numSubcarriers = numSubcarriers;
par.sequenceLength = sequenceLength;
par.N_symbol = N_symbol;
par.N_subframe = N_subframe;
par.speed_m_s = speed_m_s;
par.userHeight = userHeight;
par.M = M;
par.N = N;
par.M_H = M_H;
par.M_V = M_V;
par.cp_length = cp_length;
par.s=s;

params = cell(1,numDrops);
for n=1:numDrops
    params{1,n} = par;
    params{1,n}.l = l.copy;
end


h = cell(1,numDrops);
for n=1:numDrops
    n
    h(1,n) = genChannelDrop(params{1,n});
end
H = cell2mat(h');
clear h

% clear H
% H: [numDrops, 1, Nt(N), Nr(M), cp_length, num_symbols]
H_out = reshape(H, [numDrops, N, M, par.cp_length, par.sequenceLength]);
% change to [numDrops, 1, Nt(N), Nr(M), cp_length, num_symbols_subframe, num_subframe]
H_out = reshape(H_out, [numDrops, N, M, par.cp_length, par.N_symbol, par.N_subframe]);
% change to (Num_channels, Nt, Nr, num_symbols_subframe, num_subframe, channel_tab)
H = permute(H_out,[1,2,3,6,5,4]);
save(save_name,'H')


function H = genChannelDrop(par)
    %Create tracks
    OFDM_symbol_duration = 1/par.bandwidth * par.numSubcarriers;
    track_distance = OFDM_symbol_duration * (par.sequenceLength-1) * par.speed_m_s;
    for i=1:par.l.no_rx
        name = par.l.track(1,i).name;
        par.l.track(1,i) = qd_track('linear', track_distance);
        par.l.track(1,i).name = name;
        par.l.track(1,i).scenario = '3GPP_3D_UMa_NLOS';
        par.l.track(1,i).set_speed(par.speed_m_s);

    end

    % Add random positions
    distances = sqrt(rand(1,par.l.no_rx)*(par.maxDistance^2 - par.minDistance^2) + par.minDistance^2);
    angles = (2*rand(1,par.l.no_rx)-1)*par.sectorAngleRad;
    par.l.rx_position = [cos(angles).*distances; sin(angles).*distances; par.userHeight*ones(1,par.l.no_rx)];

    % Interpolate positions to get spacial samples
    interpolate_positions( par.l.track, par.s.samples_per_meter )

    for i=1:par.l.no_rx
        a = par.l.track(1,i).initial_position+par.l.track(1,i).positions;
        if sum(abs(atan(a(2,:)./a(1,:))) > par.sectorAngleRad)
            disp('Out of sector angle')
            i
        end
        if sum(sqrt(a(1,:).^2+a(2,:).^2) > par.maxDistance)
            disp('Out of range r')
            i
        end
    end

    % Get channel impulse reponses
    H_raw = par.l.get_channels();

    % Interpolate the channel to have continuous evolution
    for k=1:par.l.no_rx
        dist = par.l.track(1,k).interpolate_movement(OFDM_symbol_duration);
        H_interp(k,1) = H_raw(k).interpolate( dist );
        H_quant(k,1) = H_interp(k,1).quantize_delays(1/par.bandwidth, par.cp_length);
    end

    % Time domian H (1, num_users, UE, BS, CP, num_symbols)
    H_time = zeros(1,par.l.no_rx, par.N, par.M, par.cp_length, par.sequenceLength);
    for k=1:par.l.no_rx
        tap_idx = int8(H_quant(k).delay/(1/par.bandwidth)+1);
        for n=1:par.N
            for m=1:par.M
                for l=1:par.sequenceLength
                    tap_idx_temp = tap_idx(n,m,:,l);
                    tap_idx_temp_rm_extra_delay = tap_idx_temp(tap_idx_temp<par.cp_length);
                    coeff = H_quant(k).coeff(n,m,:,l);
                    coeff_rm_extra_delay = coeff(tap_idx_temp<par.cp_length);
                    H_time(1,k,n,m,tap_idx_temp_rm_extra_delay,l)= coeff_rm_extra_delay;
                end
            end
        end
    end

    % Normalize H_time
    for k=1:par.l.no_rx
        nor = sum(abs(H_time(1,k,:,:,:,:)).^2, 'all');
        H_time_nor(1,k,:,:,:,:) = H_time(1,k,:,:,:,:)/sqrt(nor)*sqrt(par.M*par.sequenceLength);
    end

    % Reshape
    H_time_out = H_time_nor;
    H = {H_time_out};
end
