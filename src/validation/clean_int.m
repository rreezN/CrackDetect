function data_int= clean_int(tick,response,tick_int)

% Add offset to multiple data (in interpolant)
ve   = cumsum(ones(size(tick))).*tick*eps;            % Scaled Offset For Non-Zero Elements
ve   = ve + cumsum(ones(size(tick))).*(tick==0)*eps;  % Add Scaled Offset For Zero Elements
vi   = tick + ve;                                     % Interpolation Vector
tick = vi;

data_int = interp1(tick,response,tick_int,'PCHIP');


