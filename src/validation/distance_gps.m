function ddist = distance_gps(gps)

lat = gps(:,1);
lon = gps(:,2);

dx  = zeros(length(gps(:,1))-1,1);
R   = 6378.137*1e3; % Radius of Earth in m

for i=1:length(dx)
    dLat = (lat(i+1)-lat(i))*pi/180;
    dLon = (lon(i+1)-lon(i))*pi/180;
    a    = sin((dLat/2))^2 + cos(lat(i)*pi/180)*cos(lat(i+1)*pi/180)*(sin(dLon/2))^2;
    c    = 2 * atan(sqrt(a)/sqrt(1-a));
    dx(i)= R*c;
end

ddist = dx;