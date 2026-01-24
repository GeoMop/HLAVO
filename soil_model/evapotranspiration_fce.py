from math import exp, sqrt

# Reference evapotranspiration [mm/day] according to FAO Penman-Monteith equation
#
# Input parameters:
# n = daylight hours [-]
# T = mean daily air temperature at 2 m height [°C]
# u2 = wind speed at 2 m height [m/s]
# month = month number of actual day [1-12]
# Tmax, Tmin = 10 day or monthly average of maximal/minimal daily temperatures [°C]
# RHmax, RHmin = 10 day or monthly average of maximal/minimal relative humidity [0-1]
#
# Calculation procedure: https://www.fao.org/4/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
def ET0(n, T, u2, month, Tmax, Tmin, RHmax, RHmin):

  # geolocation specific data:
  z = 340 # elevation above sea [m] ... Uhelna
  Ra_monthly = [8.9, 14.4, 22.2, 31.5, 38.5, 41.7, 40.2, 34.4, 25.7, 16.9, 10.2, 7.5] # extraterrestial radiation for 50° Northern latitute
  N_monthly = [ 8.3, 9.8, 11.6, 13.5, 15.2, 16.1, 15.7, 14.3, 12.3, 10.4, 8.7, 7.9 ] # mean daylight hours for 50° Northern latitude

  # vapour pressures
  es = 0.5 * (eT(Tmax) + eT(Tmin)) # saturation vapour pressure
  ea = 0.5 * (eT(Tmin)*RHmax + eT(Tmax)*RHmin) # actual vapour pressure from relative humidity averages

  Delta = 4098*eT(T)/(T+237.3)**2 # slope vapour pressure curve [kPa/°C] - Table 2.4 in https://www.fao.org/4/X0490E/x0490e0j.htm#annex%202.%20meteorological%20tables

  # radiation
  N = N_monthly[month-1] # mean daylight hours
  Ra = Ra_monthly[month-1] # extraterrestial radiation [MJ/m^2/day]
  Rs = (0.25 + 0.5*n/N)*Ra # solar radiation [MJ/m^2/day]
  Rso = (0.75 + 2e-5*z) * Ra
  Rns = 0.77*Rs
  Rnl = 0.5 * 4.903e-9*(Tmax**4 + Tmin**4) * (0.34-0.14*sqrt(ea)) * (1.35*Rs/Rso - 0.35)
  Rn = Rns - Rnl # net radiation at crop surface [MJ/m^2/day]

  G = 0 # soil heat flux density is negligible

  P = 101.3*((293-0.0065*z)/293)**5.26 # atmospheric pressure [kPa] according to simplified ideal gas law - see https://www.fao.org/4/X0490E/x0490e07.htm#atmospheric%20pressure%20(p)
  gamma = 0.665e-3 * P # psychrometric constant [kPa/°C] - https://www.fao.org/4/X0490E/x0490e07.htm#psychrometric%20constant%20(g)

  return (0.408*Delta*(Rn-G)+gamma*900/(T+273)*u2*(es-ea)) / (Delta + gamma*(1+0.34*u2))



# Saturation vapour pressure from temperature
# T = temperature [°C]
#
# Table 2.3 in https://www.fao.org/4/X0490E/x0490e0j.htm#annex%202.%20meteorological%20tables
def eT(T):
  return 0.6108 * exp(17.27*T/(T+237.3))


