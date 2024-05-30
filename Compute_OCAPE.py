import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import OCAPE_functions_modif as AOf #A is the residu of 'Annual'
import gsw as gsw
import cartopy.crs as ccrs


#(abs_sal_3d, CT_3d, depth_AX, longitude_AX, latitude_AX) = AOf.opening_annual_Levitus94(dtry=r'/ANNUAL_LEVITUS', file_sal=r'/anual_sal.nc', file_temp=r'/anual_pottemp.nc')
(abs_sal_3d, CT_3d, depth_AX, longitude_AX, latitude_AX) = AOf.opening_monthly_Levitus94(month='Jan',dtry=r'/MONTHLY_LEVITUS', file_sal=r'/Salinity.nc', file_temp=r'/potential_temperature.nc')
#(abs_sal_3d, CT_3d, depth_AX, longitude_AX, latitude_AX) = AOf.opening_HR_WOCE(month='Jan', type_data='PYC', dtry=r'/HR_WOCE')
##Parameters

##
we_are_here='N' #hemisphere considered
M=200 #Number of vertical parcels
#name=r'/home/users/di223124/HR_WOCE/Apr_M_'+str(M)
#name=r'/storage/silver/metstudent/msc/users_2024/di223124/PYC/All_month_Modif/SP/Jan_M_'+str(M)
name=r'C:\Users\titou\OneDrive\Bureau\LEVITUS\Sep_M_'+str(M)

##

if we_are_here=='N':
    long_min=-180 #-180, 360 so that both LEVITUS and WOCE are computed for all longitudes
    long_max=360
    lat_min=60
    lat_max=90

if we_are_here=='S':
    long_min=-180
    long_max=360
    lat_min=-90
    lat_max=-60


(Ndepth_3d, Nlat_3d, Nlong_3d, Nabs_sal_3d, Npress_3d, N_CT_3d, lat_AX, long_AX, press_to_interpol, press_AX, red_APEcs, col_mass,CT_at_min_APE,NCtemp_3dp,AS_at_min_APE,Nabs_sal_3dp,COORDS)=AOf.compute_OCAPE(CT_3d, abs_sal_3d, depth_AX, latitude_AX, longitude_AX, long_min, long_max, lat_min, lat_max, M, coordinate='NO')
#Total_APE=float(np.nansum((red_APEcs*col_mass)))/float(np.nansum(col_mass))
#print(Total_APE)

(N_press_AX_3d, Nlat_3d_bis, Nlong_3d_bis)=np.meshgrid(press_AX, lat_AX, long_AX)
depth_from_press_AX=-gsw.conversions.z_from_p(N_press_AX_3d, Nlat_3d_bis)[0,:,0]

Nland=AOf.where_land(long_AX, lat_AX, N_CT_3d)


#Save data
np.savez_compressed(name + r'_compressed.npz', abs_sal_3d= abs_sal_3d, CT_3d=CT_3d, depth_AX=depth_AX, longitude_AX=longitude_AX, latitude_AX=latitude_AX, Ndepth_3d=Ndepth_3d, Nlat_3d=Nlat_3d, Nlong_3d=Nlong_3d, Nabs_sal_3d=Nabs_sal_3d, Npress_3d=Npress_3d, N_CT_3d=N_CT_3d, lat_AX=lat_AX, long_AX=long_AX, press_AX=press_AX, depth_from_press_AX=depth_from_press_AX, Nland=Nland, red_APEcs=red_APEcs, col_mass=col_mass, CT_at_min_APE=CT_at_min_APE, NCtemp_3dp=NCtemp_3dp, AS_at_min_APE=AS_at_min_APE, Nabs_sal_3dp=Nabs_sal_3dp, press_to_interpol=press_to_interpol)


##OCAPE

#AOf.plot_polar_contourboth(red_APEcs, Nland, long_AX, lat_AX ,we_are_here)
#plt.title("Anual average South Pole OCAPE from vertical rearangements [J/KG]")
#plt.savefig(r'C:/Users/titou/Essai/Biblio/Thermobaricity_Projects/Python_code/everything_else/LEVITUS_annual_data/Plots/SP_OCAPE_M=800')
#plt.show()

