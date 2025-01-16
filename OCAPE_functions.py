import netCDF4 as nc
import numpy as np
import gsw as gsw
from scipy import optimize as opt
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import interpolate as intp
from matplotlib import cm
import matplotlib.colors as colors


def opening_HR_WOCE(month='Jan', type_data='PYC', dtry=r'/HR_WOCE'):
    '''
    Parameters
    ----------
    > month : month to compute, first three letters starting with a capital letter, see MONTHS below for value;
    > type_data : select data to compute 'BAR' or 'PYC', usefull if all 24 files are in the same directory;
    > dtry : directory name where the files are stored;
    Returns
    -------
    >> SA : 3d-array(depth, latitude, longitude), Absolute Salinity, in fact reference salinity but used as SA;
    >> CT : 3d-array(depth, latitude, longitude), Conservative temperature;
    >> depth : 1d-array(), depth;
    >> lon : 1d-array(), longitude;
    >> lat : 1d-array(), latitude
    '''
    MONTHS=[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']]
    month_index= MONTHS[0].index(month)
    file_num=MONTHS[1][month_index]
    file_name=r'/WAGHC_'+ type_data + '_' + file_num + '_UHAM-ICDC_v1_0_1.nc'
    
    month_data=nc.Dataset(dtry+file_name)
    
    ##Depth, latitude, longitude
    depth=np.array(month_data.variables['depth'])
    lat=np.array(month_data.variables['latitude'])    
    lon=np.array(month_data.variables['longitude'])        
    ##Pressure
    lat_3d, depth_3d, long_3d = np.meshgrid(lat, depth, lon)
    pressure = gsw.conversions.p_from_z(-depth_3d, lat_3d, geo_strf_dyn_height=0, sea_surface_geopotential=0)
    #print('press in Of', pressure[:,0,0])
    
    ##Salinity
    salinity = np.array(month_data.variables['salinity'][0])
    salinity[salinity<-8.]=np.nan
    #SA=gsw.conversions.SA_from_SP(salinity, pressure, long_3d, lat_3d) # This two lines can be change to have SA, but no significant impact on the results
    SA=gsw.conversions.SR_from_SP(salinity)
    
    ##Conservative temperature
    temperature=np.array(month_data.variables['temperature'][0])
    temperature[temperature<-8.]=np.nan
    CT=gsw.conversions.CT_from_t(SA, temperature, pressure)

    return (SA, CT, depth, lon, lat)    
    
    
def opening_annual_Levitus94(dtry=r'/ANNUAL_LEVITUS', file_sal=r'/anual_sal.nc', file_temp=r'/anual_pottemp.nc'):
    '''
    Parameters
    ----------
    > dtry : directory name where the files are stored;
    > file_sal : name of the salinity file in dtry;
    > file_temp : name of the salinity file in dtry;
    Returns
    -------
    >> abs_sal : 3d-array(depth, latitude, longitude), Absolute Salinity, in fact reference salinity but used as SA;
    >> CT_3d : 3d-array(depth, latitude, longitude), Conservative temperature;
    >> depth_AX : 1d-array(), depth;
    >> long_AX : 1d-array(), longitude;
    >> lat_AX : 1d-array(), latitude
    '''
    ##Salinity
    salinity = nc.Dataset(dtry + file_sal)
    #print("salinity")  #units: psu (practical salinity units)  #i think its practical salinity ## Make sure of that
    prac_salinity_3d=np.array(salinity.variables['sal'][:])
    abs_sal_3d=gsw.conversions.SR_from_SP(prac_salinity_3d)

    ##Conservative temperature
    pot_temp= nc.Dataset(dtry + file_temp)
    #print("potential temperature")  #units: celcius  #its in pottential temperature
    pot_temp_3d=np.array(pot_temp.variables['theta'][:])
    CT_3d=gsw.conversions.CT_from_pt(abs_sal_3d, pot_temp_3d)
    
    ##Depth, longitude, latitude
    depth_AX=np.array(pot_temp['Z'][:])
    long_AX=np.array(pot_temp['X'][:])
    lat_AX=np.array(pot_temp['Y'][:])

    return (abs_sal_3d, CT_3d, depth_AX, long_AX, lat_AX)
    
def opening_monthly_Levitus94(month='Jan',dtry=r'/MONTHLY_LEVITUS', file_sal=r'/Salinity.nc', file_temp=r'/potential_temperature.nc'):
    '''
    Parameters
    ----------
    > month : month to compute, first three letters starting with a capital letter, see MONTHS below for value;
    > dtry : directory name where the files are stored;
    > file_sal : name of the salinity file in dtry;
    > file_temp : name of the salinity file in dtry;
    Returns
    -------
    >> abs_sal : 3d-array(depth, latitude, longitude), Absolute Salinity, in fact reference salinity but used as SA;
    >> CT_3d : 3d-array(depth, latitude, longitude), Conservative temperature;
    >> depth_AX : 1d-array(), depth;
    >> long_AX : 1d-array(), longitude;
    >> lat_AX : 1d-array(), latitude
    '''
    MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_index= MONTHS.index(month)
    
    ##Salinity
    salinity = nc.Dataset(dtry + file_sal)
    salinity_4d=np.array(salinity.variables['sal'][:])
    salinity_3d=salinity_4d[month_index]
    abs_sal_3d=gsw.conversions.SR_from_SP(salinity_3d)

    ##Conservative temperature
    potential_temperature = nc.Dataset(dtry + file_temp)
    #print("potential_temperature") #units: celcius
    potential_temperature_4d=np.array(potential_temperature.variables['theta'][:])
    potential_temperature_3d=potential_temperature_4d[month_index]
    CT_3d=gsw.conversions.CT_from_pt(abs_sal_3d, potential_temperature_3d)

    ##depth, latitude, longitude
    depth_AX=np.array(potential_temperature.variables['Z'][:])
    lat_AX=np.array(potential_temperature.variables['Y'][:])
    long_AX=np.array(potential_temperature.variables['X'][:])
    

    return (abs_sal_3d, CT_3d, depth_AX, long_AX, lat_AX)

def regular_pgrid_intp(original_data, PRESS_AX, press_to_interpol, lat_AX, long_AX) :
    """
    Parameters
    ----------
    > original_data : len(press_to_interpol)*len(lat_AX)*len(long_AX) 3d-array(depth, latitude, longitude) of some factor (CT or SA);
    > press_to_interpol : 1d-array()  original irregular pressure, first axis of original_data;
    > PRESS_AX : M 1d-array() regular pressure axis that will be the first axis of interpolated_data;
    > lat_AX : latitude 1d-array, can be a slice of the original data, eg from lat=60 to lat=90;
    > long_AX : longitude 1d-array, can be a slice of the original data;
    Returns
    -------
    >> interpolated_data : M*len(lat_AX)*len(long_AX) 3d-array(depth, latitude, longitude) of the factor (CT or SA)
    """  

    long_len=len(long_AX)
    lat_len=len(lat_AX)
    P_len_M=len(PRESS_AX)
    
    d3fp_matrix=np.zeros(((P_len_M, lat_len, long_len)))
    interpolated_data=np.zeros(((P_len_M, lat_len, long_len)))

    for i in range (long_len):
        for j in range(lat_len):
            #linear interpolation
            d3fp_matrix[:, j, i]=intp.griddata(press_to_interpol, original_data[:, j, i], PRESS_AX, method='linear')

            #Piecewise Cubic Intp Hermite Polynome
            mask_ij= ~np.isnan(original_data[:, j, i])
            p_original=press_to_interpol[mask_ij]
            data_to_intp=original_data[:, j, i][mask_ij]
            if len(p_original)<=1:
                interpolated_data[:, j, i]=intp.griddata(press_to_interpol, original_data[:, j, i], PRESS_AX, method='linear')

            elif len(p_original)>1:
                interpolated_data[:, j, i] = PchipInterpolator(p_original, data_to_intp)(PRESS_AX)
    
    interpolated_data[np.isnan(d3fp_matrix)]=np.nan #Why did i do that ??
    
    return interpolated_data 


    
def OCAPE(P_vec, aS1_vec, cT_vec):
    """
    Parameters
    ----------
    > P_vec : 1d-array() regular pressure vector of the colunm;
    > aS1_vec : 1d-array() absolute salinity of the colunm;
    > cT_vec : 1d-array() conservative temperature of the colunm;

    Returns
    -------
    >> reduced_APE : scalar, maximum potential energy available for a 1d colunm in those conditions, reduced because column average J/kg;
    >> coords : coord[1] : coordinate of parcel coords[0] in the reference state of the column

    """
    
    P_lenght=len(P_vec)
    

    P_calc=np.tile(P_vec, (P_lenght,1)) #M*M matrix, each line is equal to P_vec
    aS1_calc = np.repeat(aS1_vec, P_lenght).reshape(-1, P_lenght)#M*M matrix where each column is equal to aS1_vec, SA of the column (all column are the same)
    cT_calc = np.repeat(cT_vec, P_lenght).reshape(-1, P_lenght)#M*M matrix where each column is equal to cT_vec, CT of the column (all column are the same)

    
    #initial enthalpy
    enthalpy_from_vec=gsw.energy.enthalpy(aS1_vec, cT_vec, P_vec) #M-vect, enthalpy (as,ct,p)
    enthalpy_from_matrix=gsw.energy.enthalpy(aS1_calc, cT_calc, P_calc)#M*M matrix line i column j is the enthalpy of as1_vec_i, cT_vec_i at pressure j, ie hij=h(asi,cti,pj) (diagonal is equal to enthalpy_from_vec)
    enthalpy_calc=np.repeat(enthalpy_from_vec, P_lenght).reshape(-1, P_lenght)#M*M matrix, each column is the enthalpy of the current state (all column are the same)
    
    #Resolution of the linear assignement problem
    coords=opt.linear_sum_assignment(enthalpy_from_matrix)
    iPE=np.mean(enthalpy_calc) #initial potential energy = np.mean(enthalpy_from_vec)
    RPE=0
    for k in range(0,P_lenght):
        RPE += enthalpy_from_matrix[coords[0][k], coords[1][k]]


    reduced_APE=iPE-RPE/P_lenght
    
    return (coords,reduced_APE)
    


def mass_of_colunm(P_vec,AS_vec, cT_vec, delta_long, delta_lat, lat):
    rho_vec=gsw.density.rho(AS_vec, cT_vec, P_vec)  
    z=-gsw.z_from_p(P_vec[-1], lat) 
    E_rad=6367.5*10**3 #Earth radius in m
    #assume earth is a perfect sphere 
    vol=z*(delta_long/360)*(delta_lat/360)*(2*np.pi*E_rad)**2 #don't understand the calculation
    col_mass=np.mean(rho_vec)*vol
    return col_mass 



def colunmreducedOCAPE(ASofP, CTofP, PRESS_AX, lat_AX,long_AX, coordinate):

    """
    Parameters
    ----------
    > ASofP : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), the absolute salinity (g/KG) interpolated on a regular pressure grid;
    > CTofP : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), the conservative temperature (°C) interpolated on a regular pressure grid;
    > PRESS_AX : M 1d-array(), regular pressure axis;
    > lat_AX : latitude 1d-array, can be a slice of the original data, eg from lat=60 to lat=90;
    > long_AX : longitude 1d-array, can be a slice of the original data;
    > coordinate : 'YES' or 'NO', used to save the initial and reference position of all parcels, rarely used because time consuming;
     
    Returns
    ----------
    >> red_APEcs : len(lat_AX)*len(long_AX) 2d-array(), thermobaric energy available from vertical rearangements per unit mass (J/KG);
    >> col_Mass : len(lat_AX)*len(long_AX) 2d-array(), colunm masses;
    >> CT_at_min_APE : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), conservative temperature of the reference state, minimum PE;
    >> AS_at_min_APE : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), absolute salinity of the reference state, minimum PE;
    >> COORDS : COORDS[1] : coordinate of parcel COORDS[0] in the reference state, or np.zeros(10) if coordinate=='NO'
    """
    ntheta=len(ASofP[0,0,:]) #len_longitude_array
    nphi=len(ASofP[0,:,0]) #len_latitude_array

    T_APE=0 #Total APE 
    #APEcs=np.zeros((nphi, ntheta)) it isn't use nor return
    red_APEcs=np.zeros((nphi, ntheta))
    col_Mass=np.zeros((nphi,ntheta))
    T_mass=0
    
    CT_at_min_APE=np.ones(np.shape(CTofP))
    AS_at_min_APE=np.ones(np.shape(ASofP))
    if coordinate=='YES':
        COORDS=np.ones((2,)+np.shape(ASofP))
    else : 
        COORDS=np.zeros(10)
    for theta_i in range (ntheta):
        if int(theta_i%30)==0 :
            print(theta_i/30,'/12')
        for phi_i in range (nphi):#180, 30 if we have cut between 60 and 90
            #print(phi_i)
            # Check if we have data to compute OCAPE
            P_vec=PRESS_AX ## Reminder : press_AX=np.arange(0,press_max,delta_p)
            AS_vec=ASofP[:,phi_i, theta_i] #Abs_Sal at lat_i and long_i, np.array(1*len(P_vec))
            CT_vec=CTofP[:, phi_i, theta_i] # Csvt_temp at lat_i and long_i np.array(1*len(P_vec))
            #if theta_i ==0 and phi_i==0 :
             #   print(AS_vec,CT_vec)
            the_mask =~np.isnan(AS_vec) & ~np.isnan(CT_vec)
            P_vec_r=P_vec[the_mask]
            AS_vec_r=AS_vec[the_mask] 
            CT_vec_r=CT_vec[the_mask]
        
            
            
            if len(P_vec_r)!=0:     #if we have data non 'NaN' (ie we're not on land) , we calculate the column ape and its mass from the data we have
                coords_c,red_APEc=OCAPE(P_vec_r, AS_vec_r, CT_vec_r) #reduce OCAPE
                   
                #Collect data after the rearrangement in order to compare after/before
                CT_vec_r_minAPEc=CT_vec_r[coords_c[1]]
                AS_vec_r_minAPEc=AS_vec_r[coords_c[1]]
                
                # Replace non-NaN values in cons_temp_minAPE with values from CT_vec_r_minAPE in the new order
                cons_temp_minAPE_column=np.ones(len(CT_vec))*np.nan
                cons_temp_minAPE_column[the_mask] = CT_vec_r_minAPEc[:np.sum(the_mask)] # it's the CT of the column after rearangement, ie at minimum PE
                CT_at_min_APE[:, phi_i, theta_i]=cons_temp_minAPE_column
                
                abs_sal_minAPE_column=np.ones(len(AS_vec))*np.nan
                abs_sal_minAPE_column[the_mask] = AS_vec_r_minAPEc[:np.sum(the_mask)]
                AS_at_min_APE[:, phi_i, theta_i]=abs_sal_minAPE_column

                
                column_mass=mass_of_colunm(P_vec_r, AS_vec_r, CT_vec_r, np.abs(long_AX[-2]-long_AX[-1]), np.abs(lat_AX[-2]-lat_AX[-1]), lat_AX[phi_i]) 
                
                #try to "plot" the rearragement
                if coordinate == 'YES' :
                    coord_c_0=np.ones(len(CT_vec))*np.nan
                    coord_c_1=np.ones(len(CT_vec))*np.nan
                    coord_c_0[the_mask]=coords_c[0][:np.sum(the_mask)]
                    coord_c_1[the_mask]=coords_c[1][:np.sum(the_mask)]
                    
                    COORDS[0,:,phi_i, theta_i]=coord_c_0
                    COORDS[1,:,phi_i, theta_i]=coord_c_1


                
            else:
               red_APEc=np.nan #to be confirmed, 0 originnaly
               column_mass=np.nan #same
            
       
        
            T_mass += column_mass
        
       
            T_APE += red_APEc*column_mass
        
            #APEcs[phi_i, theta_i]=red_APEc*col_Mass
            red_APEcs[phi_i, theta_i]=red_APEc
            col_Mass[phi_i, theta_i]=column_mass
    """          
    print('Total APE=',T_APE, 'J')
    APE_bymass=T_APE/T_mass
    print("specific APE=", APE_bymass, "J/KG")
    """
    CT_at_min_APE[CT_at_min_APE == 1] = np.nan
    AS_at_min_APE[AS_at_min_APE == 1] = np.nan
    
    return (red_APEcs,col_Mass,CT_at_min_APE,AS_at_min_APE, COORDS)



def compute_OCAPE(Ctemp_3dz, abs_sal_3dz, depth_AX, latitude_AX, longitude_AX, long_min=-180, long_max=360, lat_min=60, lat_max=90, M=100, coordinate ='NO'):
    """
    Parameters
    ----------
    > Ctemp_3dz : 3d-array(depth_AX,latitude_AX,longitude_AX), conservative temperature from ocean dataset;
    > abs_sal_3dz : 3d-array(depth_AX,latitude_AX,longitude_AX), absolute salinity from ocean dataset;
    > depth_AX : 1d-array(), all original depth;
    > latitude_AX : 1d-array(), all original latitude;
    > longitude_AX : 1d-array(), all original longitude;
    > M : number of vertical parcels;
    > coordinate : 'YES' or 'NO', used to save the initial and reference position of all parcels, rarely used because time consuming;
     
    Returns
    ----------
    >> Ndepth_3d, Nlat_3d, Nlong_3d : 3d-array() of original depth and selected latitudes and longitudes;
    >> Nabs_sal_3dz : *len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), the absolute salinity (g/KG) interpolated on a regular pressure grid;
    >> Npress_3d : 3d-array(), pressure at Ndepth_3d, rarely used because depth axis is not the interpolated one
    >> N_CT_3d : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), the conservative temperature (°C) interpolated on a regular pressure grid;
    >> lat_AX : latitude 1d-array, can be a slice of the original data, eg from lat=60 to lat=90;
    >> long_AX : longitude 1d-array, can be a slice of the original data;
    >> press_to_interpol : 1d-array()  original irregular pressure to interpolate to press_AX
    >> press_AX : M 1d-array(), regular pressure axis;
    >> red_APEcs : len(lat_AX)*len(long_AX) 2d-array(), thermobaric energy available from vertical rearangements per unit mass (J/KG);
    >> col_Mass : len(lat_AX)*len(long_AX) 2d-array(), colunm masses;
    >> CT_at_min_APE : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), conservative temperature of the reference state (minimum PE);
    >> NCtemp_3dp : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), conservative temperature of original data on the interpolated pressure grid;
    >> AS_at_min_APE : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), absolute salinity of the reference state (minimum PE);
    >> Nabs_sal_3dp  : M*len(lat_AX)*len(long_AX) 3d-array(pressure, latitude, longitude), absolute salinity of original data on the interpolated presure grid;
    >> COORDS : COORDS[1] : coordinate of parcel COORDS[0] in the reference state, or np.zeros(10) if coordinate=='NO'
    """
    
    lat_min_i=np.abs(latitude_AX-lat_min).argmin()
    lat_max_i=np.abs(latitude_AX-lat_max).argmin()
    long_min_i=np.abs(longitude_AX-long_min).argmin()
    long_max_i=np.abs(longitude_AX-long_max).argmin()

    lat_AX=latitude_AX[lat_min_i:lat_max_i+1]
    long_AX=longitude_AX[long_min_i:long_max_i+1]
    
    #Slice the original data to conserved only the considered latitudes and longitudes
    N_CT_3d=Ctemp_3dz[:,lat_min_i:lat_max_i+1, long_min_i:long_max_i+1]
    Nabs_sal_3dz=abs_sal_3dz[:,lat_min_i:lat_max_i+1,long_min_i:long_max_i+1]

    #Conversion of depth axis to pressure axis (for WOCE it is weird bc we convert pressure to depth to pressure again, but it is so that LEVITUS and WOCE return and call the same parameters. But it could be change in opening LEVITUS because depth_AX doesn't seem used elsewhere)
    (Nlat_3d, Ndepth_3d, Nlong_3d)=np.meshgrid(lat_AX, depth_AX, long_AX)
    Npress_3d=gsw.conversions.p_from_z(-Ndepth_3d, Nlat_3d, geo_strf_dyn_height=0, sea_surface_geopotential=0)
    press_to_interpol=Npress_3d[:,-1,0]

    
    #Interpolated pressure grid, \Delta p constant, corresponding same mass parcels
    press_max=np.nanmax(Npress_3d)
    press_AX=np.linspace(0,press_max, M)
    
    #Interpolation to a regular pressure grid
    NCtemp_3dp=regular_pgrid_intp(N_CT_3d, press_AX, press_to_interpol, lat_AX,long_AX)
    Nabs_sal_3dp=regular_pgrid_intp(Nabs_sal_3dz, press_AX, press_to_interpol, lat_AX, long_AX)

    (red_APEcs,col_Mass,CT_at_min_APE,AS_at_min_APE,COORDS)=colunmreducedOCAPE(Nabs_sal_3dp, NCtemp_3dp, press_AX, lat_AX,long_AX, coordinate)
    
    return (Ndepth_3d, Nlat_3d, Nlong_3d, Nabs_sal_3dz, Npress_3d, N_CT_3d, lat_AX, long_AX, press_to_interpol, press_AX,red_APEcs,col_Mass,CT_at_min_APE,NCtemp_3dp,AS_at_min_APE,Nabs_sal_3dp,COORDS) 
    
def where_land(long_AX, lat_AX, Oceanparam_3d):
    """
    Parameters
    ----------
    > lat_AX : latitude 1d-array, can be a slice of the original data, eg from lat=60 to lat=90;
    > long_AX : longitude 1d-array, can be a slice of the original data;
    > Oceanparam_3d : M*len(lat_AX)*len(long_AX) 3d_array(pressure, latitude, longitude), Ct or AS or redAPE, to identify np.nan corresponding to land for plots
    
    Return
    ----------
    land : 2d-array(latitude, longitude), a value of 1 means there's land at the surface, 0 means there isn't
    """
    nlong_AX=len(long_AX)
    nlat_AX=len(lat_AX)
 
    land=np.zeros((nlat_AX, nlong_AX))
    land[np.isnan(Oceanparam_3d[0])] = 1
    
    return land


def plot_polar_contourboth(values1, values2, azimuths, radii, hemisphere, label_values1='Not specified',size_of_fig=(6.4, 4.8)): 

    #if values2 == Nland, it can be used as mask to eliminate land from values1
    
    if hemisphere=='S':
        radii=90+radii
    if hemisphere=='N': 
        radii=90-radii
    
    values1[values2 == 1] = np.nan #now that i change thing in columnreduced, this line should be useless -> to confirm
    
    """Plot a polar contour plot, with 0 degrees at the North.
 
    Arguments:
 
     * `values` -- A 2d array of values with azimuths (x axis, 2nd index in numpy) and radii (y axis, 1st index in numpy)
     * `azimuths` -- A list of azimuths (in degrees)
     * `radii` -- A list of radii
 
    The shapes of these lists are important, and are designed for a particular
    use case (but should be more generally useful). The values list should be `len(azimuths) * len(zeniths)`
    long with data for the first azimuth for all the radii, then the second azimuth for all the zeniths etc.
  
    """
    theta = np.radians(azimuths)
    radii = np.array(radii)
 
    values1 = np.array(values1)
    #values2 = np.array(values2)
    
 
    theta, r = np.meshgrid(np.radians(azimuths), radii)
    #print(len(r[:,0]), len(r[0,:]))
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=size_of_fig)
    
    if hemisphere=='N':
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
    elif hemisphere=='S':
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    #print((values1))
    #print((values2))
    
    #plt.autumn()
    cax2 = ax.contour(theta, r, values2, colors='black', alpha=1.,linewidths=1.,levels=0)
    cax2 = ax.contourf(theta, r, values2, colors='black', alpha=0.025,levels=0)

    
    # Apply a fancy colormap to the figure
    #cmap = plt.get_cmap('coolwarm')
    cmap = cm.coolwarm
    #plt.set_cmap(cmap)
    
    cax1 = ax.contourf(theta, r, values1,norm=colors.CenteredNorm(), cmap=cmap)
    
    #cax1 = ax.contourf(theta, r, values1, cmap=cmap)

    
    
    cb = fig.colorbar(cax1,label=label_values1)
    
 
    return fig, ax, cax1, cax2




