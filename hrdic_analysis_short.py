
import matplotlib.pyplot as plt
import numpy as np


class DeformationMap():
    """ A class for importing Davis displacement data with
    methods for returning deformation maps

    Requires: path and filename
    Returns: a deformation map object

    Usage:

    deformation_map=deformation_map('path','filename')

    """

    def __init__(self,path,fname) :

        self.path=path
        self.fname=fname
        self.data=np.loadtxt(self.path+self.fname,skiprows=1)
        self.xc=self.data[:,0] #x coordinates
        self.yc=self.data[:,1] #y coordinates
        self.xd=self.data[:,2] #x displacement
        self.yd=self.data[:,3] #y displacement

        self.xdim=(self.xc.max()-self.xc.min()
                  )/min(abs((np.diff(self.xc))))+1 #size of map along x
        self.ydim=(self.yc.max()-self.yc.min()
                  )/max(abs((np.diff(self.yc))))+1 #size of map along y
        self.x_map=self._map(self.xd,self.ydim,self.xdim) #u (displacement component along x)
        self.y_map=self._map(self.yd,self.ydim,self.xdim) #v (displacement component along x)
        self.du11=self._grad(self.x_map)[1]#du11
        self.du22=self._grad(self.y_map)[0]#du22
        self.du12=self._grad(self.x_map)[0]#du12
        self.du21=self._grad(self.y_map)[1]#du21

        self.max_shear=np.sqrt((((self.du11-self.du22)/2.)**2)
                               + ((self.du12+self.du21)/2.)**2)# max shear component
        self.mapshape=np.shape(self.max_shear)

    def _map(self,data_col,ydim,xdim):
        data_map=np.reshape(np.array(data_col),(int(ydim),int(xdim)))
        return data_map

    def _grad(self,data_map) :
        grad_step=min(abs((np.diff(self.xc))))
        data_grad=np.gradient(data_map,grad_step,grad_step)
        return data_grad

class DeformationMapFast():
    """ A class for importing Davis displacement data with
    methods for returning deformation maps. This is the fast
    version, which reads the python binary file from the data.
    Use davis_text_to_bin(fname) to create binary file.

    Requires: path and filename
    Returns: a deformation map object

    Usage:

    deformation_map=deformation_map('path','filename')

    """

    def __init__(self,path,fname) :

        self.path=path
        self.fname=fname
        self.data=np.load(self.path+self.fname)
        self.xc=self.data[:,0] #x coordinates
        self.yc=self.data[:,1] #y coordinates
        self.xd=self.data[:,2] #x displacement
        self.yd=self.data[:,3] #y displacement

        self.xdim=(self.xc.max()-self.xc.min()
                  )/min(abs((np.diff(self.xc))))+1 #size of map along x
        self.ydim=(self.yc.max()-self.yc.min()
                  )/max(abs((np.diff(self.yc))))+1 #size of map along y
        self.x_map=self._map(self.xd,self.ydim,self.xdim) #u (displacement component along x)
        self.y_map=self._map(self.yd,self.ydim,self.xdim) #v (displacement component along x)
        self.du11=self._grad(self.x_map)[1]#du11
        self.du22=self._grad(self.y_map)[0]#du22
        self.du12=self._grad(self.x_map)[0]#du12
        self.du21=self._grad(self.y_map)[1]#du21

        self.max_shear=np.sqrt((((self.du11-self.du22)/2.)**2)
                               + ((self.du12+self.du21)/2.)**2)# max shear component
        self.mapshape=np.shape(self.max_shear)

    def _map(self,data_col,ydim,xdim):
        data_map=np.reshape(np.array(data_col),(int(ydim),int(xdim)))
        return data_map

    def _grad(self,data_map):
        grad_step=min(abs((np.diff(self.xc))))
        data_grad=np.gradient(data_map,grad_step,grad_step)
        return data_grad

def davis_text_to_bin(fname):
    data=np.loadtxt(fname,skiprows=1)
    fname_out=fname[:-4]
    np.save(fname_out,data)
    print('Created file {:s}.npy.\n'.format(fname_out))

def scrub(fig,fmap,component,colourmap='bwr',cmin=0.0,cmax=0.5):
    if component == 'du11':
        fig.set_data(fmap.du11)
        fig.axes.set_title(r'$\partial u_{1} / \partial x_{1}$')
    if component == 'du22':
        fig.set_data(fmap.du22)
        fig.axes.set_title(r'$\partial u_{2} / \partial x_{2}$')
    if component == 'du12':
        fig.set_data(fmap.du12)
        fig.axes.set_title(r'$\partial u_{2} / \partial x_{1}$')
    if component == 'du21':
        fig.set_data(fmap.du21)
        fig.axes.set_title(r'$\partial u_{1} / \partial x_{2}$')
    if component=='Max shear':
        fig.set_data(fmap.max_shear)
        fig.axes.set_title(r'$\gamma_{eff}$')
    fig.set_clim([cmin,cmax])
    fig.set_cmap(colourmap)
    fig.set_clim([cmin,cmax])
    plt.draw()

def scrub_max_shear_log(fig,fmap,cmin=0.0,cmax=0.5):
    fig.axes.set_title(r'$\gamma_{eff}$')
    fig.set_cmap('viridis')
    fig.set_clim([cmin,cmax])
    fig.set_data(fmap.max_shear)
    plt.draw()
