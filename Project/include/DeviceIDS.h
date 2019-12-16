#ifndef DEVICEIDS_H
#define DEVICEIDS_H

    #include "Alloc.h"
    #include "InterpDensSpecies.h"

    struct DeviceIDS {
        FPinterp* rhon_flat;
        FPinterp* rhoc_flat;
        
        FPinterp* Jx_flat;
        FPinterp* Jy_flat;
        FPinterp* Jz_flat;

        FPinterp *pxx_flat;
        FPinterp *pxy_flat;
        FPinterp *pxz_flat;
        FPinterp *pyy_flat;
        FPinterp *pyz_flat;
        FPinterp *pzz_flat;
    };

#endif