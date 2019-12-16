#ifndef DEVICEGRID_H
#define DEVICEGRID_H

    #include "Alloc.h"
    #include "Grid.h"

    struct DeviceGrid {
        FPfield* XN_flat;
        FPfield* YN_flat;
        FPfield* ZN_flat;
    };

#endif