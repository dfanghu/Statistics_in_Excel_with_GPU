// cuxl.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "cuxl.h"


// This is an example of an exported variable
CUXL_API int ncuxl=0;

// This is an example of an exported function.
CUXL_API int fncuxl(void)
{
    return 42;
}

// This is the constructor of a class that has been exported.
// see cuxl.h for the class definition
Ccuxl::Ccuxl()
{
    return;
}
