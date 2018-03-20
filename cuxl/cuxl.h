// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the CUXL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// CUXL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef CUXL_EXPORTS
#define CUXL_API __declspec(dllexport)
#else
#define CUXL_API __declspec(dllimport)
#endif

// This class is exported from the cuxl.dll
class CUXL_API Ccuxl {
public:
	Ccuxl(void);
	// TODO: add your methods here.
};

extern CUXL_API int ncuxl;

CUXL_API int fncuxl(void);
