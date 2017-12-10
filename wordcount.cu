#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define SRC_LINES 10000 // 40
#define CMP_LINES 1000 // 4
#define LINE_MAXLEN 13000 // 100
#define BLOCK_SIZE 1000 // 4


/* Print error message and exit with error status. If PERR is not 0,
    display current errno status. */
static void
error_print (int perr, char *fmt, va_list ap)
{
    vfprintf (stderr, fmt, ap);
    if (perr)
        perror (" ");
    else
        fprintf (stderr, "\n");
    exit (1);  
}
     
/* Print error message and exit with error status. */
static void
errf (char *fmt, ...)
{
    va_list ap;
       
    va_start (ap, fmt);
    error_print (0, fmt, ap);
    va_end (ap);
}
     
/* Print error message followed by errno status and exit
    with error code. */
static void
perrf (char *fmt, ...)
{
    va_list ap;
       
    va_start (ap, fmt);
    error_print (1, fmt, ap);
    va_end (ap);
}
/*
static void
cudaErrhnd(cudaError_t err){
    if(err)
        errf("Error: %s", cudaGetErrorString(err));
}
*/
unsigned int getLineAmount(FILE *fp)
{
    int line = 0;
    char c;

    // Extract characters from file and store in character c
    for (c = getc(fp); c != EOF; c = getc(fp)){
        if (c == '\n') // Increment count if this character is newline
            line = line + 1;
    }

    rewind(fp);

    return line;
}

__device__
int cdStrcmp(const char *s1, const char *s2){
    for ( ; *s1 == *s2; s1++, s2++)
	    if (*s1 == '\0')
	        return 0;
    return ((*(unsigned char *)s1 < *(unsigned char *)s2) ? -1 : +1);
}

__global__
void ckCountWord(char *src, char *cmp, int *count) {
	// get next line from sfp
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lineIdx = tid * LINE_MAXLEN;
    int i;
    char *srcWord, *cmpWord;
    
	for (i = 0; i < SRC_LINES; i++) {
        srcWord = &src[i * LINE_MAXLEN];
        cmpWord = &cmp[lineIdx];
        //int isMatch = 1;
        /*
        while(!(*cmpWord == '\0' && *srcWord == '\0')){
            if(*srcWord != *cmpWord){
                if(*srcWord == '\0' || *cmpWord == '\0')
                    if(*srcWord == '\n' || *cmpWord == '\n') break;
                isMatch = 0;
                break;
            } 
            srcWord++;
            cmpWord++;
        }
        */
		if(cdStrcmp(srcWord, cmpWord) == 0) count[tid] += 1;
        
	}
	// is it matched
	// count
}

int main(int argc, char **argv)
{
    if (argc < 3)
        errf("Please fill the input file and comparer");
    
    FILE *sfp = fopen (argv[1], "r");
    FILE *cfp = fopen (argv[2], "r");

    if (!sfp)
        perrf("cannot open file `%s'", argv[1]);
    else if(!cfp)
        perrf("cannot open file `%s'", argv[2]);

    int i,j;

	char *D_srcSec, *H_srcSec, *D_cmpSec, *H_cmpSec;
    int *D_count, *H_count;	
    
    size_t srcSecSize = SRC_LINES * LINE_MAXLEN;
    size_t cmpSecSize = CMP_LINES * LINE_MAXLEN;

	H_srcSec = (char*)malloc(srcSecSize);
	H_cmpSec = (char*)malloc(cmpSecSize);
    H_count = (int*)malloc(CMP_LINES);
	
    cudaMalloc((void**)&D_srcSec, srcSecSize);
    cudaMalloc((void**)&D_cmpSec, cmpSecSize);
    cudaMalloc((void**)&D_count, CMP_LINES);

	// main task

	for(i = 0; i < CMP_LINES; i++){
		if(fgets(&H_cmpSec[i * LINE_MAXLEN], LINE_MAXLEN, cfp) == NULL) break;
    }
    
    cudaMemcpy(D_cmpSec, H_cmpSec, cmpSecSize, cudaMemcpyHostToDevice);

    cudaMemset(D_count,0,CMP_LINES);

    int numSrcSec = getLineAmount(sfp) / SRC_LINES + 1;

    //printf("%d sections.\n", numSrcSec);

	for(i = 0; i < numSrcSec; i++){
		for(j = 0; j < SRC_LINES; j++){
			if(fgets(&H_srcSec[j * LINE_MAXLEN], LINE_MAXLEN, sfp) == NULL) break;
        }
        
        cudaMemcpy(D_srcSec, H_srcSec, srcSecSize, cudaMemcpyHostToDevice);
				
        ckCountWord<<<CMP_LINES/BLOCK_SIZE,BLOCK_SIZE>>>(D_srcSec, D_cmpSec, D_count);

        cudaDeviceSynchronize();

        // cudaGetLastError();

    }

    cudaMemcpy(H_count, D_count, CMP_LINES, cudaMemcpyDeviceToHost);
    
    cudaFree(D_srcSec);
    cudaFree(D_cmpSec);
    cudaFree(D_count);
    free(H_srcSec);
    free(H_cmpSec);
    
    // test output
    for(i=0; i<CMP_LINES; i++){
        printf("%d\n", H_count[i]);
    }
    // modify files

    free(H_count);
    fclose(sfp);
    fclose(cfp);
   
    return 0;

}