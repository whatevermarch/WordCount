#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

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

static void
cudaErrhnd(cudaError_t err){
    if(err)
        errf("Error: %s", cudaGetErrorString(err));
}

void replaceFoundWord(FILE *sfp, FILE *ofp, char *marker, unsigned int lineCount, fpos_t startPos)
{
    char *strbuf = (char*)malloc(LINE_MAXLEN);
    int i;

    fsetpos (sfp, &startPos);
    for (i = 0; i < lineCount; i++) 
    {
        strcpy(strbuf, "\0");
        if (fgets(strbuf, LINE_MAXLEN, sfp) != NULL) 
        {
            //linectr++;
            if (marker[i] == 'n') 
            {
                fprintf(ofp, "%s", strbuf);
            } 
            else 
            {
                fprintf(ofp, "%s", "---\n");
            }
        }
    }

}

__device__
int cdStrcmp(const char *s1, const char *s2){
    for ( ; *s1 == *s2; s1++, s2++)
	    if (*s1 == '\0')
	        return 0;
    return ((*(unsigned char *)s1 < *(unsigned char *)s2) ? -1 : +1);
}

__global__
void ckCountWord(char *src, unsigned int srcLineAmount, char *cmp, char *elim, int *count) {
	// get next line from sfp
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lineIdx = tid * LINE_MAXLEN;
    int i;
    char *srcWord, *cmpWord;
	
	for (i = 0; i < srcLineAmount; i++) {
        srcWord = &src[i * LINE_MAXLEN];
        cmpWord = &cmp[lineIdx];

	    if(cdStrcmp(srcWord, cmpWord) == 0){
            elim[i] = 'y'; // mark as found
            count[tid] += 1;
        }
        
	}
}

int main(int argc, char **argv)
{
    if (argc < 3)
        errf("Please fill the input file and comparer");

    time_t begin = time(NULL);

    FILE *sfp = fopen (argv[1], "r");
    FILE *cfp = fopen (argv[2], "r");
    FILE *ofp = fopen ("replace-result.txt", "w");
    FILE *nfp = fopen ("count-result.txt", "w");

    if (!sfp)
        perrf("cannot open file `%s'", argv[1]);
    if (!cfp)
        perrf("cannot open file `%s'", argv[2]);
    if (!ofp)
        perrf("cannot create file `%s'", "replace-result.txt");
    if (!nfp)
        perrf("cannot create file `%s'", "count-result.txt");
    
    int i,j;

	char *D_srcSec, *H_srcSec, *D_elim, *H_elim, *D_cmpSec, *H_cmpSec;
    int *D_count, *H_count;	
    
    size_t srcSecSize = SRC_LINES * LINE_MAXLEN;
    size_t cmpSecSize = CMP_LINES * LINE_MAXLEN;

	printf("Allocating Memory for Host\n");
    H_srcSec = (char*)malloc(srcSecSize);
    H_elim = (char*)malloc(SRC_LINES);
	H_cmpSec = (char*)malloc(cmpSecSize);
    H_count = (int*)malloc(sizeof(int) * CMP_LINES);
	printf("Allocating Memory for Device\n");
    cudaErrhnd(cudaMalloc((void**)&D_srcSec, srcSecSize));
    cudaErrhnd(cudaMalloc((void**)&D_elim, SRC_LINES));
    cudaErrhnd(cudaMalloc((void**)&D_cmpSec, cmpSecSize));
    cudaErrhnd(cudaMalloc((void**)&D_count, sizeof(int) * CMP_LINES));

	// main task
	printf("Initializing Data\n");
	for(i = 0; i < CMP_LINES; i++){
		if(fgets(&H_cmpSec[i * LINE_MAXLEN], LINE_MAXLEN, cfp) == NULL) break;
    }
    
    cudaErrhnd(cudaMemcpy(D_cmpSec, H_cmpSec, cmpSecSize, cudaMemcpyHostToDevice));

    cudaErrhnd(cudaMemset(D_count, 0, sizeof(int) * CMP_LINES));

    int itr = 0;
    fpos_t startPos;

	for(i = 1; i != 0;){
        itr += 1;
	if(itr % 100 == 0) printf("Computing section number %d.\n",itr);
        fgetpos(sfp, &startPos);

		for(j = 0; j < SRC_LINES; j++){
			if(fgets(&H_srcSec[j * LINE_MAXLEN], LINE_MAXLEN, sfp) == NULL){
                i = 0;
                break;
            }
        }

        if(j == 0) break;

        cudaErrhnd(cudaMemset(D_elim, 'n', j));
        
        cudaErrhnd(cudaMemcpy(D_srcSec, H_srcSec, srcSecSize, cudaMemcpyHostToDevice));
				
        ckCountWord<<<CMP_LINES/BLOCK_SIZE,BLOCK_SIZE>>>(D_srcSec, j, D_cmpSec, D_elim, D_count);

        cudaDeviceSynchronize();

        cudaErrhnd(cudaGetLastError());

        cudaErrhnd(cudaMemcpy(H_elim, D_elim, j, cudaMemcpyDeviceToHost));

        replaceFoundWord(sfp, ofp, H_elim, j, startPos);
    }

    cudaErrhnd(cudaMemcpy(H_count, D_count, sizeof(int) * CMP_LINES, cudaMemcpyDeviceToHost));
    
    cudaFree(D_srcSec);
    cudaFree(D_elim);
    cudaFree(D_cmpSec);
    cudaFree(D_count);
    free(H_srcSec);
    free(H_elim);
    
    // test output
/*
    for(i=0; i<CMP_LINES; i++){
        printf("%d\n", H_count[i]);
    }
*/    
    // modify files
    for(i=0; i < CMP_LINES; i++){
        fprintf(nfp, "%d ", H_count[i]);
        fprintf(nfp, "%s", &H_cmpSec[i * LINE_MAXLEN]);
    }

    time_t end = time(NULL);

    fprintf(nfp, "\nComputed in %.6f seconds.\n", difftime(end,begin));


    free(H_cmpSec);
    free(H_count);
    
    fclose(sfp);
    fclose(cfp);
    fclose(ofp);
    fclose(nfp);

    return 0;

}
