#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "../common.h"

typedef struct Params {
    unsigned long m_size;
    unsigned long n_size;
    unsigned int  n_warmup;
    unsigned int  n_reps;
    unsigned int  n_threads;
}Params;

static void usage() {
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\nGeneral options:"
            "\n    -h        help"
            "\n    -w <W>    # of untimed warmup iterations (default=1)"
            "\n    -e <E>    # of timed repetition iterations (default=3)"
            "\n    -t <T>    # of threads (default=4)"
            "\n"
            "\nBenchmark-specific options:"
            "\n    -m <I>    m_size (default=8192 elements)"
            "\n    -n <I>    n_size (default=8192 elements)"
            "\n");
}

struct Params input_params(int argc, char **argv) {
    struct Params p;
    p.m_size        = 8192;
    p.n_size        = 8192;
    p.n_warmup      = 1;
    p.n_reps        = 3;
    p.n_threads     = 4;

    int opt;
    while((opt = getopt(argc, argv, "ht:m:n:w:e:")) >= 0) {
        switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'm': p.m_size        = atol(optarg); break;
            case 'n': p.n_size        = atol(optarg); break;
            case 'w': p.n_warmup      = atoi(optarg); break;
            case 'e': p.n_reps        = atoi(optarg); break;
            case 't': p.n_threads     = atoi(optarg); break;
            default:
                      fprintf(stderr, "\nUnrecognized option!\n");
                      usage();
                      exit(0);
        }
    }

    return p;
}
#endif
