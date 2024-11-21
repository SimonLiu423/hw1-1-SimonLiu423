#include "mpi.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <bitset>
#include <climits>
#include <fstream>
#include <string>
using namespace std;

unsigned int max_m_perm(const int m){
    unsigned int ret = 1;
    for (int i = 0 ; i < m-1; i++){
        ret <<= 1;
        ret |= 1;
    }
    return ret;
}

int test(const int &n, const int &m, unsigned int perm, const int covers[32][32]){
    bitset<32> covered;
    // int cost = 0;

    for(int i = 0; i < 32; i++){
        if(perm & 1){
            for(int j = 0; j < 32; j++){
                int target = covers[i][j];
                if (target == -1) break;
                covered[target] = true;
            }
        }
        perm >>= 1;
    }
    return (covered.count() == n);
    // if (covered.count() == n) return cost;
    // return INT_MAX;
}

int main(int argc, char *argv[]){
    int n, m;
    int tot_count;
    unsigned int last_perm;

    int numprocs, myid;
    double startwtime = 0.0, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    int covers[32][32];


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);

    // cout << "Process " << myid << " of " << numprocs << " is on " << processor_name << endl;

    if (myid == 0){
        startwtime = MPI_Wtime();

        string filePath;
        ifstream inputFile;

        cin >> filePath;
        inputFile.open(filePath);
    
        inputFile >> n >> m;

        for (int i = 0; i < m; i++){
            int k, cost;
            inputFile >> k >> cost;
            for(int j = 0 ; j< k; j++){
                int tmp;
                inputFile >> covers[i][j];
            }
            covers[i][k] = -1;
            // cout << "done input " << i << endl;
        }

        last_perm = max_m_perm(m);
        // cout << "last perm: " << last_perm << endl;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&last_perm, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&covers, 1024, MPI_INT, 0, MPI_COMM_WORLD);

    int count = 0;
    for(unsigned int i = myid; i <= last_perm; i += numprocs){
        // cout << "permutation id: " << i ;
        count += test(n, m, i, covers);
        // cout << " " << count << endl;
    }

    MPI_Reduce(&count, &tot_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0){
        endwtime = MPI_Wtime();
        cout << tot_count;
        // cout << "wall clock time = " << endwtime - startwtime << endl;
    }

    MPI_Finalize();
    return 0;
}
