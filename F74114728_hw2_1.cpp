#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>

#include "mpi.h"
using namespace std;

constexpr int MAXN = 1005;
constexpr int MAXM = 1005;
constexpr int MAXD = 1005;

void read_input(const int &my_rank, int &t, int &n, int &m, int &d, int arr[MAXN][MAXM], int ker[MAXD][MAXD]) {
    if (my_rank != 0) return;
    ifstream input_file;
    string fname;

    cin >> fname;
    input_file.open(fname);

    // t
    input_file >> t;

    // n, m
    input_file >> n >> m;

    // A0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            input_file >> arr[i][j];
        }
    }

    // d
    input_file >> d;

    // K
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            input_file >> ker[i][j];
        }
    }
}

void calc_active_procs(const int &n, const int &d, const int &numprocs, int &n_active){
    n_active = numprocs;
    int min_block_size = (d-1)/2;
    while (n/n_active < min_block_size) n_active--;
}

void send_data(const int my_rank, const int &n_active_procs, int &t, const int &n, int &m, int arr[][MAXM], int ker[][MAXD], int &start_r, int &end_r){
    int blocksize = n / n_active_procs;
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ker, MAXD*MAXD, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(arr, MAXN*MAXM, MPI_INT, 0, MPI_COMM_WORLD);
    for(int rk = 0; rk < n_active_procs; rk++){
        int start_row = rk * blocksize;
        int end_row;
        if(rk == n_active_procs - 1) end_row = n;
        else end_row = start_row + blocksize;
        if (rk == 0){
            start_r = start_row;
            end_r = end_row;
        } else {
            MPI_Send(&start_row, 1, MPI_INT, rk, 0, MPI_COMM_WORLD);
            MPI_Send(&end_row, 1, MPI_INT, rk, 0, MPI_COMM_WORLD);
        }
    }
}

void recv_data(int &start_r, int &end_r){
    MPI_Recv(&start_r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&end_r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void print_results(const int arr[][MAXM], const int &n, const int &m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            cout << arr[i][j] << ' ';
        }
    }
}

void print_results(ofstream &file, const int arr[][MAXM], const int &n, const int &m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            file << arr[i][j] << ' ';
        }
        file << endl;
    }
}

void exchange_temp_data(const int &my_rank, const int &numprocs, int stride, const int &n, const int &m, int arr[][MAXM], const int &start_row, const int &end_row){
    vector<MPI_Request> requests(stride*4);
    int prev_rk = (my_rank - 1 + numprocs) % numprocs;
    int next_rk = (my_rank + 1)  % numprocs;
    int req_id = 0;
    for(int i = 0; i < stride; i++){
        MPI_Isend(arr[start_row + i], m, MPI_INT, prev_rk, my_rank+i, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(arr[end_row - stride + i], m, MPI_INT, next_rk, my_rank+i, MPI_COMM_WORLD, &requests[req_id++]);

        int nr_up = (start_row - stride + i + n) % n;
        int nr_do = (end_row + i) % n;
        MPI_Irecv(arr[nr_do], m, MPI_INT, next_rk, next_rk+i, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Irecv(arr[nr_up], m, MPI_INT, prev_rk, prev_rk+i, MPI_COMM_WORLD, &requests[req_id++]);
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void exchange_temp_data(ofstream &file, const int &my_rank, const int &numprocs, int stride, const int &n, const int &m, int arr[][MAXM], const int &start_row, const int &end_row){
    vector<MPI_Request> requests(stride*4);
    int prev_rk = (my_rank - 1 + numprocs) % numprocs;
    int next_rk = (my_rank + 1)  % numprocs;
    int req_id = 0;
    for(int i = 0; i < stride; i++){
        file << "sending row " << start_row + i << " to " << prev_rk << endl;
        file << "sending row " << end_row - stride + i << " to " << next_rk << endl;
        MPI_Isend(arr[start_row + i], m, MPI_INT, prev_rk, my_rank+i, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Isend(arr[end_row - stride + i], m, MPI_INT, next_rk, my_rank+i, MPI_COMM_WORLD, &requests[req_id++]);

        int nr_up = (start_row - stride + i + n) % n;
        int nr_do = (end_row + i) % n;
        file << "receiving row " << nr_up << " from " << prev_rk << endl;
        file << "receiving row " << nr_do << " from " << next_rk << endl;
        MPI_Irecv(arr[nr_do], m, MPI_INT, next_rk, next_rk+i, MPI_COMM_WORLD, &requests[req_id++]);
        MPI_Irecv(arr[nr_up], m, MPI_INT, prev_rk, prev_rk+i, MPI_COMM_WORLD, &requests[req_id++]);
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    file << "Data at proccess " << my_rank << " after exchange: " << endl;
    print_results(file, arr, n, m);
}

int calc_conv(const int arr[][MAXM], const int ker[][MAXD], const int &r, const int &c, const int &d, const int &n, const int &m){
    const int stride = (d-1)/2;
    int sum = 0;
    // cout << "calculating (" << r << "," << c << "): " << endl;
    for(int dr = -stride; dr <= stride; dr++){
        for(int dc = -stride; dc <= stride; dc++){
            int nr = (r + dr + n) % n;
            int nc = (c + dc + n) % n;
            int prod = arr[nr][nc] * ker[stride+dr][stride+dc];
            // cout << "(" << nr << "," << nc << "): " << arr[nr][nc] << " * " << ker[stride+dr][stride+dc] << " = " << prod << endl; 
            sum += prod;

        }
    }
    // return ((double)sum/(d*d))+0.5;
    return sum/(d*d);
}

void send_results(int arr[][MAXM], const int &start_r, const int &end_r, const int &m){
    int r_count = end_r - start_r;
    MPI_Send(&r_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    for(int i = start_r; i < end_r; i++){
        MPI_Send(&arr[i][0], m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void combine_results(const int &numprocs, int arr[][MAXM], int &end_r, const int &n, const int &m){
    for(int i = 1; i < numprocs; i++){
        int r_count;
        MPI_Recv(&r_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int j = 0; j < r_count; j++){
            MPI_Recv(&arr[end_r][0], m, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            end_r++;
        }
    }
}


int main(int argc, char *argv[]) {
    int t, n, m, d;
    int arr[MAXN][MAXM];
    int ker[MAXD][MAXD];
    int start_row, end_row;

    int numprocs, myid;
    int n_active_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // file for debug log
    // ofstream file;
    // file.open(to_string(myid) + ".txt");

    read_input(myid, t, n, m, d, arr, ker);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    calc_active_procs(n, d, numprocs, n_active_procs);
    send_data(myid, n_active_procs, t, n, m, arr, ker, start_row, end_row);

    if (myid < n_active_procs){
        if (myid != 0){
            recv_data(start_row, end_row);
        }
        for(int step = 0; step < t; step++){
            if (step > 0) exchange_temp_data(myid, n_active_procs, (d-1)/2, n, m, arr, start_row, end_row);
            vector<vector<int>> tmp_results((end_row-start_row), vector<int>(m, 0));
            for(int i = start_row; i < end_row; i++){
                for(int j = 0; j < m; j++){
                    tmp_results[i-start_row][j] = calc_conv(arr, ker, i, j, d, n, m);
                }
            }
            for(int i = 0; i < tmp_results.size(); i++){
                memcpy(arr[start_row+i], tmp_results[i].data(), sizeof(int) * m);
            }
            // file << "Data at process " << myid << " after calculation: " << endl;
            // print_results(file, arr, n, m);
        }
        if (myid == 0){
            combine_results(n_active_procs, arr, end_row, n, m);
            print_results(arr, n, m);
        }else {
            send_results(arr, start_row, end_row, m);
        }
    }

    MPI_Finalize();
}