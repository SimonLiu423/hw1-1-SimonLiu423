#include "mpi.h"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

constexpr int MAX_N = 70000 + 5;

void get_input(int rank, int& n, int* numbers) {
    if (rank == 0) {
        cin >> n;
        for (int i = 0; i < n; i++) {
            cin >> numbers[i];
        }
    }
}

void merge(int* numbers, int left, int mid, int right) {
    // Merge two sorted subarrays in-place
    // left: start of left subarray
    // mid: start of right subarray
    // right: end of right subarray
    vector<int> temp;
    int k = left, l = left, r = mid;
    while (l < mid && r < right) {
        temp.push_back((numbers[l] <= numbers[r]) ? numbers[l++] : numbers[r++]);
    }
    while (l < mid) temp.push_back(numbers[l++]);
    while (r < right) temp.push_back(numbers[r++]);
    for (int i = left; i < right; i++) {
        numbers[i] = temp[i - left];
    }
}

int main(int argc, char** argv) {
    int n;
    int rank, size;
    int numbers[MAX_N];
    int local_numbers[MAX_N];

    int sendcounts[MAX_N];
    int displs[MAX_N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    get_input(rank, n, numbers);
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local array size
    int local_n = n / size;
    int remainder = n % size;
    int local_size = local_n + (rank < remainder ? 1 : 0);
    
    // Calculate send counts and displacements for scatterv
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = n / size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Distribute data
    MPI_Scatterv(numbers, sendcounts, displs, MPI_INT, 
                 local_numbers, local_size, MPI_INT, 
                 0, MPI_COMM_WORLD);

    // Local sort
    sort(local_numbers, local_numbers + local_size);

    // Gather all sorted pieces
    MPI_Gatherv(local_numbers, local_size, MPI_INT,
                numbers, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Final merge on root and output
    if (rank == 0) {
        // Create temporary array for merging
        for (int step = 1; step < size; step *= 2) {
            for (int i = 0; i < size; i += 2 * step) {
                int left = displs[i];
                int mid = (i + step < size) ? displs[i + step] : n;
                int right = (i + 2 * step < size) ? displs[i + 2 * step] : n;
                
                // Merge sorted subarrays using merge() function
                merge(numbers, left, mid, right);
            }
        }

        // Output sorted array
        for (int i = 0; i < n; i++) {
            cout << numbers[i] << " ";
        }
        cout << endl;

    }

    MPI_Finalize();
    return 0;
}