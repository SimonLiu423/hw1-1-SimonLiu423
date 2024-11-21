#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int t, n, m, d1, d2;
  vector<int> A, K;
  if (rank == 0) {
    string filename;
    cin >> filename;
    ifstream fin(filename);

    fin >> t >> n >> m;
    A.resize(n * m);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        fin >> A[i * m + j];
      }
    }

    fin >> d1 >> d2;
    K.resize(d1 * d2);
    for (int i = 0; i < d1; ++i) {
      for (int j = 0; j < d2; ++j) {
        fin >> K[i * d2 + j];
      }
    }
  }

  MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&d1, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&d2, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) K.resize(d1 * d2);
  MPI_Bcast(K.data(), d1 * d2, MPI_INT, 0, MPI_COMM_WORLD);

  int buffer_size = m * (d1 - 1) / 2;

  vector<int> sendcounts(world_size, (n / world_size) * m);
  vector<int> displs(world_size, 0);
  for (int i = 0; i < world_size; ++i) {
    if (i < n % world_size) {
      sendcounts[i] += m;
    }
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  vector<int> loc_A(sendcounts[rank] + 2 * buffer_size);
  MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_INT,
               loc_A.data() + buffer_size, sendcounts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  // if (rank == 0) {
  //   for (int i = 0; i < loc_A.size(); ++i) {
  //     cout << loc_A[i] << ' ';
  //     if ((i + 1) % m == 0) {
  //       cout << endl;
  //     }
  //   }
  //   cout << endl;
  // }

  int prev_rank = (rank - 1 + world_size) % world_size,
      next_rank = (rank + 1) % world_size;

  MPI_Request request;
  MPI_Status status;

  while (t--) {
    MPI_Isend(loc_A.data() + buffer_size, buffer_size, MPI_INT, prev_rank, 0,
              MPI_COMM_WORLD, &request);
    MPI_Recv(loc_A.data() + sendcounts[rank] + buffer_size, buffer_size,
             MPI_INT, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&request, &status);

    MPI_Isend(loc_A.data() + sendcounts[rank], buffer_size, MPI_INT, next_rank,
              0, MPI_COMM_WORLD, &request);
    MPI_Recv(loc_A.data(), buffer_size, MPI_INT, prev_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Wait(&request, &status);

    MPI_Isend(loc_A.data() + buffer_size, buffer_size, MPI_INT, prev_rank, 0,
              MPI_COMM_WORLD, &request);
    MPI_Recv(loc_A.data() + sendcounts[rank] + buffer_size, buffer_size,
             MPI_INT, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&request, &status);

    MPI_Isend(loc_A.data() + sendcounts[rank], buffer_size, MPI_INT, next_rank,
              0, MPI_COMM_WORLD, &request);
    MPI_Recv(loc_A.data(), buffer_size, MPI_INT, prev_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Wait(&request, &status);

    // if (t == 1 && rank == 0) {
    //   for (int i = 0; i < loc_A.size(); ++i) {
    //     cout << loc_A[i] << ' ';
    //     if ((i + 1) % m == 0) {
    //       cout << endl;
    //     }
    //   }
    // }

    // convolute
    vector<int> tmp(sendcounts[rank] + 2 * buffer_size);
    for (int i = buffer_size; i < sendcounts[rank] + buffer_size; ++i) {
      int sum = 0;
      for (int j = 0; j < d1; ++j) {
        for (int k = 0; k < d2; ++k) {
          int x = (n + i / m + j - d1 / 2) % n;
          int y = (m + i % m + k - d2 / 2) % m;
          if (x >= 0 && x < n && y >= 0 && y < m) {
            sum += loc_A[x * m + y] * K[j * d2 + k];
          }
        }
      }
      tmp[i] = sum / (d1 * d2);
    }

    loc_A = tmp;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Gather
  MPI_Gatherv(loc_A.data() + buffer_size, sendcounts[rank], MPI_INT, A.data(),
              sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < n * m; ++i) {
      cout << A[i] << ' ';
    }
  }

  // for (int i = 0; i < world_size; ++i) {
  //   if (rank == i) {
  //     cout << "Rank " << rank << ":";
  //     for (int j = 0; j < sendcounts[rank]; ++j) {
  //       cout << loc_A[j + buffer_size] << ' ';
  //     }
  //     cout << endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  MPI_Finalize();
  return 0;
}