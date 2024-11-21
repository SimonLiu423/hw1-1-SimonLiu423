#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

vector<int> merge(const vector<int> &vl, const vector<int> &vr) {
  int n = vl.size() + vr.size();
  vector<int> res(n);
  int l = 0, r = 0;
  for (; l < vl.size() && r < vr.size();) {
    if (vl[l] < vr[r]) {
      res[l + r] = vl[l];
      ++l;
    } else {
      res[l + r] = vr[r];
      ++r;
    }
  }

  for (; l < vl.size(); ++l) {
    res[l + r] = vl[l];
  }
  for (; r < vr.size(); ++r) {
    res[l + r] = vr[r];
  }

  return res;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n, arr[70005];
  if (rank == 0) {
    string filename;
    cin >> filename;
    ifstream fin(filename);

    fin >> n;

    for (int i = 0; i < n; ++i) {
      fin >> arr[i];
    }
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  vector<int> sendcounts(world_size, n / world_size);
  vector<int> displs(world_size, 0);
  for (int i = 0; i < world_size; ++i) {
    if (i < n % world_size) {
      ++sendcounts[i];
    }
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  vector<int> loc_arr(sendcounts[rank]);
  MPI_Scatterv(&arr, sendcounts.data(), displs.data(), MPI_INT, loc_arr.data(), sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);
  sort(loc_arr.begin(), loc_arr.end());

  for (int i = world_size / 2; i > 0; i /= 2) {
    if (rank < i) {
      int recvcount;
      MPI_Recv(&recvcount, 1, MPI_INT, rank + i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      vector<int> arr2(recvcount);
      MPI_Recv(arr2.data(), recvcount, MPI_INT, rank + i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // if (i == 4 && rank == 2) {
      //   for (auto &j : loc_arr) {
      //     cout << j << ' ';
      //   }
      //   cout << endl;

      //   for (auto &j : arr2) {
      //     cout << j << ' ';
      //   }
      //   cout << endl << endl;
      // }

      loc_arr = merge(loc_arr, arr2);

      // if (i == 4 && rank == 2) {
      //   for (auto &j : loc_arr) {
      //     cout << j << ' ';
      //   }
      //   // cout << endl;

      //   // for (auto &j : arr2) {
      //   //   cout << j << ' ';
      //   // }
      // }
    } else {
      // send
      int sendcount = loc_arr.size();
      MPI_Send(&sendcount, 1, MPI_INT, rank - i, 0, MPI_COMM_WORLD);
      MPI_Send(loc_arr.data(), loc_arr.size(), MPI_INT, rank - i, 0, MPI_COMM_WORLD);
      break;
    }
  }

  if (rank == 0) {
    for (auto &i : loc_arr) {
      cout << i << ' ';
    }
  }

  // for (int i = 0; i < world_size; ++i) {
  //   if (rank == i) {
  //     cout << "Rank " << rank << ": ";
  //     for (const auto &i : loc_arr) {
  //       cout << i << ' ';
  //     }
  //     cout << "\n";
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  MPI_Finalize();

  return 0;
}