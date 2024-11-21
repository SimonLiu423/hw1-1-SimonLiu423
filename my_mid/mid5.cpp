#include <algorithm>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>

#include "mpi.h"
using namespace std;

constexpr int MAX_N = 100 + 5;

struct Point {
    int x, y, id;
    Point(int x, int y, int id) : x(x), y(y), id(id) {}
    Point() : x(0), y(0), id(0) {}
    Point operator-(const Point &p) const {
        return Point(x - p.x, y - p.y, -1);
    }
    int operator^(const Point &p) const {
        return x * p.y - y * p.x;
    }
    bool operator<(const Point &p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
    friend ostream &operator<<(ostream &os, const Point &p) {
        os << '(' << p.x << ',' << p.y << ')';
        return os;
    }
};

int cross(Point p1, Point p2, Point p3);
unsigned int max_m_perm(const int m);

double euc_dist(Point p1, Point p2);

void compute_convex_hull(int &convex_size, Point points[], int total_points, int hull_indices[]);

double mst_cost(Point points[], int indices[], int n, int perm);

void read_input(const int &rank, int &n, Point points[]);

void define_struct(MPI_Datatype *tstype);

void send_data(int &n, Point points[], MPI_Datatype mpi_point_type);

bool is_set_convex(int perm, int indices[], int n);

int main(int argc, char **argv) {
    int n;
    int rank, numprocs;
    int hull_size;
    int hull_indices[MAX_N];
    Point points[MAX_N];

    MPI_Datatype mpi_point_type;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    define_struct(&mpi_point_type);

    read_input(rank, n, points);
    send_data(n, points, mpi_point_type);

    if (rank == 0) {
        int hull_size;
        compute_convex_hull(hull_size, points, n, hull_indices);
        cout << "hull_size: " << hull_size << endl;
        for (int i = 0; i < hull_size; i++) {
            cout << hull_indices[i] << ' ';
            // cout << points[hull_indices[i]].id << ' ';
        }
        cout << endl;
    }
    MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(hull_indices, hull_size, MPI_INT, 0, MPI_COMM_WORLD);

    int last_perm = max_m_perm(n);
    cout << "last_perm: " << last_perm << endl;
    double min_cost = 1e9;

    for (unsigned int i = rank; i <= last_perm; i += numprocs) {
        if (is_set_convex(i, hull_indices, n)) {
            cout << "i: " << i << endl;
            double cost = mst_cost(points, hull_indices, n, i);
            min_cost = min(min_cost, cost);
        }
    }
    cout << min_cost << endl;

    MPI_Type_free(&mpi_point_type);
    MPI_Finalize();

    return 0;
}

bool is_set_convex(int perm, int indices[], int n) {
    for (int i = 0; i < n; i++) {
        if (!(perm & (1 << indices[i]))) {
            return false;
        }
    }
    return true;
}

void send_data(int &n, Point points[], MPI_Datatype mpi_point_type) {
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(points, n, mpi_point_type, 0, MPI_COMM_WORLD);
}

void define_struct(MPI_Datatype *tstype) {
    Point dummy_point;

    const int count = 1;
    int blocklens[count] = {3};
    MPI_Datatype types[1] = {MPI_INT};
    MPI_Aint disps[count];
    MPI_Aint base_address;
    MPI_Get_address(&dummy_point, &base_address);
    MPI_Get_address(&dummy_point, &disps[0]);
    // MPI_Get_address(&dummy_point, &disps[1]);
    disps[0] = MPI_Aint_diff(disps[0], base_address);
    // disps[1] = MPI_Aint_diff(disps[1], base_address);

    MPI_Type_create_struct(count, blocklens, disps, types, tstype);
    MPI_Type_commit(tstype);
}

int cross(Point p1, Point p2, Point p3) {
    return (p2 - p1) ^ (p3 - p1);
}

unsigned int max_m_perm(const int m) {
    unsigned int ret = 1;
    for (int i = 0; i < m - 1; i++) {
        ret <<= 1;
        ret |= 1;
    }
    return ret;
}

double euc_dist(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

double mst_cost(Point points[], int indices[], int n, int perm) {
    double tot_cost = 0;
    vector<bool> visited(n, false);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    pq.push({0, indices[0]});

    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();
        if (visited[u]) continue;
        visited[u] = true;

        tot_cost += cost;
        for (int v = 0; v < n; v++) {
            if (!visited[v] && (perm & (1 << v))) {
                cout << "u: " << u << " v: " << v << endl;
                pq.push({euc_dist(points[indices[u]], points[indices[v]]), v});
            }
        }
    }
    cout << "tot_cost: " << tot_cost << endl;
    return tot_cost;
}

void compute_convex_hull(int &convex_size, Point points[], int total_points, int hull_indices[]) {
    int hull_size = 0;

    for (int i = 0; i < total_points; i++) {
        while (hull_size >= 2 && cross(points[hull_indices[hull_size - 2]], points[hull_indices[hull_size - 1]], points[i]) > 0) {
            hull_size--;
        }
        hull_indices[hull_size++] = i;
    }
    int half_idx = hull_size;

    for (int i = total_points - 2; i >= 0; i--) {
        while (hull_size > half_idx && cross(points[hull_indices[hull_size - 2]], points[hull_indices[hull_size - 1]], points[i]) > 0) {
            hull_size--;
        }
        hull_indices[hull_size++] = i;
    }

    convex_size = --hull_size;
}

void read_input(const int &rank, int &n, Point points[]) {
    if (rank == 0) {
        string filePath;
        ifstream inputFile;

        cin >> filePath;
        inputFile.open(filePath);

        inputFile >> n;
        for (int i = 0; i < n; i++) {
            inputFile >> points[i].x >> points[i].y;
            points[i].id = i + 1;
        }
        sort(points, points + n);

        inputFile.close();
    }
}
