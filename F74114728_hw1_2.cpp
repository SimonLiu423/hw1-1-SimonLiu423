#include "mpi.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace std;


struct Point{
    int x, y, id;
};

bool cmp(Point &a, Point &b){
    if(a.x == b.x) return a.y < b.y;
    return a.x < b.x;
}

int cross(Point &a, Point &b, Point &c){
    return (b.x-a.x) * (c.y-a.y) - (b.y-a.y) * (c.x-a.x);
}


void compute_convex_hull(int &convex_size, Point points[], int start_idx, int total_points, int hull_indices[]){
    int hull_size = 0;
    
    for(int i = start_idx; i < start_idx + total_points; i++){
        while(hull_size >= 2 && cross(points[hull_indices[hull_size - 2]], points[hull_indices[hull_size - 1]], points[i]) > 0){
            hull_size--;
        }
        hull_indices[hull_size++] = i;
    }

    int half_idx = hull_size;

    for(int i = start_idx + total_points - 2; i >= start_idx; i--){
        while(hull_size > half_idx && cross(points[hull_indices[hull_size - 2]], points[hull_indices[hull_size - 1]], points[i]) > 0){
            hull_size--;
        }
        hull_indices[hull_size++] = i;
    }

    convex_size = --hull_size;
}

int orientation(Point p, Point q, Point r) {
    int val = cross(p,q,r);
    if (val == 0) return 0; // collinear
    return (val > 0) ? 1 : 2; // counterclockwise or clockwise
}

int next(int idx, int n) {
    return (idx + 1) % n;
}

void compute_lower_tangent(Point points[], const int &l_n, const int &r_n, int l_convex_idx[], int r_convex_idx[], int &lp_idx, int &rp_idx){
    int lid = 0, rid = 0;

    
    for (int i = 1; i < l_n; i++){
        int prev_x = points[l_convex_idx[i-1]].x;
        int cur_x = points[l_convex_idx[i]].x;
        if( cur_x < prev_x ) break;
        lid = i;
    }

    bool done = false;
    while(!done){
        done = true;
        while(orientation(points[l_convex_idx[lid]], points[r_convex_idx[rid]], points[r_convex_idx[(rid-1+r_n)%r_n]]) == 2) {
            rid -= 1;
            if (rid < 0) rid += r_n;
        }
        while(orientation(points[r_convex_idx[rid]], points[l_convex_idx[lid]], points[l_convex_idx[lid+1]]) == 1){
            lid += 1;
            if (lid >= l_n) lid -= l_n;
            done = false;
        }
    }
    lp_idx = lid, rp_idx = rid;
}

void compute_upper_tangent(Point points[], const int &l_n, const int &r_n, int l_convex_idx[], int r_convex_idx[], int &lp_idx, int &rp_idx){
    int lid, rid = 0;

    for (int i = 1; i < l_n; i++){
        int prev_x = points[l_convex_idx[i-1]].x;
        int cur_x = points[l_convex_idx[i]].x;
        if( cur_x < prev_x ) break;
        lid = i;
    }

    bool done = false;
    while(!done){
        done = true;
        while(orientation(points[l_convex_idx[lid]], points[r_convex_idx[rid]], points[r_convex_idx[rid+1]]) == 1) {
            rid += 1;
            if (rid >= r_n) rid -= r_n;
        }
        while(orientation(points[r_convex_idx[rid]], points[l_convex_idx[lid]], points[l_convex_idx[lid-1]]) == 2){
            lid -= 1;
            if (lid < 0) lid += l_n;
            done = false;
        }
    }
    lp_idx = lid, rp_idx = rid;
}

void combine_convex_hulls(Point points[], int &left_size, int &right_size, int left_hull[], int right_hull[]){
    int ll, lr, ul, ur;
    if (left_size == 0 || right_size == 0){
        if(right_size == 0) return;

        left_size = right_size;
        int i = 0;
        while(i < right_size){
            left_hull[i] = right_hull[i];
            i++;
        }
        return;
    }
    compute_lower_tangent(points, left_size, right_size, left_hull, right_hull, ll, lr);
    compute_upper_tangent(points, left_size, right_size, left_hull, right_hull, ul, ur);

    int sz = 0;
    int tmp[12005];
    int i = 0;
    while (i != next(ul, left_size)){
        tmp[sz++] = left_hull[i];
        i++;
    }

    if (ur <= lr){
        i = ur;
        while(i <= lr){
            tmp[sz++] = right_hull[i];
            i++;
        }
    } else {
        i = ur;
        while(i <= lr + right_size){
            tmp[sz++] = right_hull[i % right_size];
            i++;
        }
    }

    if (ll != ul){
        i = ll;
        while(i < left_size){
            tmp[sz++] = left_hull[i];
            i++;
        }
    }
    left_size = sz;
    for(int i = 0; i < sz; i++){
        left_hull[i] = tmp[i];
    }
}

int main(int argc, char *argv[]){
    int n;
    Point points[13000];
    int convex1_idx[13000];
    int convex2_idx[13000];

    int numprocs, myid, n_active;

    MPI_Datatype my_type;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Point tmp;

    MPI_Aint displacement[1];
    MPI_Aint addr;
    MPI_Datatype type[1] = {MPI_INT};
    int block[1] = {3};
    MPI_Get_address(&tmp, &addr);
    MPI_Get_address(&tmp, &displacement[0]);
    displacement[0] = MPI_Aint_diff(displacement[0], addr);

    MPI_Type_create_struct(1, block, displacement, type, &my_type);
    MPI_Type_commit(&my_type);

    if (myid == 0){
        ifstream file;
        string fname;

        getline(cin, fname);
        file.open(fname, fstream::in);

        file >> n;
        for (int i = 0; i < n; i++){
            Point tmp;
            file >> tmp.x >> tmp.y;
            tmp.id = i + 1;
            points[i] = tmp;
        }
        sort(points, points+n, cmp);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(points, n, my_type, 0, MPI_COMM_WORLD);

    int convex_size;
    int start = 0;
    int remain = n;
    n_active = numprocs;

    int blocksize = n / n_active;
    while(blocksize < 3) {
        n_active--;
        blocksize = n / n_active;
    }

    for(int i = 0; i < numprocs; i++){
        if (i == numprocs-1){
            blocksize = remain;
        }if (remain == 0){
            blocksize = 0;
        }else if(remain - blocksize < 3){
            blocksize = remain;
        }
        if(i == myid){
            compute_convex_hull(convex_size, points, start, blocksize, convex1_idx);
            break;
        }
        start += blocksize;
        remain -= blocksize;
    }

    if (myid == 0){
        int i = 1;
        while(i < numprocs){
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(convex2_idx, recv_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            combine_convex_hulls(points, convex_size, recv_size, convex1_idx, convex2_idx);
            i++;
        }
    } else {
        MPI_Send(&convex_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(convex1_idx, convex_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } 
    if(myid == 0) {
        for(int i = 0; i < convex_size; i++){
            cout << points[convex1_idx[i]].id << ' ';
        }
    }
    MPI_Type_free(&my_type);
    MPI_Finalize();
    return 0;
}

