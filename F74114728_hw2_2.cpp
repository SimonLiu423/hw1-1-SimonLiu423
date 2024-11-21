#include <mpi.h>

#include <climits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <cassert>
#include <fstream>

using namespace std;

#define INF INT_MAX

void print_graph(vector<vector<pair<int,int>>> &graph){
    for(const auto &g : graph){
        for(const auto &e : g){
            cout << "(" << e.first << "," << e.second << "),";
        }
        cout << endl;
    }

}

// Function to read the graph from a file
void readGraphFromFile(vector<vector<pair<int, int>>> &graph, int &numVertices) {
    string filename;
    cin >> filename;

    ifstream infile(filename);

    if (!infile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read the number of vertices
    infile >> numVertices;

    // Initialize graph with infinity (no edges)
    graph.resize(numVertices, vector<pair<int, int>>());

    // Read the edges from the file (x, y, z)
    int x, y, z;
    while (infile >> x >> y >> z) {
        graph[x].push_back(make_pair(y, z));
    }
    // print_graph(graph);
    infile.close();
}

void get_portion(const int &numprocs, const int &numVertices, vector<int> &counts) {
    // Each process will handle a portion of the graph
    int portion = numVertices / numprocs;

    counts.resize(numprocs, 0);

    for (int i = 0; i < numprocs; i++) {
        int s = i * portion;
        int e = (i + 1) * portion;
        counts[i] = max(0, e - s);
        if (i == numprocs - 1) {
            counts[i] = max(0, numVertices - s);
        }
    }
    
    // for(int i = 0; i < counts.size(); i++){
    //     cout << counts[i] << ",";
    // }
    // cout << endl;
}

void print_n_edges(vector<int> &edges){
    cout << "n_edges: ";
    for(const auto &x : edges){
        cout << x << ",";
    }
    cout << endl;
}

void calc_start_idx(const vector<int> &portions, const int &rank, int &start){
    start = 0;
    for(int i = 0; i < rank; i++){
        start += portions[i];
    }
}

void calc_init_dist(vector<vector<pair<int,int>>> &graph, vector<int> &dist, const int &source){
    dist[source] = 0;
    for(const auto &e : graph[source]){
        dist[e.first] = e.second;
    } 
}

void send_data(const int &rank, vector<vector<pair<int, int>>> &graph, vector<int> &dist, int &numVertices, vector<int> &portions, const int &numprocs) {
    // Broadcast the number of vertices to all processes
    MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    get_portion(numprocs, numVertices, portions);
    if (rank == 0) {
        int idx = portions[0];
        dist.resize(numVertices, INF);
        calc_init_dist(graph, dist, 0);
        MPI_Bcast(dist.data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> n_edges;  // number of edges of each node the proccess needs to handle
        vector<int> targets;
        vector<int> costs;

        for(int i = 0; i < numVertices; i++){
            n_edges.push_back(graph[i].size());
        }
        MPI_Bcast(n_edges.data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);

        for(const auto &g : graph){
            for(const auto &e : g){
                targets.push_back(e.first);
                costs.push_back(e.second);
            }
        }

        int tot_edges = targets.size();
        MPI_Bcast(&tot_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(targets.data(), tot_edges, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(costs.data(), tot_edges, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        vector<int> n_edges(numVertices, 0);
        vector<int> targets;
        vector<int> costs;
        int tot_edges;
        dist.resize(numVertices, INF);
        MPI_Bcast(dist.data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(n_edges.data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&tot_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        targets.resize(tot_edges, 0);
        costs.resize(tot_edges, 0);

        MPI_Bcast(targets.data(), tot_edges, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(costs.data(), tot_edges, MPI_INT, 0, MPI_COMM_WORLD);


        graph.resize(numVertices, vector<pair<int,int>>());
        int idx = 0;
        for(int i = 0; i < graph.size(); i++){
            for(int j = 0; j < n_edges[i]; j++){
                graph[i].push_back(make_pair(targets[idx], costs[idx]));
                idx++;
            }
        }
        // print_graph(graph);
    }
}


// Parallel Dijkstra using MPI
void parallelDijkstra(const vector<vector<pair<int,int>>> &graph, const vector<int> &portions, const int &start_idx, vector<int> &dist, const int &numVertices) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // ofstream db_file;
    // db_file.open(to_string(rank) + ".txt");

    vector<bool> sptSet(numVertices, false);  // Shortest path tree set

    sptSet[0] = true;

    // db_file << "portions: " << portions[rank] << endl;
    // db_file << "start_idx: " << start_idx << endl;

    for (int count = 0; count < numVertices - 1; count++) {
        // for (int count = 0; count < 1; count++) {
        // Each process checks its portion for the minimum distance node
        int localMin[] = {INF, -1};

        for (int i = 0; i < portions[rank]; i++) {
            int v = i + start_idx;
            if (!sptSet[v] && dist[v] < localMin[0]) {
                localMin[0] = dist[v];
                localMin[1] = v;
            }
        }
        // db_file << "loop " << count << ": {" << localMin[0] << "," << localMin[1] << "}" << endl;

        int globalMin[2];

        // Use MPI_Allreduce to find the global minimum node
        MPI_Allreduce(localMin, globalMin, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        if(globalMin[1] == -1) break;

        sptSet[globalMin[1]] = true;
        dist[globalMin[1]] = globalMin[0];


        for(const auto &e : graph[globalMin[1]]){
            if(!sptSet[e.first]){
                dist[e.first] = min(dist[e.first], globalMin[0] + e.second);
            }
        }
    }

    // Print the results from process 0
    if (rank == 0) {
        int idx = portions[0];
        for(int i = 1; i < portions.size(); i++){
            vector<int> other_dist(numVertices, 0);
            MPI_Recv(other_dist.data(), numVertices, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 0; j < portions[i]; j++){
                dist[idx] = other_dist[idx];
                idx++;
            }
        }
        for(int i = 0; i < numVertices; i++){
            cout << dist[i] << ' ';
        }
    } else {
        MPI_Send(dist.data(), numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    int numVertices;
    int start_idx;

    int my_rank, numprocs;
    vector<int> portions;
    vector<int> dist;
    vector<vector<pair<int, int>>> graph;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // Root process reads the graph from the file and broadcasts to others
    if (my_rank == 0) {
        readGraphFromFile(graph, numVertices);
    }
    send_data(my_rank, graph, dist, numVertices, portions, numprocs);
    calc_start_idx(portions, my_rank, start_idx);

    // // Run parallel Dijkstra's algorithm
    // int source = 0;  // Set the source node
    parallelDijkstra(graph, portions, start_idx, dist, numVertices);

    MPI_Finalize();
    return 0;
}
