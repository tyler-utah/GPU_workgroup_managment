__kernel void graph_color( __global int   *row,                        //0
                           __global int   *col,                        //1
                           __global float *node_value,                 //2
                           __global int   *color_array,                //3
                           __global int   *stop1,                      //4
                           __global int   *stop2,                      //5
                           __global float *max_d,                      //6
                           const  int num_nodes,                       //7
                           const  int num_edges) {

  __global int * write_stop = stop1;
  __global int * read_stop = stop2;
  __global int * swap;
  int graph_color = 1;

  while (1) {

    // Original application --- color --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = get_global_id(0); i < num_nodes; i+=get_global_size(0)) {

      // If the vertex is still not colored
      if (color_array[i] == -1) {

        // Get the start and end pointer of the neighbor list
        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1];
        else
          end = num_edges;

        float maximum = -1;

        // Navigate the neighbor list
        for (int edge = start; edge < end; edge++) {

          // Determine if the vertex value is the maximum in the neighborhood
          if (color_array[col[edge]] == -1 && start != end - 1) {
            *write_stop = 1;
            if (node_value[col[edge]] > maximum)
              maximum = node_value[col[edge]];
          }
        }
        // Assign maximum the max array
        max_d[i] = maximum;
      }
    }

    // Two terminating variables allow us to only use 1
    // inter-workgroup barrier and still avoid a data-race
    swap = read_stop;
    read_stop = write_stop;
    write_stop = swap;

    // Original application --- color --- end

    // Inter-workgroup barrier
    global_barrier();

    // Original application --- color2 --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = get_global_id(0); i < num_nodes; i+=get_global_size(0)) {

      // If the vertex is still not colored
      if (color_array[i] == -1) {
        if (node_value[i] > max_d[i])

          // Assign a color
          color_array[i] = graph_color;
      }
    }

    if (*read_stop == 0) {
      break;
    }

    graph_color = graph_color + 1;
    *write_stop = 0;

    // Original application --- color2 --- end

    // Inter-workgroup barrier
    global_barrier();
  }
}